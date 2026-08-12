from flask import Flask, request, jsonify
from flask_cors import CORS
import imaplib
import email
from email.header import decode_header
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# 🔥 Highlight spam words
def highlight_words(text):
    spam_keywords = ["free", "win", "offer", "sale", "money", "urgent", "click"]

    for word in spam_keywords:
        text = re.sub(f"(?i){word}", f"<mark>{word}</mark>", text)

    return text

@app.route("/check_gmail", methods=["POST"])
def check_gmail():
    try:
        data = request.get_json()
        user_email = data["email"]
        password = data["password"]

        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(user_email, password)
        mail.select("inbox")

        status, messages = mail.search(None, 'UNSEEN')
        email_ids = messages[0].split()

        spam_list = []
        ham_list = []
        suspicious_list = []

        # ✅ latest 10 unread emails
        for e_id in email_ids[-20:]:

            status, msg_data = mail.fetch(e_id, '(RFC822)')

            for response in msg_data:
                if isinstance(response, tuple):

                    msg = email.message_from_bytes(response[1])

                    # Decode subject
                    subject = msg["subject"]
                    if subject:
                        subject, encoding = decode_header(subject)[0]
                        if isinstance(subject, bytes):
                            subject = subject.decode(encoding if encoding else "utf-8", errors="ignore")
                    else:
                        subject = ""

                    # Get body
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                try:
                                    body = part.get_payload(decode=True).decode(errors="ignore")
                                except:
                                    body = ""
                    else:
                        try:
                            body = msg.get_payload(decode=True).decode(errors="ignore")
                        except:
                            body = ""

                    # 🔥 Use subject + body
                    full_text = subject + " " + body

                    cleaned = preprocess(full_text)
                    vector = vectorizer.transform([cleaned]).toarray()

                    prediction = model.predict(vector)[0]

                    prob = model.predict_proba(vector)[0]
                    spam_prob = prob[1]
                    confidence = round(max(prob) * 100, 2)

                    # 🚨 Suspicious logic
                    if 0.4 <= spam_prob <= 0.6:
                        category = "suspicious"
                    elif spam_prob > 0.6:
                        category = "spam"
                    else:
                        category = "ham"

                    # Highlight words
                    subject = highlight_words(subject)

                    mail_data = {
                        "subject": subject,
                        "confidence": confidence
                    }

                    if category == "spam":
                        spam_list.append(mail_data)
                    elif category == "ham":
                        ham_list.append(mail_data)
                    else:
                        suspicious_list.append(mail_data)

                    mail.store(e_id, '+FLAGS', '\\Seen')

        total = len(spam_list) + len(ham_list) + len(suspicious_list)

        spam_percent = round((len(spam_list)/total)*100, 2) if total > 0 else 0
        ham_percent = round((len(ham_list)/total)*100, 2) if total > 0 else 0
        suspicious_percent = round((len(suspicious_list)/total)*100, 2) if total > 0 else 0

        return jsonify({
            "spam": spam_list,
            "ham": ham_list,
            "suspicious": suspicious_list,
            "spam_percent": spam_percent,
            "ham_percent": ham_percent,
            "suspicious_percent": suspicious_percent,
            "total": total
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/")
def home():
    return "Server running..."

if __name__ == "__main__":
    app.run(debug=True)