import imaplib
import email
import joblib
import re
import tkinter as tk
from tkinter import scrolledtext

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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

EMAIL = "your_email@gmail.com"
APP_PASSWORD = "your_app_password"

def check_emails():
    output.delete(1.0, tk.END)

    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(EMAIL, APP_PASSWORD)
    mail.select("inbox")

    status, messages = mail.search(None, 'UNSEEN')
    email_ids = messages[0].split()

    spam_list = []
    ham_list = []

    for e_id in email_ids:
        status, msg_data = mail.fetch(e_id, '(RFC822)')

        for response in msg_data:
            if isinstance(response, tuple):
                msg = email.message_from_bytes(response[1])

                subject = msg["subject"] or ""
                body = ""

                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode(errors="ignore")
                else:
                    body = msg.get_payload(decode=True).decode(errors="ignore")

                full_text = subject + " " + body
                cleaned = preprocess(full_text)

                vector = vectorizer.transform([cleaned]).toarray()

                prediction = model.predict(vector)[0]

                # REAL confidence
                prob = model.predict_proba(vector)[0]
                confidence_percent = max(prob) * 100

                if prediction == 1:
                    spam_list.append((subject, confidence_percent))
                else:
                    ham_list.append((subject, confidence_percent))

                mail.store(e_id, '+FLAGS', '\\Seen')

    total = len(spam_list) + len(ham_list)

    spam_percent = (len(spam_list)/total*100) if total > 0 else 0
    ham_percent = (len(ham_list)/total*100) if total > 0 else 0

    # DISPLAY
    output.insert(tk.END, "🚫 SPAM EMAILS\n", "spam_header")
    for sub, conf in spam_list:
        output.insert(tk.END, f"\nSubject: {sub}\nConfidence: {conf:.2f}%\n", "spam")

    output.insert(tk.END, "\n\n✅ HAM EMAILS\n", "ham_header")
    for sub, conf in ham_list:
        output.insert(tk.END, f"\nSubject: {sub}\nConfidence: {conf:.2f}%\n", "ham")

    output.insert(tk.END, "\n\n📊 SUMMARY\n", "summary")
    output.insert(tk.END, f"\nTotal: {total}\nSpam: {spam_percent:.2f}%\nHam: {ham_percent:.2f}%\n")

root = tk.Tk()
root.title("Email Spam Detector")
root.geometry("700x600")

btn = tk.Button(root, text="Check Emails", command=check_emails, font=("Arial", 14))
btn.pack(pady=10)

output = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Arial", 11))
output.pack(expand=True, fill="both")

output.tag_config("spam", foreground="red")
output.tag_config("ham", foreground="green")
output.tag_config("spam_header", font=("Arial", 14, "bold"), foreground="red")
output.tag_config("ham_header", font=("Arial", 14, "bold"), foreground="green")
output.tag_config("summary", font=("Arial", 14, "bold"))

root.mainloop()