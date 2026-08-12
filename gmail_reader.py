import imaplib
import email
import joblib
import re
import csv
from datetime import datetime

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load model
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# NLP setup
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Text preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# Logging function
def log_email(subject, prediction):
    with open("email_log.csv", mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now(), subject, prediction])

# Gmail credentials
EMAIL = "your_email@gmail.com"
APP_PASSWORD = "your_app_password"

# Connect to Gmail
mail = imaplib.IMAP4_SSL("imap.gmail.com")
mail.login(EMAIL, APP_PASSWORD)
mail.select("inbox")

print("📬 Connected to Gmail...\n")

# Fetch unread emails
status, messages = mail.search(None, 'UNSEEN')
email_ids = messages[0].split()

if not email_ids:
    print("No unread emails found.")

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

            # Combine subject + body
            full_text = subject + " " + body

            # Clean text
            cleaned = preprocess(full_text)

            # Predict
            vector = vectorizer.transform([cleaned]).toarray()
            prediction = model.predict(vector)[0]

            result = "SPAM" if prediction == 1 else "HAM"

            print("📧 Subject:", subject)
            print("🧠 Prediction:", result)
            print("-" * 50)

            # Save to log
            log_email(subject, result)

            # Move spam emails
            if prediction == 1:
                mail.copy(e_id, '[Gmail]/Spam')
                mail.store(e_id, '+FLAGS', '\\Deleted')

            # Mark as read
            mail.store(e_id, '+FLAGS', '\\Seen')

# Apply deletion
mail.expunge()

print("✅ Done processing emails")