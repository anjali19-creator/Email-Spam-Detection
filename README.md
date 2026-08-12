# Email-Spam-Detection-System
Machine Learning project for detecting spam and legitimate emails.

An ML-based email spam detection system that analyzes Gmail emails and classifies them as **Spam, Ham, or Suspicious**.

## 🚀 Features

* 📧 Gmail email integration using IMAP
* 🤖 Machine Learning-based spam detection
* 🧠 NLP text preprocessing
* 🔢 TF-IDF vectorization
* 🎯 SVM classification
* 🚫 Spam detection
* ✅ Ham (legitimate email) detection
* ⚠️ Suspicious email detection
* 📊 Confidence-based classification
* 📈 Web dashboard with email statistics
* ✨ Spam keyword highlighting

## 🛠️ Technologies Used

* **Python**
* **Scikit-learn**
* **NLTK**
* **Pandas**
* **Flask**
* **HTML, CSS, JavaScript**
* **Chart.js**
* **Gmail IMAP**

## 🧠 Machine Learning

The project uses:

* **TF-IDF Vectorizer** for converting email text into numerical features
* **Support Vector Machine (SVM)** for classification
* **NLTK** for stopword removal and Porter stemming

The dataset is split into **80% training and 20% testing data**.

## 🔄 How It Works

```text
Gmail Emails
     ↓
Text Preprocessing
     ↓
Stopword Removal + Stemming
     ↓
TF-IDF Vectorization
     ↓
SVM Model
     ↓
Spam / Ham / Suspicious
     ↓
Web Dashboard
```

The system fetches unread Gmail emails, processes their subject and body, and uses the trained model to classify them.

## 📊 Dashboard

The dashboard displays:

* Total emails analyzed
* Spam emails
* Ham emails
* Suspicious emails
* Classification confidence
* Pie chart of email distribution

## 📁 Project Structure

```text
Email-Spam-Detection-System/
│
├── app.py
├── train_model.py
├── gmail_reader.py
├── email_gui.py
├── index.html
├── result.html
│
├── data/
│   └── spam.csv
│
├── model/
│   ├── spam_model.pkl
│   └── vectorizer.pkl
│
├── static/
│   └── style.css
│
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

Download NLTK stopwords:

```python
import nltk
nltk.download('stopwords')
```

Train the model:

```bash
python train_model.py
```

Run the Flask application:

```bash
python app.py
```

## 🔐 Security Note

This project uses a Gmail **App Password** for email access.

**Never upload your real Gmail password, App Password, API keys, or other sensitive credentials to GitHub.**

## 🎯 Objective

To develop a practical Machine Learning system that can automatically analyze Gmail messages and identify potentially unwanted or suspicious emails.

## 👩‍💻 Author

**Anjali Kumari**
