import re
import joblib
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Download stopwords (only first time)
nltk.download('stopwords')

print("📥 Loading dataset...")

# Load dataset
data = pd.read_csv("data/spam.csv", encoding="latin1")

# Keep only required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'email']

# Preprocessing setup
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [stemmer.stem(word) for word in words if word not in stop_words]

    return " ".join(words)

print("🧹 Cleaning text...")
data['cleaned'] = data['email'].apply(preprocess)

print("🔢 Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(data['cleaned']).toarray()
y = data['label'].map({'ham': 0, 'spam': 1})

print("✂️ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🧠 Training SVM model...")
model = SVC(probability=True)
model.fit(X_train, y_train)

# Accuracy check
y_pred = model.predict(X_test)
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))

print("💾 Saving model...")
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("✅ Training complete!")
print("📁 Files saved in the model/ folder")