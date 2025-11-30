import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download stopwords (first run only)
nltk.download("stopwords")

# Load dataset
df = pd.read_csv("tweets.csv")

# Text Preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)

# Vectorization
cv = CountVectorizer()
X = cv.fit_transform(df["clean_text"])
y = df["sentiment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate Model
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("\n==============================")
print(" Twitter Sentiment Analysis ")
print("==============================")
print("Model Accuracy:", accuracy)
print("==============================\n")

# Predict custom tweet
def predict_sentiment(text):
    cleaned = clean_text(text)
    vector = cv.transform([cleaned])
    return model.predict(vector)[0]

# Test prediction
sample = "I love this product! Amazing experience."
print("Sample Tweet:", sample)
print("Predicted Sentiment:", predict_sentiment(sample))
