import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# downloads (cached idea recommended later)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------------- Preprocessing ----------------
def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def preprocess(text):
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t, pos=wordnet.VERB) for t in tokens]
    return " ".join(tokens)

# ---------------- Load saved objects ----------------
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# ---------------- Streamlit UI ----------------
st.title("IMDB Sentiment Analysis")

review = st.text_area("Enter review")

if st.button("Predict"):
    if review.strip():
        processed = preprocess(review)
        vector = tfidf.transform([processed])   # ✅ no dataset needed
        pred = model.predict(vector)

        sentiment = "Positive" if pred[0] == 1 else "Negative"
        st.success(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter a review")
