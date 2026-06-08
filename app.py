import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd # Needed to re-fit the TF-IDF vectorizer

# Download necessary NLTK data (if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Preprocessing functions (copied from the notebook)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def lemmatize_text(text):
    words = text.split()
    lemmas = [lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in words] # Lemmatize as verbs
    return ' '.join(lemmas)

def preprocess_review(text):
    cleaned = clean_text(text)
    processed = tokenize_and_remove_stopwords(cleaned)
    lemmatized = lemmatize_text(processed)
    return lemmatized

# Load the saved model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Recreate and fit the TF-IDF Vectorizer
# This is crucial because the model expects vectorized input
# We need the original data to fit the vectorizer with the same vocabulary
# For deployment, you would typically save the vectorizer as well.
# For this demonstration, we'll re-process a small part of the dataset
# or assume data_cleaned is available (which it isn't in a fresh app.py).
# A more robust solution would be to load the original TF-IDF fitted object.

# For this demonstration, we will load the original dataset to re-fit the vectorizer.
# In a real deployment, you would save and load the fitted vectorizer.

# --- This part assumes IMDB Dataset.csv is available in the same directory as app.py ---
# Or pass it directly from the Colab environment.
# For now, let's simplify and use a placeholder to fit the vectorizer.
# A robust solution would be to save/load the fitted tfidf_vectorizer object directly.

# Placeholder for fitting TF-IDF: This is a critical step and needs the original data.
# In a real scenario, you would save the fitted `tfidf_vectorizer` object along with the model.
# For this Colab deployment, we'll refit it using the original dataset.

# Load the original data to refit the TF-IDF vectorizer
data_original = pd.read_csv('/content/IMDB Dataset.csv')
# Apply the same preprocessing steps to the original data to fit the vectorizer
data_original['cleaned_review'] = data_original['review'].apply(clean_text)
data_original['processed_review'] = data_original['cleaned_review'].apply(tokenize_and_remove_stopwords)
data_original['lemmatized_review'] = data_original['processed_review'].apply(lemmatize_text)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_vectorizer.fit(data_original['lemmatized_review'])
# --- End of TF-IDF refitting section ---


# Streamlit app layout
st.title("IMDB Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive/negative).")

# Text input from user
user_input = st.text_area("Your Review:", "Type your review here...")

if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess the user input
        preprocessed_input = preprocess_review(user_input)

        # Transform the preprocessed input using the fitted TF-IDF vectorizer
        # Handle case where input might result in empty string after preprocessing
        if preprocessed_input.strip() == '':
            st.warning("Please enter a valid review for analysis.")
        else:
            input_vectorized = tfidf_vectorizer.transform([preprocessed_input])

            # Make prediction
            prediction = model.predict(input_vectorized)

            # Map prediction to sentiment label
            sentiment = "Positive" if prediction[0] == 1 else "Negative"

            st.success(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter a review to analyze.")
