from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import string
import spacy
import joblib  # or pickle
import nltk
import pickle

from autocorrect import Speller
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec
import emoji
import re

# Load the spaCy model outside the route function
try:
    nlp_en = spacy.load("en_core_web_sm")
    print("spaCy model loaded successfully.")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    nlp_en = None  # Handle the case where the model fails to load

class SpellingCorrectionTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.lemmatizer = WordNetLemmatizer()
        self.spell=Speller(lang='en')
        return X.apply(self._preprocess_text)

    def _preprocess_text(self, text):
        # Correct Spelling: correct the whole text first
        words = text.split()
        
        corrected_words = [self.spell(word) for word in words]
        text = " ".join(corrected_words)

        # Convert emojis to text
        text = emoji.demojize(text, delimiters=("", ""))

        # Handle contractions
        text = re.sub(r"can't", "can not", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'m", " am", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r"'ll", " will", text)

        # Remove URLs and email addresses
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"\S*@\S*\s?", "", text)

        # Remove special characters
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

        # Remove digits and lowercase
        text = "".join([char.lower() for char in text if not char.isdigit()])

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenize
        words = word_tokenize(text)

        # Lemmatize
        words = [self.lemmatizer.lemmatize(word) for word in words]

        # Remove multiple spaces
        text = re.sub(r"\s+", " ", " ".join(words)).strip()

        # Rejoin words into a single string
        return text


class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model=None, vector_size=100):
        self.model = model
        self.vector_size = vector_size
    
    def fit(self, X, y=None):
        if self.model is None:
            sentences = [text.split() for text in X]
            self.model = Word2Vec(sentences, vector_size=self.vector_size, window=5, min_count=1, workers=4)
        return self
    
    def transform(self, X):
        vectors = []
        for text in X:
            words = text.split()
            word_vectors = [self.model.wv[word] for word in words if word in self.model.wv]
            if word_vectors:
                vectors.append(np.mean(word_vectors, axis=0))
            else:
                vectors.append(np.zeros(self.vector_size))  # Return a zero vector if no words are found
        return np.array(vectors, dtype=np.float32)  # Ensure output is of type float32)


try:
    with open('C:\\Users\\ujjwa_n18433z\\Desktop\\ujjwal\\All Projects\\Project_WoC_7.0_Fake_Review_Detection\\Checkpoint 4\\Extension\\server\\model.pkl', 'rb') as file:
        model = joblib.load(file)
    # model = joblib.load('best_model.pkl')  # Replace with your actual model file
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file not found.")
    model = None
except Exception as e:
    print(e)
    print(f"Error loading model: {type(e).__name__} - {str(e)}")
    print(f"Error loading model: {e}")
    model = None


app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

def is_english_spacy(text, nlp, threshold=0.7):
    if nlp is None:
        print("spaCy model is not loaded, skipping language detection.")
        return True  # Assume English if spaCy model failed to load
    doc = nlp(text)
    english_tokens = [token for token in doc if token.has_vector or not token.is_oov]
    num_english_tokens = len(english_tokens)
    total_tokens = len(doc)
    if total_tokens == 0:
        return False
    percentage_english = num_english_tokens / total_tokens
    return percentage_english >= threshold


@app.route('/analyze', methods=['POST'])
def analyze_review():
    data = request.get_json()
    review_text = data.get('review', '')
    rating = float(data.get('rating', None))
    category = data.get('category', None)
    if category is not None:
        category=re.sub(r"\s+", "_", category)
    else:
        category = "" 

    print("Server received request body:", data)
    print("Server received review for analysis:", review_text, "Rating:", rating, "Category:", category)

    is_non_english = False
    try:
        language_prediction = is_english_spacy(review_text, nlp_en)
        if language_prediction is not None and language_prediction == False: # Inverted the condition
            is_non_english = True
            print("Detected language: Non-English")  # Log when non-English is detected
        else:
            print("Detected language: English (or language detection failed)")
    except Exception as e:
        print(f"Error detecting language: {e}, assuming English.")

    if model is None:
        print("Model not loaded, returning placeholder response.")
        return jsonify({'isFake': False, 'nonEnglish': is_non_english, 'error': 'Model not loaded'}) # Also include nonEnglish

    try:
        # 1. Prepare the input data
        input_data = pd.DataFrame({
            'category': [category],
            'text_': [review_text],  # Name should match what your model was trained on
            'rating': [rating]
        })
        print("Input data:", input_data)
        # 2. Make the prediction
        prediction = model.predict(input_data)
        print("Prediction output", prediction)
        is_fake = bool(prediction[0] == 0)# modify according to your output data
        print("Model output", is_fake)

        # 3. Prepare the response
        return jsonify({'isFake': is_fake, 'nonEnglish': is_non_english})#Also included nonEnglish


    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'isFake': False, 'nonEnglish': is_non_english, 'error': str(e)}), 500  # Also include nonEnglish

if __name__ == '__main__':
    app.run(port=3000, debug=True)