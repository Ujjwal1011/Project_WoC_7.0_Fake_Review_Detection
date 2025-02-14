from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import spacy
import joblib  # or pickle
import pandas as pd  # For creating a DataFrame
import numpy as np
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin


# Load the spaCy model outside the route function
nlp_en = spacy.load("en_core_web_sm")



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

#Load model before app to make model global
# Load the model outside the route function
try:
    model = joblib.load('model.pkl')  # Replace with your actual model file
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file not found.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

def is_english_spacy(text, nlp=nlp_en, threshold=0.7):
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
    rating = data.get('rating', None)
    category = data.get('category', None)

    print("Server received request body:", data)
    print("Server received review for analysis:", review_text, "Rating:", rating, "Category:", category)

    is_non_english = False
    try:
        language_prediction = is_english_spacy(review_text)
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