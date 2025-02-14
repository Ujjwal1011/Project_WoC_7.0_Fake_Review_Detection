# Fake Review Detection Project

This project aims to detect fake reviews using various machine learning techniques. It includes data preprocessing, feature engineering (TF-IDF, Word2Vec, POS tagging), and model training with pipelines.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Data](#data)
- [Code Structure](#code-structure)
- [Pipelines](#pipelines)
- [Models](#models)

## Introduction

The proliferation of fake reviews online poses a significant challenge to consumers and businesses alike. This project leverages natural language processing (NLP) and machine learning (ML) to identify deceptive reviews. The project explores several feature extraction and modeling techniques to build an effective fake review detection system.

## Dependencies

Ensure you have the following dependencies installed:

```bash
pip install pandas scikit-learn nltk gensim joblib emoji autocorrect matplotlib
```

Additionally, download the required NLTK data:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

## Data

The dataset used in this project is `fakeReviewData.csv`, which includes the following columns:

- **category**: Product category.
- **rating**: Review rating.
- **label**: 'CG' (computer-generated/fake) or 'OR' (organic/real).
- **text_**: Review text.

### Data Preprocessing Steps

- Removing `_5` from the `category` column.
- Text cleaning (emoji conversion, contraction handling, URL removal, special character removal, lowercasing, punctuation removal).
- Tokenization and Lemmatization.

## Code Structure

The main notebook is `Model_Training.ipynb`, which includes:

- **Data Loading and Preprocessing**: Loading the dataset, cleaning text, and encoding labels.
- **Feature Engineering**: Using TF-IDF and Word2Vec for text vectorization, and POS tagging for additional features.
- **Model Training**: Building and training various machine learning models (Logistic Regression, SVM, Random Forest, MLP).
- **Pipeline Creation**: Creating pipelines for each model to streamline the preprocessing and training steps.
- **Model Evaluation**: Evaluating the models using confusion matrices and classification reports.

## Pipelines

The project implements several pipelines, including:

- **TF-IDF Pipeline**:
  - `TextPreprocessor`: Cleans and preprocesses text.
  - `TfidfVectorizer`: Converts text to TF-IDF vectors.
  - `OneHotEncoder`: Encodes categorical features (`category`).
  - `StandardScaler`: Scales numerical features (`rating`).
  - Classifiers: `LogisticRegression`, `SVC`, `RandomForestClassifier`, `MLPClassifier`.

- **Word2Vec Pipeline**:
  - `TextPreprocessor`: Cleans and preprocesses text.
  - `Word2VecTransformer`: Trains and transforms text using Word2Vec.
  - `OneHotEncoder`: Encodes categorical features (`category`).
  - `StandardScaler`: Scales numerical features (`rating`).
  - `MLPClassifier`: Multi-Layer Perceptron Classifier.

- **Spelling Correction and Word2Vec Pipeline**:
  - `SpellingCorrectionTransformer`: Corrects spelling in the text.
  - `Word2VecTransformer`: Trains and transforms text using Word2Vec.
  - `OneHotEncoder`: Encodes categorical features (`category`).
  - `StandardScaler`: Scales numerical features (`rating`).
  - `MLPClassifier`: Multi-Layer Perceptron Classifier.

- **POS Tagging and Word2Vec Pipeline**:
  - `TextPreprocessor`: Cleans and preprocesses text.
  - `Word2VecTransformer`: Trains and transforms text using Word2Vec.
  - `PosTaggerTransformer`: Tags the POS in the text.
  - `PosEncoderTransformer`: Encodes the POS tags.
  - `OneHotEncoder`: Encodes categorical features (`category`).
  - `StandardScaler`: Scales numerical features (`rating`).
  - `MLPClassifier`: Multi-Layer Perceptron Classifier.

## Models

The following models are trained and evaluated:

- **Logistic Regression**
- **Support Vector Classifier (SVC)**
- **Random Forest Classifier**
- **Multi-Layer Perceptron (MLP) Classifier**

The best-performing model (MLP with Word2Vec) is saved as `best_mlp_word2vec_model.pkl`.


