# Fake Review Detection Project

This project aims to detect fake reviews using natural language processing (NLP) techniques and machine learning algorithms.

## Project Structure

- **preprocessing_code.ipynb**: Jupyter notebook containing the preprocessing steps for the dataset.
- **fakeReviewData.csv**: Dataset containing reviews for analysis.
- **README.md**: Project documentation.

## Preprocessing Steps

1. **Import Libraries**: Import necessary libraries such as pandas, nltk, and sklearn.
2. **Load Dataset**: Load the dataset from the CSV file.
3. **Text Cleaning**: Remove digits, punctuation, and extra spaces from the text.
4. **Remove Stopwords**: Remove common stopwords from the text.
5. **Lemmatization**: Lemmatize words to their base form.
6. **Vectorization**: Convert text data into numerical data using TF-IDF vectorization.

## How to Run

1. Clone the repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Open `preprocessing_code.ipynb` and run the cells to preprocess the data.
4. Use the preprocessed data for training machine learning models to detect fake reviews.

## Requirements

- Python 3.11.2
- pandas
- nltk
- scikit-learn

## Acknowledgements

This project uses the NLTK library for natural language processing and scikit-learn for machine learning.
