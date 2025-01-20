# 🕵️‍♀️ Fake Review Detection Project (DA-IICT Winter of Code)

This project aims to detect fake reviews using a combination of Natural Language Processing (NLP) techniques and machine learning algorithms. It's part of the **Winter of Code** initiative at DA-IICT.

## 🚀 Project Overview

Online reviews significantly influence consumer decisions. However, the presence of fake reviews undermines their credibility and can mislead potential buyers. This project tackles the challenge of identifying fake reviews by leveraging the power of machine learning. We preprocess review datasets, engineer relevant features, train various models, and rigorously evaluate their performance to build a robust fake review detection system.

## 🏗️ Project Status

The project has progressed through several key checkpoints:

### **Checkpoint 1: Data Preprocessing and Initial Exploration**

-   **Completed initial data preprocessing, including:**
    -   Handling missing values (if any).
    -   Tokenizing and cleaning text data (removing digits, punctuation, extra spaces).
    -   Removing stop words (common words like "the," "a," "is," etc.).
    -   Lemmatization (reducing words to their root form, e.g., "running" to "run").
    -   Loading and exploring the `fakeReviewData.csv` dataset.

### **Checkpoint 2: Model Training and Evaluation**

-   **Feature Engineering:**
    -   **TF-IDF Vectorization:** Representing text as numerical features based on word frequency and importance.
    -   **Word2Vec Embeddings:** Capturing semantic relationships between words using pre-trained or custom-trained Word2Vec models.
    -   **One-Hot Encoding:** Transforming categorical features (like product categories) into numerical representations.
    -   **StandardScaler:** Standardizing numerical features (like ratings) for better model performance.

-   **Model Training and Tuning:**
    -   Trained and evaluated the following models:
        -   Logistic Regression
        -   Support Vector Classifier (SVC)
        -   Random Forest
        -   Multi-layer Perceptron (MLP)
    -   Utilized `Pipeline` and `ColumnTransformer` for efficient data processing and model building.
    -   Employed `GridSearchCV` for hyperparameter tuning to optimize model performance.

-   **Evaluation:**
    -   Assessed model performance using:
        -   Confusion Matrices
        -   Classification Reports (precision, recall, F1-score, accuracy)
    -   Achieved the best performance with an **MLP model using Word2Vec embeddings**, reaching **90% accuracy** and an **F1-score of 0.90** on the test set.
    -   **Random Forest with TF-IDF Vectorization** also showed promising results with **82% accuracy** and an **F1-score of 0.82**.

-   **Model Saving:**
    -   Saved the best-performing models using `joblib` for later use.

- **Exploration of Advanced Techniques**
    -   Implemented Recurrent Neural Networks (RNNs) using Keras and TensorFlow for enhanced sequence modeling.
    -   Integrated the Keras model into the scikit-learn pipeline using `KerasClassifier`.
    -   Conducted hyperparameter tuning specifically for the RNN model to optimize its architecture and learning parameters.



## ✨ Key Findings

-   **Word embeddings (Word2Vec) significantly improve performance** by capturing semantic meaning, leading to better identification of fake reviews.
-   Traditional machine learning models like **Random Forest (with TF-IDF)** can also provide good results, offering a balance between performance and interpretability.
-   MLP neural network demonstrates strong potential for this task, achieving high accuracy.


## 💾 Saved Models

The best-performing model is saved as:

-   `best_mlp_word2vec_model.pkl` (MLP with Word2Vec Embeddings)


## 📚 References

-   [Scikit-learn](https://scikit-learn.org/)
-   [NLTK](https://www.nltk.org/)
-   [Gensim](https://radimrehurek.com/gensim/)
-   [Pandas](https://pandas.pydata.org/)
-   [Matplotlib](https://matplotlib.org/)
-   [TensorFlow](https://www.tensorflow.org/)
-   [Keras](https://keras.io/)

## 🧑‍💻 Author

Ujjwal Bhansali