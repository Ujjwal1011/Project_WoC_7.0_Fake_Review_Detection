# ✨ Fake Review Detection Project ✨

This project focuses on building robust machine learning models to detect fake reviews, a crucial task for maintaining the integrity and trustworthiness of online platforms. We explore various techniques, including traditional machine learning classifiers and advanced embedding methods like Word2Vec, to achieve the best possible performance in identifying those pesky fake reviews! 🕵️‍♀️

## Project Structure

The project is organized in a Jupyter Notebook (`Model Training.ipynb`) that details the entire workflow. Here's a sneak peek:

1. **Data Loading and Preprocessing:**
    *   Loads the dataset `fakeReviewData.csv` (Make sure you have this dataset in your working directory).
    *   Preprocesses text data using techniques like lowercasing, tokenization, stop word removal, and lemmatization. These steps help clean and normalize the text for better model understanding.
    *   Encodes the target variable 'label' (which probably indicates whether a review is fake or not) using LabelEncoder.
    *   Splits the data into training and testing sets to evaluate model performance.

2. **Feature Engineering:**
    *   **TF-IDF:** Employs TF-IDF vectorization to represent text data as numerical features. TF-IDF captures the importance of words within a document relative to the entire corpus.
    *   **Word2Vec:** Utilizes Word2Vec embeddings to capture semantic relationships between words. Words with similar meanings will have similar vector representations.
    *   **One-Hot Encoding:** Converts categorical features like 'category' into a numerical format that machine learning models can understand.
    *   **StandardScaler:** Standardizes numerical features like 'rating' to have zero mean and unit variance, which can improve model performance.

3. **Model Training and Evaluation:**
    *   Trains and evaluates the following classification models:
        *   **Logistic Regression:** A simple and interpretable linear model.
        *   **Support Vector Classifier (SVC):** A powerful model capable of finding complex decision boundaries.
        *   **Random Forest:** An ensemble method that combines multiple decision trees to improve accuracy and reduce overfitting.
        *   **Multi-layer Perceptron (MLP):** A neural network model that can learn complex patterns in the data.
    *   Uses `Pipeline` to streamline the data processing and modeling steps, making the code cleaner and more organized.
    *   Employs `ColumnTransformer` to apply different preprocessing steps to different columns in the dataset.
    *   Performs hyperparameter tuning with `GridSearchCV` to find the best model configurations for optimal performance.
    *   Evaluates model performance using:
        *   **Confusion Matrices:** To visualize the performance of the classification models.
        *   **Classification Reports:** Providing detailed metrics like precision, recall, F1-score, and accuracy.

4. **Model Saving:**
    *   Saves the best-performing models using `joblib` so you can load and reuse them later without retraining.

## Key Findings 🏆

-   The **MLP model with Word2Vec embeddings** achieved the highest accuracy of **90%** and an F1-score of **0.90** on the test set. This indicates that using word embeddings to capture semantic meaning significantly helps in distinguishing fake reviews.
-   The **Random Forest with TF-IDF Vectorization** also performed well, achieving an accuracy of **82%** and an F1-score of **0.82**.

-   The confusion matrices and classification reports provide detailed insights into model performance across different classes.

## Saved Models 💾

The best-performing models is saved as:


-   `best_mlp_word2vec_model.pkl` (MLP with Word2Vec Embeddings)


## Future Improvements 🚀

-   Explore different embedding models like GloVe and FastText to see if they further improve performance.
-   Experiment with more advanced deep learning architectures, such as Recurrent Neural Networks (RNNs) .
-   Incorporate additional features or data sources (e.g., user metadata, review metadata) to potentially enhance model accuracy.


## Author

Ujjwal Bhansali

## References 📚

*   [Scikit-learn](https://scikit-learn.org/)
*   [NLTK](https://www.nltk.org/)
*   [Gensim](https://radimrehurek.com/gensim/)
*   [Pandas](https://pandas.pydata.org/)
*   [Matplotlib](https://matplotlib.org/)