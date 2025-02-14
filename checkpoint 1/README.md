# EDA and Preprocessing Code

This repository contains two Jupyter notebooks for the **Fake Review Detection Project**: `EDA.ipynb` and `Preprocessing code.ipynb`. These notebooks are part of the **Winter of Code (DA-IICT)** initiative and aim to detect fake reviews using natural language processing (NLP) techniques and machine learning algorithms.

---

## 📊 **EDA.ipynb**

### **Exploratory Data Analysis (EDA)**

The `EDA.ipynb` notebook focuses on understanding the dataset through various data visualization and analysis techniques.

### **Key Sections:**

1. **Preprocessing Data**
   - Import necessary libraries.
   - Load the dataset.
   - Clean and preprocess the text data.

2. **Checking Data Quality**
   - Verify if all words are in English.
   - Check for the presence of Unicode characters.

3. **Visualizing Data**
   - Generate pie charts, bar charts, and histograms to understand the distribution of categories, labels, and ratings.
   - Create word clouds to visualize the most frequent words in original and fake comments.

---

## 🏗️ **Preprocessing code.ipynb**

### **Data Preprocessing**

The `Preprocessing code.ipynb` notebook is dedicated to preparing the dataset for machine learning models by performing various text preprocessing steps.

### **Key Sections:**

1. **Import Libraries**
   - Import essential libraries such as `pandas`, `nltk`, and `scikit-learn`.

2. **Load Dataset**
   - Load the dataset from the `fakeReviewData.csv` file.

3. **Text Cleaning**
   - Remove digits, punctuation, and extra spaces.
   - Standardize text formatting.

4. **Remove Stopwords**
   - Eliminate common stopwords using the `nltk.corpus` module.

5. **Lemmatization**
   - Convert words to their base forms using the `WordNetLemmatizer`.

6. **Vectorization**
   - Transform cleaned text into numerical features using **TF-IDF vectorization** to prepare data for machine learning models.

---

Both notebooks are essential for understanding and preparing the dataset for the Fake Review Detection Project. Follow the steps in each notebook to preprocess the data and perform exploratory data analysis.
