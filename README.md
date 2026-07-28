# Fake News Detector: An NLP Binary Classifier

This repository contains a complete Natural Language Processing (NLP) machine learning pipeline designed to classify news headlines as either Real (1) or Fake (0). 

It was developed to demonstrate the end-to-end process of text preprocessing, feature engineering, model benchmarking, and hyperparameter tuning.

## Project Objective

The rapid spread of misinformation requires automated, scalable solutions. The goal of this project was to ingest a dataset of ~34,000 labeled news headlines, build a highly accurate text classifier, and use it to predict and replace dummy labels in a separate, unseen testing dataset.

## Key Features & Pipeline Steps

1. **Exploratory Data Analysis (EDA):**
   - Analyzed headline character lengths to identify behavioral differences between deceptive and objective reporting (e.g., discovering Fake News headlines are generally 30% longer).

2. **Deep Text Preprocessing:**
   - Aggressive regex cleaning (removing punctuation, numbers, and tabs).
   - Stopword removal to eliminate structural linguistic noise.
   - **Advanced Lemmatization:** Integrated NLTK's `pos_tag` to dynamically map Part-of-Speech context (adjectives, verbs, adverbs) to ensure the `WordNetLemmatizer` accurately reduced words to their true roots.

3. **Feature Engineering (TF-IDF):**
   - Converted the cleaned text into a matrix of numerical weights using `TfidfVectorizer`.
   - Utilized an N-gram range of (1, 3) to capture critical contextual bigrams and trigrams (e.g., "not true", "white house").

4. **Model Benchmarking & Tuning:**
   - Tested 4 baseline algorithms (Logistic Regression, Naive Bayes, Linear SVC, and Random Forest).
   - Selected Linear SVC as the primary classifier and optimized it using `GridSearchCV` across 12 unique parameter combinations (3-fold cross-validation).

## Final Model Performance

The tuned Linear SVC model achieved outstanding, balanced results on the validation set, demonstrating its ability to catch fake news without falsely penalizing objective reporting:

- **Overall Accuracy:** ~92.5%
- **Precision:** ~91.7%
- **Recall:** ~93.7%
- **F1-Score:** ~92.7%

## Repository Structure

```text
├── dataset/
│   ├── training_data.csv           # The labeled training data (not included in repo if large)
│   ├── testing_data.csv            # The unlabeled testing data
│   └── predicted_testing_data.csv  # Output: The final predictions
├── nlp_comprehensive_classifier.py # The main Python pipeline script
├── confusion_matrix.png            # Visual output of model evaluation
├── presentation_nlp.pptx           # Powerpoint presentation of the project
└── README.md
```

 ## How to Run

1. Clone the repository.
2. Ensure you have the required libraries installed: `pip install pandas numpy scikit-learn nltk matplotlib seaborn`.
3. Ensure the `dataset` folder is in the same directory as the script and contains the necessary `.csv` files.
4. Run the script:

```bash
   python nlp_comprehensive_classifier.py
```
