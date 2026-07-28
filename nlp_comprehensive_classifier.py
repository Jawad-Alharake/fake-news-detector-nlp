import os
import pandas as pd
import numpy as np
import re
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score
)

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True) 

def load_and_explore_data():
    """Loads datasets and prints out basic Exploratory Data Analysis (EDA)."""
    print("--- 1. LOADING DATA ---")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, 'dataset', 'training_data.csv')
    test_path = os.path.join(script_dir, 'dataset', 'testing_data.csv')
    
    train_df = pd.read_csv(train_path, sep='\t', header=None, names=['label', 'headline'])
    test_df = pd.read_csv(test_path, sep='\t', header=None, names=['label', 'headline'])
    
    print(f"Original Training data shape: {train_df.shape}")
    train_df.drop_duplicates(inplace=True)
    print(f"Training data shape after dropping duplicates: {train_df.shape}")
    
    print(f"Class distribution:\n{train_df['label'].value_counts(normalize=True) * 100}")
    
    train_df['length'] = train_df['headline'].apply(lambda x: len(str(x)))
    real_len = train_df[train_df['label'] == 1]['length'].mean()
    fake_len = train_df[train_df['label'] == 0]['length'].mean()
    print(f"\nAverage Real News length: {real_len:.1f} characters")
    print(f"Average Fake News length: {fake_len:.1f} characters\n")
    
    return train_df, test_df

def get_wordnet_pos(word):
    """Maps standard NLTK POS tags to WordNet POS tags for better lemmatization."""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_text(text):
    """
    Cleans text aggressively using Regex, removes stopwords, 
    and applies POS-tagged lemmatization.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # 1. Convert to lowercase
    text = str(text).lower()
    # 2. Replace tabs and breaklines with space
    text = re.sub(r"[\t\n]", " ", text)
    # 3. Remove punctuation and non-alphanumeric characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    # 4. Remove numbers
    text = re.sub(r'\d+', ' ', text)
    # 5. Remove standalone single characters (e.g. 's' after removing apostrophe)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'^[a-zA-Z]\s+', ' ', text)
    # 6. Substitute multiple spaces with a single space and strip edges
    text = re.sub(r'\s+', ' ', text).strip()
    # 7. Tokenize, remove stopwords, and Lemmatize
    words = text.split()
    
    cleaned_words = [
        lemmatizer.lemmatize(w, get_wordnet_pos(w)) 
        for w in words if w not in stop_words
    ]
    
    return " ".join(cleaned_words)

def benchmark_models(X_train, y_train, X_val, y_val):
    """Tests multiple models to find the best baseline."""
    print("--- 2. BENCHMARKING MULTIPLE MODELS ---")
    
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,3), max_df=0.75, min_df=5)
    X_train_vec = tfidf.fit_transform(X_train)
    X_val_vec = tfidf.transform(X_val)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Naive Bayes": MultinomialNB(),
        "Linear SVC": LinearSVC(dual=False),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best_model_name = ""
    best_acc = 0
    
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_val_vec)
        acc = accuracy_score(y_val, preds)
        print(f"{name} Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            
    print(f"\nBest Baseline Model: {best_model_name} with {best_acc:.4f} accuracy.\n")
    return best_model_name

def tune_best_model(X_train, y_train):
    """Builds a pipeline for the best model and tunes hyperparameters."""
    print("--- 3. HYPERPARAMETER TUNING ---")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,3), max_df=0.75, min_df=5)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    param_grid = {
        'tfidf__max_df': [0.75, 1.0],
        'tfidf__ngram_range': [(1, 2), (1, 3)],
        'clf__C': [0.1, 1, 10]
    }
    
    print("Running Grid Search (this may take a moment)...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}\n")
    
    return grid_search.best_estimator_

def evaluate_and_visualize(model, X_val, y_val):
    """Calculates deep metrics and plots a confusion matrix."""
    print("--- 4. DETAILED MODEL EVALUATION ---")
    
    y_pred = model.predict(X_val)
    
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\nClassification Report:\n")
    print(classification_report(y_val, y_pred, target_names=['Fake (0)', 'Real (1)']))
    
    cm = confusion_matrix(y_val, y_pred)
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake (0)', 'Real (1)'], 
                yticklabels=['Fake (0)', 'Real (1)'])
    plt.title('Fake News Detection - Confusion Matrix', fontsize=14)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save the plot to the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(script_dir, 'confusion_matrix.png')
    try:
        plt.savefig(plot_path)
        print(f"Confusion matrix visualization saved to:\n{plot_path}\n")
    except PermissionError:
        print("[WARNING] Could not save confusion matrix image. Close existing file and try again.")
    
    plt.show()

def main():
    # 1. Load Data
    train_df, test_df = load_and_explore_data()
    
    # 2. Deep Preprocessing
    print("Preprocessing text data (Lemmatization & Stopword removal)...")
    train_df['cleaned_headline'] = train_df['headline'].apply(preprocess_text)
    test_df['cleaned_headline'] = test_df['headline'].apply(preprocess_text)
    
    # 3. Split for Benchmarking and Evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        train_df['cleaned_headline'], 
        train_df['label'], 
        test_size=0.2, 
        random_state=42,
        stratify=train_df['label'] # Ensures equal fake/real ratio in splits
    )
    
    # 4. Benchmark Different Algorithms
    benchmark_models(X_train, y_train, X_val, y_val)
    
    # 5. Tune the Final Model on Training Split
    best_model = tune_best_model(X_train, y_train)
    
    # 6. Evaluate and Visualize on Validation Split
    evaluate_and_visualize(best_model, X_val, y_val)
    
    # 7. Retrain on 100% of available data for maximum accuracy before final prediction
    print("--- 5. FINAL PREDICTIONS ---")
    print("Retraining best model on 100% of the training data...")
    best_model.fit(train_df['cleaned_headline'], train_df['label'])
    
    # 8. Predict and Export
    predictions = best_model.predict(test_df['cleaned_headline'])
    test_df['label'] = predictions
    test_df = test_df.drop(columns=['cleaned_headline'])
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, 'dataset', 'predicted_testing_data.csv')
    
    test_df.to_csv(output_filename, sep='\t', header=False, index=False)
    print(f"Predictions mapped and successfully saved to:\n{output_filename}")

if __name__ == "__main__":
    main()