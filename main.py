# Jeremy Pretty
# Natural Language Processing with Diaster Tweets Kaggle
# Jan 28, 2024
"""
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Preprocessing function for text data
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Load the datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Apply preprocessing to the text columns
train_data['text_clean'] = train_data['text'].apply(preprocess_text)
test_data['text_clean'] = test_data['text'].apply(preprocess_text)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['text_clean'])
X_test_tfidf = tfidf_vectorizer.transform(test_data['text_clean'])

# Target variable
y_train = train_data['target']

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(X_train_tfidf, y_train, test_size=0.2, random_state=42)

# Initialize logistic regression model
log_reg_model = LogisticRegression(random_state=42)

# Train the model
log_reg_model.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = log_reg_model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_val_pred)
report = classification_report(y_val, y_val_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Predict on the test set
test_predictions = log_reg_model.predict(X_test_tfidf)

# Prepare the submission file
submission = pd.DataFrame({'id': test_data['id'], 'target': test_predictions})
submission.to_csv('disaster_tweets_predictions.csv', index=False)
"""

import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

nltk.download('wordnet')
nltk.download('stopwords')

# Preprocessing function for text data
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs, special characters, and numbers
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text) # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

# Enhanced preprocessing function with lemmatization and custom stopwords handling
def enhanced_preprocess_text(text):
    text = preprocess_text(text)  # Apply basic preprocessing

    # Lemmatization and removing stopwords (except 'not' and 'no')
    lemmatizer = WordNetLemmatizer()
    stopwords_set = set(stopwords.words('english')) - {'not', 'no'}
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stopwords_set])
    return text

# Load the datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Apply the enhanced preprocessing to the text columns
train_data['text_clean'] = train_data['text'].apply(enhanced_preprocess_text)
test_data['text_clean'] = test_data['text'].apply(enhanced_preprocess_text)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['text_clean'])
X_test_tfidf = tfidf_vectorizer.transform(test_data['text_clean'])

# Target variable
y_train = train_data['target']

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(X_train_tfidf, y_train, test_size=0.2, random_state=42)

# Initialize logistic regression model
log_reg_model = LogisticRegression(random_state=42)

# Train the model
log_reg_model.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = log_reg_model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_val_pred)
report = classification_report(y_val, y_val_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Predict on the test set
test_predictions = log_reg_model.predict(X_test_tfidf)

# Prepare the submission file
submission = pd.DataFrame({'id': test_data['id'], 'target': test_predictions})
submission.to_csv('disaster_tweets_predictions.csv', index=False)
