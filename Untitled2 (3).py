#!/usr/bin/env python
# coding: utf-8

# In[97]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix


# In[96]:


# Read the data set from a CSV file with Latin-1 encoding to handle special characters
data = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only the necessary columns and rename them for better understanding
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Print summary statistics of the dataset to understand its structure and contents
print("Summary Statistics of the Data Set")
print("="*40)
print(data.describe())
print("\n")

# Separate the dataset into features (the actual messages) and the target variable (labels indicating spam or not)
x = data['message']
y = data['label']

# Convert the text data into numerical data using TF-IDF vectorization
# TF-IDF (Term Frequency-Inverse Document Frequency) helps to transform text into meaningful numerical representations
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(x)

# Split the data into training and testing sets
# 33% of the data will be used for testing, and 67% for training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train a Naive Bayes classifier on the training data ( based on Bayes' Theorem)
clf_nb = MultinomialNB()
clf_nb.fit(X_train, y_train)

# Evaluate the performance of the Naive Bayes classifier on the test data
# Print classification report and confusion matrix to understand the model's accuracy and errors
y_pred_nb = clf_nb.predict(X_test)
print("Naive Bayes Classifier Report")
print("="*40)
print(classification_report(y_test, y_pred_nb))
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred_nb))
print("\n")

# Train a Random Forest classifier on the training data
clf_rf = RandomForestClassifier(random_state=42)
clf_rf.fit(X_train, y_train)

# Evaluate the performance of the Random Forest classifier on the test data
# Print classification report and confusion matrix to understand the model's accuracy and errors
y_pred_rf = clf_rf.predict(X_test)
print("Random Forest Classifier Report")
print("="*40)
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred_rf))
print("\n")

# Train a Decision Tree classifier on the training data
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt.fit(X_train, y_train)

# Evaluate the performance of the Decision Tree classifier on the test data
# Print classification report and confusion matrix to understand the model's accuracy and errors
y_pred_dt = clf_dt.predict(X_test)
print("Decision Tree Classifier Report")
print("="*40)
print(classification_report(y_test, y_pred_dt))
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred_dt))
print("\n")

# Allow the user to input a sample message for prediction
sample = input('Enter a message: ')
input_data = vectorizer.transform([sample])

# Print predictions for the sample message using all three classifiers
print("Sample Prediction - Ham (Not Spam) or Spam")
print("="*40)
print("Naive Bayes Prediction: ", clf_nb.predict(input_data)[0])
print("Random Forest Prediction: ", clf_rf.predict(input_data)[0])
print("Decision Tree Prediction: ", clf_dt.predict(input_data)[0])


# In[ ]:




