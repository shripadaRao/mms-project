# train a xgboost classifer for the processed vggish data

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

import tensorflow as tf
import vggish_input
import vggish_slim
import tensorflow_hub as hub



""" 2 models were trained, latter one is hyper parameter optimized"""

""" 
        Best parameters: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}
"""

# # Load the positive and negative features
# positive_features = np.load('positive_features.npy')
# negative_features = np.load('negative_features.npy')

# # Create the labels for positive and negative samples
# positive_labels = np.ones(len(positive_features))
# negative_labels = np.zeros(len(negative_features))

# # Concatenate the features and labels
# features = np.concatenate((positive_features, negative_features), axis=0)
# labels = np.concatenate((positive_labels, negative_labels), axis=0)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# # Create the XGBoost classifier
# classifier = xgb.XGBClassifier()

# # Train the classifier
# classifier.fit(X_train, y_train)

# # Evaluate the classifier
# accuracy = classifier.score(X_test, y_test)
# print('Test accuracy:', accuracy)

# # Save the model
# classifier.save_model('audio_classification_model.xgb')



def load_features(file_path):
    return np.load(file_path)

def create_labels(positive_features, negative_features):
    positive_labels = np.ones(len(positive_features))
    negative_labels = np.zeros(len(negative_features))
    return positive_labels, negative_labels

def concatenate_features_labels(positive_features, negative_features, positive_labels, negative_labels):
    features = np.concatenate((positive_features, negative_features), axis=0)
    labels = np.concatenate((positive_labels, negative_labels), axis=0)
    return features, labels

def split_data(features, labels, test_size=0.2, random_state=42):
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)

def train_classifier(X_train, y_train):
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)
    return classifier

def evaluate_classifier(classifier, X_test, y_test):
    accuracy = classifier.score(X_test, y_test)
    print('Test accuracy:', accuracy)

def save_model(classifier, file_path):
    classifier.save_model(file_path)

def perform_grid_search(classifier, param_grid, X_train, y_train):
    grid_search = GridSearchCV(classifier, param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    print('Best parameters:', grid_search.best_params_)
    return grid_search.best_estimator_

def main():
    # Load the positive and negative features
    positive_features = load_features('positive_features.npy')
    negative_features = load_features('negative_features.npy')

    # Limit the negative features to match the number of positive features
    negative_features = negative_features[:len(positive_features)]

    # Create the labels for positive and negative samples
    positive_labels, negative_labels = create_labels(positive_features, negative_features)

    # Concatenate the features and labels
    features, labels = concatenate_features_labels(positive_features, negative_features, positive_labels, negative_labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(features, labels)

    # Define the parameter grid for grid search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.1, 0.01, 0.001]
    }

    # Create the XGBoost classifier
    classifier = XGBClassifier()

    # Perform grid search
    best_classifier = perform_grid_search(classifier, param_grid, X_train, y_train)

    # Evaluate the best model
    evaluate_classifier(best_classifier, X_test, y_test)

    # Save the best model
    save_model(best_classifier, 'audio_classification_model-v2.xgb')



