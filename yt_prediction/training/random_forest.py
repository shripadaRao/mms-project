import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load the positive and negative features
positive_features = np.load('/content/drive/MyDrive/positive_features.npy')
negative_features = np.load('/content/drive/MyDrive/negative_features.npy')

negative_features = negative_features[:len(positive_features)]

# Create the labels for positive and negative samples
positive_labels = np.ones(len(positive_features))
negative_labels = np.zeros(len(negative_features))

# Concatenate the features and labels
features = np.concatenate((positive_features, negative_features), axis=0)
labels = np.concatenate((positive_labels, negative_labels), axis=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Create the Random Forest classifier
classifier = RandomForestClassifier()

# Perform grid search
grid_search = GridSearchCV(classifier, param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Print the best parameters found
print('Best parameters:', grid_search.best_params_)

# Get the best model
best_classifier = grid_search.best_estimator_

# Evaluate the best model
accuracy = best_classifier.score(X_test, y_test)
print('Test accuracy:', accuracy)
