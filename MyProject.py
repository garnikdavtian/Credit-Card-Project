# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load the data
data_path = "C:/Users/User/OneDrive/Desktop/cvs/creditcard.csv"
df = pd.read_csv(data_path)

# Explore the data
print(df.shape)  # Print the shape of the dataset
print(type(df))  # Check the type of data
print(df.head())  # Display the first few rows of the dataset
print(df.columns)  # Display the columns in the dataset

# Check for null values
print(df.isnull().sum())  # Ensure there are no null values in the data

# Visualize potential noise in the data
sns.boxplot(x=df['V1'])  # Boxplot for feature 'V1' to identify outliers
plt.show()

# Prepare the data for modeling
X = df.drop(['Class'], axis=1)  # Features
y = df['Class']  # Target variable
print(y.value_counts())  # Print the class distribution

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Calculate scale_pos_weight for XGBoost
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])  # Adjust for class imbalance

# Initialize and train the XGBoost model
model_x = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)
xgb_cv_scores = cross_val_score(model_x, X, y, cv=5)  # Cross-validation scores
model_x.fit(X_train, y_train)  # Fit the model

# Make predictions on the test set
y_pred = model_x.predict(X_test)

# Evaluate the XGBoost model
print("Cross-Validation Scores for XGBoost:", xgb_cv_scores)
print("Mean CV Score:", xgb_cv_scores.mean())
print(f"Confusion Matrix with XGBoost:\n{confusion_matrix(y_test, y_pred)}")  # Confusion matrix
print(f"Classification Report with XGBoost:\n{classification_report(y_test, y_pred)}")  # Classification report

# Visualize the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix for XGBoost')
plt.show()

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(rf_model, X, y, cv=5)  # Cross-validation
rf_model.fit(X_train, y_train)  # Fit the model
rf_y_pred = rf_model.predict(X_test)  # Make predictions

# Evaluate the Random Forest model
print("Confusion Matrix for Random Forest:")
print(confusion_matrix(y_test, rf_y_pred))  
print("\nClassification Report for Random Forest:")
print(classification_report(y_test, rf_y_pred))
print("Cross-Validation Scores for Random Forest:", cv_scores)
print("Mean CV Score:", cv_scores.mean())
print("Random Forest Accuracy:", accuracy_score(y_test, rf_y_pred))

# Handling class imbalance with SMOTE
smote = SMOTE(random_state=42)  # Initialize SMOTE
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)  # Resample the training data
print(f"Original dataset shape: {Counter(y_train)}")
print(f"Resampled dataset shape: {Counter(y_resampled)}")

# Fit the Random Forest model with resampled data
rf_model_smote = RandomForestClassifier(random_state=42)
rf_model_smote.fit(X_resampled, y_resampled)
rf_y_pred_smote = rf_model_smote.predict(X_test)

# Evaluate the Random Forest model after SMOTE
print("Classification Report for Random Forest with SMOTE:")
print(classification_report(y_test, rf_y_pred_smote)) 

# Visualize the confusion matrix after SMOTE
rf_conf_matrix_smote = confusion_matrix(y_test, rf_y_pred_smote)
plt.figure(figsize=(10, 7))
sns.heatmap(rf_conf_matrix_smote, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix for Random Forest with SMOTE')
plt.show()

# Hyperparameter optimization with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model_smote, param_grid=param_grid, cv=3, scoring='recall', n_jobs=-1, verbose=2)
grid_search.fit(X_resampled, y_resampled)

# Output the best parameters and score
print("Best Parameters:", grid_search.best_params_)  # Print best parameters
print("Best Recall Score:", grid_search.best_score_)  # Best Recall Score

# Train the model with the best parameters
best_rf_model = grid_search.best_estimator_
best_rf_model.fit(X_resampled, y_resampled)
rf_y_pred_optimized = best_rf_model.predict(X_test)

# Evaluate the optimized Random Forest model
print("Classification Report for Optimized Random Forest:")
print(classification_report(y_test, rf_y_pred_optimized))

# ROC-AUC analysis
rf_y_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]  # Get predicted probabilities for the positive class
roc_auc = roc_auc_score(y_test, rf_y_pred_proba)  # Calculate the ROC-AUC score
print(f"ROC-AUC Score: {roc_auc}")

# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, rf_y_pred_proba)

# Plot the ROC curve
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='blue', label=f'Optimized Random Forest (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Dashed diagonal
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
