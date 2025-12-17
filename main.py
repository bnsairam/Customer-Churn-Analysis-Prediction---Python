# ============================================
# File: src/data_preprocessing.py
# ============================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(filepath):
    """Load the customer churn dataset"""
    dataset = pd.read_csv(filepath)
    print(f"Dataset loaded successfully with shape: {dataset.shape}")
    return dataset


def handle_missing_values(dataset):
    """Handle missing and incorrect values in the dataset"""
    # Convert TotalCharges to numeric and fill missing values
    dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')
    dataset['TotalCharges'].fillna(dataset['TotalCharges'].median(), inplace=True)
    
    print(f"Missing values after handling:\n{dataset.isnull().sum()}")
    return dataset


def encode_categorical_features(dataset):
    """Encode categorical variables to numerical values"""
    labelencoder = LabelEncoder()
    
    categorical_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
        'PaperlessBilling', 'PaymentMethod', 'Churn'
    ]
    
    for col in categorical_cols:
        if col in dataset.columns:
            dataset[col] = labelencoder.fit_transform(dataset[col])
    
    print("Categorical features encoded successfully")
    return dataset


def split_features_target(dataset):
    """Split dataset into features and target variable"""
    X = dataset.drop(['customerID', 'Churn'], axis=1)
    y = dataset['Churn']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    return X, y


def scale_features(X_train, X_test):
    """Apply feature scaling to training and test data"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Feature scaling completed")
    return X_train_scaled, X_test_scaled, scaler


def preprocess_data(filepath, test_size=0.2, random_state=0):
    """Complete preprocessing pipeline"""
    # Load data
    dataset = load_data(filepath)
    
    # Handle missing values
    dataset = handle_missing_values(dataset)
    
    # Encode categorical features
    dataset = encode_categorical_features(dataset)
    
    # Split features and target
    X, y = split_features_target(dataset)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    print("\nPreprocessing completed successfully!")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ============================================
# File: src/model_training.py
# ============================================

from sklearn.ensemble import RandomForestClassifier
import pickle


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """Train a Random Forest Classifier"""
    print("Training Random Forest Classifier...")
    
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    
    print("Model training completed!")
    return clf


def save_model(model, filepath='models/random_forest_model.pkl'):
    """Save the trained model to disk"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath='models/random_forest_model.pkl'):
    """Load a trained model from disk"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


def train_model(X_train, y_train, save_path=None):
    """Complete model training pipeline"""
    model = train_random_forest(X_train, y_train)
    
    if save_path:
        save_model(model, save_path)
    
    return model


# ============================================
# File: src/model_evaluation.py
# ============================================

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import numpy as np


def make_predictions(model, X_test):
    """Make predictions on test data"""
    y_pred = model.predict(X_test)
    return y_pred


def calculate_metrics(y_test, y_pred):
    """Calculate various performance metrics"""
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    return metrics


def get_confusion_matrix(y_test, y_pred):
    """Generate confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    return cm


def print_evaluation_report(metrics, cm):
    """Print detailed evaluation report"""
    print("\n" + "="*50)
    print("MODEL EVALUATION REPORT")
    print("="*50)
    
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"True Negatives:  {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives:  {cm[1][1]}")
    
    print("\nInterpretation:")
    print(f"- Correctly identified non-churners: {cm[0][0]}")
    print(f"- Correctly identified churners: {cm[1][1]}")
    print(f"- Missed churners (False Negatives): {cm[1][0]}")
    print(f"- False alarms (False Positives): {cm[0][1]}")
    print("="*50 + "\n")


def evaluate_model(model, X_test, y_test):
    """Complete model evaluation pipeline"""
    # Make predictions
    y_pred = make_predictions(model, X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # Get confusion matrix
    cm = get_confusion_matrix(y_test, y_pred)
    
    # Print report
    print_evaluation_report(metrics, cm)
    
    return y_pred, metrics, cm


# ============================================
# File: src/visualization.py
# ============================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


def plot_churn_distribution(dataset, save_path=None):
    """Plot the distribution of churned vs non-churned customers"""
    plt.figure(figsize=(8, 6))
    
    sns.countplot(x='Churn', data=dataset, palette='coolwarm')
    plt.title('Customer Churn Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Churn (0 = No, 1 = Yes)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for container in plt.gca().containers:
        plt.gca().bar_label(container)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Churn distribution plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(8, 6))
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=["No Churn", "Churn"]
    )
    disp.plot(cmap="coolwarm", values_format='d')
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.grid(False)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()


def plot_feature_importance(model, feature_names, top_n=10, save_path=None):
    """Plot feature importance from the trained model"""
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.title(f'Top {top_n} Feature Importances', fontsize=16, fontweight='bold')
    plt.bar(range(top_n), importances[indices], color='steelblue', alpha=0.8)
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()


# ============================================
# File: src/main.py
# ============================================

import os
import sys
from data_preprocessing import preprocess_data
from model_training import train_model, save_model
from model_evaluation import evaluate_model
from visualization import plot_confusion_matrix


def main():
    """Main execution pipeline"""
    print("\n" + "="*60)
    print("CUSTOMER CHURN ANALYSIS & PREDICTION")
    print("="*60 + "\n")
    
    # Configuration
    DATA_PATH = 'data/raw/telco_customer_churn.csv'
    MODEL_PATH = 'models/random_forest_model.pkl'
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please download the dataset and place it in the data/raw/ directory")
        return
    
    # Step 1: Data Preprocessing
    print("Step 1: Data Preprocessing")
    print("-" * 60)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(DATA_PATH)
    
    # Step 2: Model Training
    print("\nStep 2: Model Training")
    print("-" * 60)
    model = train_model(X_train, y_train, save_path=MODEL_PATH)
    
    # Step 3: Model Evaluation
    print("\nStep 3: Model Evaluation")
    print("-" * 60)
    y_pred, metrics, cm = evaluate_model(model, X_test, y_test)
    
    # Step 4: Visualization
    print("\nStep 4: Generating Visualizations")
    print("-" * 60)
    plot_confusion_matrix(cm, save_path='outputs/figures/confusion_matrix.png')
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()


# ============================================
# File: tests/test_preprocessing.py
# ============================================

import unittest
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from src.data_preprocessing import handle_missing_values, encode_categorical_features


class TestPreprocessing(unittest.TestCase):
    
    def setUp(self):
        """Create sample data for testing"""
        self.sample_data = pd.DataFrame({
            'customerID': ['1', '2', '3'],
            'gender': ['Male', 'Female', 'Male'],
            'TotalCharges': ['100.5', '200.3', ' '],
            'Churn': ['No', 'Yes', 'No']
        })
    
    def test_handle_missing_values(self):
        """Test missing value handling"""
        df = handle_missing_values(self.sample_data.copy())
        self.assertFalse(df['TotalCharges'].isnull().any())
    
    def test_encode_categorical_features(self):
        """Test categorical encoding"""
        df = self.sample_data.copy()
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = encode_categorical_features(df)
        self.assertTrue(df['gender'].dtype in [np.int64, np.int32])


if __name__ == '__main__':
    unittest.main()
