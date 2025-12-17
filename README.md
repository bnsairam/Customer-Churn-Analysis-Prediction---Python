# 📊 Customer Churn Analysis & Prediction

A machine learning project to predict customer churn using the Telco Customer Churn dataset. This project demonstrates end-to-end data analysis, preprocessing, model training, and evaluation techniques.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)

## 🎯 Project Overview

Customer churn occurs when customers stop using a company's service, leading to revenue loss. This project analyzes the Telco Customer Churn dataset to:
- Identify patterns in customer behavior
- Predict which customers are likely to churn
- Provide actionable insights for customer retention

**Model Accuracy: 78%**

## 📁 Dataset

The dataset contains customer information including:
- **Demographics**: Gender, Partner status, Dependents
- **Services**: Phone Service, Internet Service, Streaming services
- **Account Info**: Tenure, Contract type, Payment method
- **Charges**: Monthly charges, Total charges
- **Target**: Churn status (Yes/No)

**Dataset Source**: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## 🚀 Features

- **Data Preprocessing**: Handles missing values and encodes categorical variables
- **Exploratory Data Analysis**: Visualizes churn distribution and patterns
- **Feature Engineering**: Standardizes features for optimal model performance
- **Machine Learning**: Random Forest Classifier for prediction
- **Model Evaluation**: Comprehensive metrics including accuracy, confusion matrix, and classification report
- **Feature Importance**: Identifies key factors contributing to churn

## 📋 Requirements

```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
seaborn>=0.11.0
matplotlib>=3.4.0
jupyter>=1.0.0
joblib>=1.0.0
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/bnsairam/customer-churn-analysis.git
cd customer-churn-analysis
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the `data/` directory

## 💻 Usage

Run the Jupyter notebook or Python script:

```bash
# Using Jupyter Notebook
jupyter notebook notebooks/customer_churn_analysis.ipynb

# Or run the Python script
python churn_prediction.py
```

## 📊 Project Structure

```
customer-churn-analysis/
│
├── data/
│   ├── .gitkeep
│   └── telco_customer_churn.csv
│
├── notebooks/
│   └── customer_churn_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   └── visualization.py
│
├── models/
│   ├── .gitkeep
│   └── churn_model.pkl
│
├── results/
│   ├── .gitkeep
│   ├── confusion_matrix.png
│   ├── churn_distribution.png
│   └── feature_importance.png
│
├── churn_prediction.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## 🔍 Methodology

### 1. Data Preprocessing
- Handle missing values in `TotalCharges` column using median imputation
- Encode categorical variables using Label Encoding
- Split data into training (80%) and testing (20%) sets with stratification
- Apply feature scaling using StandardScaler for normalization

### 2. Model Training
- **Algorithm**: Random Forest Classifier
- Ensemble learning approach combining multiple decision trees
- Hyperparameters: 100 estimators, random_state=42
- Trained on preprocessed and scaled features

### 3. Evaluation Metrics
- **Accuracy Score**: 78%
- **Confusion Matrix** for visual performance analysis
- **Classification Report** with precision, recall, and F1-score
- **Feature Importance** analysis to identify key churn predictors

## 📈 Results

### Performance Metrics
- **Accuracy**: 78%
- **Precision (Churn)**: Measures how many predicted churners actually churned
- **Recall (Churn)**: Measures how many actual churners were identified

### Confusion Matrix Analysis
- **True Negatives**: 924 (correctly predicted non-churners)
- **True Positives**: 181 (correctly predicted churners)
- **False Positives**: 117 (non-churners predicted as churners)
- **False Negatives**: 187 (churners predicted as non-churners)

### Key Insights
- The model shows good performance in identifying non-churners
- Higher false negative rate suggests need for model tuning to catch more churners
- Class imbalance may be affecting model performance
- Feature importance analysis reveals critical factors in customer churn

## 🎨 Visualizations

The project generates several visualizations:
- **Churn Distribution**: Bar plot showing the balance of churned vs non-churned customers
- **Confusion Matrix**: Heatmap showing model prediction performance
- **Feature Importance**: Bar chart highlighting the most influential features

## 🔮 Future Improvements

- [ ] Implement hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- [ ] Try alternative algorithms (XGBoost, LightGBM, Neural Networks)
- [ ] Handle class imbalance with SMOTE or class weights
- [ ] Perform extensive feature engineering and selection
- [ ] Create interactive dashboard using Streamlit or Plotly Dash
- [ ] Deploy model as REST API using Flask or FastAPI
- [ ] Add cross-validation for more robust evaluation
- [ ] Implement model interpretability with SHAP values

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Sai Ram BN**

- 🌐 Portfolio: [bnsairam.vercel.app](https://bnsairam.vercel.app/)
- 💼 LinkedIn: [linkedin.com/in/sairambn](https://www.linkedin.com/in/sairambn/)
- 🐙 GitHub: [@bnsairam](https://github.com/bnsairam)
- 📧 Email: bnsairam14@gmail.com

## 🙏 Acknowledgments

- Dataset provided by [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Inspired by real-world business problems in customer retention
- Built with scikit-learn and the Python data science ecosystem
- Special thanks to the open-source community

## 📚 Related Projects

Check out my other machine learning projects:
- [GitHub Profile](https://github.com/bnsairam)
- [Portfolio Website](https://bnsairam.vercel.app/)

---

⭐ If you found this project helpful, please consider giving it a star!

**Keywords**: Machine Learning, Customer Churn, Predictive Analytics, Random Forest, Python, Data Science, scikit-learn, Classification
