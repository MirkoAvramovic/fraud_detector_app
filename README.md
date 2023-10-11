# Fraud Detector

This repository contains the code and data for a fraud detection project implemented in Python. The project was part of the final assignment for the "Fraud Detection in Python" course on Udemy.

## Overview

The project's goal is to detect fraudulent transactions in a highly imbalanced dataset of credit card transactions. It leverages machine learning models and evaluation metrics to identify fraudulent activities while minimizing false positives.

### Dataset

The dataset used for this project is called `credit_card.csv`. It contains transactions made by credit cards in September 2013 by European cardholders. Key details of the dataset include:

- 284,807 transactions
- 492 frauds (0.172% of all transactions)
- Numerical input variables obtained through PCA transformation
- Non-transformed features: 'Time' (seconds elapsed) and 'Amount' (transaction amount)
- Response variable: 'Class' (1 for fraud, 0 for non-fraud)

Given the class imbalance ratio, the project uses the Area Under the Precision-Recall Curve (AUPRC) as the primary evaluation metric, as recommended.

### Code Files

- `app.py`: A Streamlit web app for interacting with the fraud detection model and exploring evaluation metrics.
- `eval.py`: A Python script containing functions to calculate evaluation metrics based on specified thresholds.
- `error_df.csv`: Extracted data file containing 'Target variable' (actual labels) and 'Score' (model predictions).
- Other dependencies and Python libraries.

## Usage

1. Clone this repository to your local machine
2. Install the required Python dependencies
3. Run the Streamlit app to interact with the model
4. Adjust the threshold slider and input costs to explore evaluation metrics based on your preferences.

## Evaluation Metrics
The evaluation metrics include:
- AUPRC (Area Under the Precision-Recall Curve) - Primary metric
- Total cost of fraud detection (with adjustable cost parameters)
- Confusion matrix statistics (True Positives, False Positives, True Negatives, False Negatives)
- Number of fraudulent transactions detected and not detected
- Number of good transactions classified as fraudulent and good

## Project Structure
The project structure includes:
- Data preprocessing and feature selection
- Model training and prediction
- Evaluation metrics calculation
- Interactive web app for exploring evaluation metrics

## Acknowledgments
- [Udemy Course: Fraud Detection using Python](https://www.udemy.com/course/fraud-detection-using-python/)
- [Dataset source: Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
