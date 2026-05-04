
# Disease Prediction System

## Problem Statement
This project predicts diseases based on patient symptoms using Machine Learning and Streamlit.

## Dataset
Dataset used:
https://www.kaggle.com/datasets/algozee/healthcare-disease-prediction-dataset

## Features
- Data Cleaning
- Label Encoding
- SMOTE Balancing
- Random Forest Classifier
- Evaluation Metrics
- Confusion Matrix
- Streamlit GUI

## Model Used
RandomForestClassifier

## How to Run

### Install Requirements
pip install -r requirements.txt

### Train Model
python train.py

### Run App
streamlit run app.py

## Project Structure

project/
│
├── train.py
├── app.py
├── model.pkl
├── label_encoder.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
└── dataset.csv

## Results Summary
The model was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Stratified K-Fold Cross Validation
