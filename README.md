# Vertex AI Classification & Deployment Project

## Overview
This project demonstrates an end-to-end machine learning pipeline on **Google Cloud Platform (GCP)** using:
- **Google Cloud Storage (GCS)** for data storage
- **BigQuery** for data preprocessing and feature engineering
- **Vertex AI** for training, evaluation, and deployment of a classification model as a **REST API endpoint**

The primary goal is to build and deploy a supervised classification model that predicts fraudulent credit card transactions.

---

## Source Data
**Dataset:**  
The dataset is sourced from **BigQuery public datasets**:  
`bigquery-public-data.ml_datasets.ulb_fraud_detection`.

**Description:**  
- **Rows:** 284,807 credit card transactions  
- **Target:** `Class`  
  - `1` → Fraudulent transaction  
  - `0` → Normal transaction  
- **Features:**  
  - `V1 ... V28` → PCA-transformed components  
  - `Time` (int) → Seconds since first transaction  
  - `Amount` (float) → Transaction amount  

**Data Reference:**  
- [Kaggle Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- [BigQuery Public Datasets](https://cloud.google.com/bigquery/public-data)

---

## Project Workflow
1. **Data Preparation**  
   - Export BigQuery data to GCS (CSV format).  
   - Add:
     - `transaction_id` (unique string ID)
     - `splits` (TRAIN: 80%, VALIDATE: 10%, TEST: 10%)

2. **Modeling**  
   - Load prepared data from BigQuery into Vertex AI.  
   - Train a **classification model** (using AutoML).  
   - Evaluate using metrics like ROC-AUC, Precision, and Recall.

3. **Deployment**  
   - Deploy the trained model as a **Vertex AI Endpoint**.  
   - Expose it via a **REST API** for real-time predictions.

---

## Tech Stack
- **Google Cloud Storage (GCS)** – raw data and model artifacts
- **BigQuery** – feature engineering and preprocessing
- **Vertex AI** – model training, evaluation, and deployment
- **Python & Vertex AI KFP** – pipeline automation
- **REST API** – model serving

---

## Getting Started
### Prerequisites
- A GCP project with **Vertex AI, BigQuery, and GCS enabled**
- Python 3.10+ environment with:
  ```bash
  pip install google-cloud-bigquery google-cloud-storage google-cloud-aiplatform pandas scikit-learn
