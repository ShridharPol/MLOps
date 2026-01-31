# FastAPI Lab 1 — Iris Classification API (with Probabilities & Model Info)

## Overview

In this lab, we expose a **machine learning model as a REST API** using **FastAPI** and **Uvicorn**.

- The model is a **Decision Tree Classifier** trained on the **Iris dataset**
- The API supports:
  - Health checks
  - Predictions with **class probabilities**
  - Model metadata inspection (without affecting predictions)

### Technologies Used

- **FastAPI** – high-performance Python API framework
- **Uvicorn** – ASGI server for FastAPI
- **scikit-learn** – model training
- **joblib** – model serialization

---

## Workflow

1. Train a **Decision Tree Classifier** on the Iris dataset
2. Save the trained model to disk
3. Serve predictions via FastAPI
4. Expose model metadata via a read-only endpoint

---

## Project Structure

mlops_labs
└── fastapi_lab1
    ├── assets/
    ├── fastapi_lab1_env/
    ├── model/
    │   └── iris_model.pkl
    ├── src/
    │   ├── __init__.py
    │   ├── data.py
    │   ├── train.py
    │   ├── predict.py
    │   └── main.py
    ├── README.md
    └── requirements.txt

---

## Setup Instructions

### Create Virtual Environment

python -m venv fastapi_lab1_env

### Activate Environment (Windows)

fastapi_lab1_env\Scripts\Activate

### Install Dependencies

pip install -r requirements.txt

---

## Running the API

python -m uvicorn src.main:app --reload

Open browser at:
http://127.0.0.1:8000/docs

---

## API Endpoints

### GET /
Returns health status.

### POST /predict
Returns predicted class, class name, and probabilities.

### GET /model-info
Returns model metadata and class mapping.

---

## Notes

Decision Trees support predict_proba(), which produces hard probabilities.
The screenshot folder contains the snapshots of the implementation.
---

## Summary

This lab demonstrates exposing an ML model via FastAPI with clean structure, documentation, and probabilistic outputs.
