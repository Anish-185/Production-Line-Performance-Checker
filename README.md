# 🏭 Production Line Performance Checker

An end-to-end Machine Learning and MLOps project that predicts machine failures in a manufacturing production line using sensor data. The project includes model training, explainability, a REST API built with FastAPI, Docker containerization, and cloud deployment on Render.

---

## 🌐 Live Demo

### API URL
https://production-line-performance-checker.onrender.com

### Swagger Documentation
https://production-line-performance-checker.onrender.com/docs

### GitHub Repository
https://github.com/Anish-185/Production-Line-Performance-Checker

---

# 📌 Project Overview

Unexpected machine failures can cause significant downtime and financial losses in manufacturing environments.

This project uses machine sensor readings to predict whether a machine is likely to fail before failure occurs, enabling predictive maintenance and reducing operational costs.

The system:

- Collects machine operating parameters
- Predicts machine failure probability
- Provides explainable AI insights using SHAP
- Exposes predictions through a FastAPI REST API
- Runs locally with Docker
- Is deployed publicly on Render

---

# 🎯 Objectives

- Build a predictive maintenance model
- Identify the most important factors affecting machine failure
- Explain model decisions using SHAP
- Deploy the model as a production-ready API
- Containerize the application using Docker

---

# 📊 Dataset

### AI4I 2020 Predictive Maintenance Dataset

The dataset contains simulated manufacturing machine data including:

| Feature | Description |
|----------|------------|
| Type | Machine Type |
| Air Temperature | Ambient temperature |
| Process Temperature | Process operating temperature |
| Rotational Speed | Speed of machine rotation |
| Torque | Torque generated |
| Tool Wear | Tool usage duration |
| Machine Failure | Target Variable |

Dataset Size:

```text
10,000 rows
14 columns
```

Target:

```text
Machine Failure
0 = No Failure
1 = Failure
```

---

# 🧠 Machine Learning Pipeline

## Data Preprocessing

- Removed unnecessary columns
- Selected predictive features
- Feature scaling using StandardScaler
- Train-Test Split

## Model

### Random Forest Classifier

Chosen because:

- Strong baseline performance
- Handles nonlinear relationships
- Robust to noise
- Provides feature importance scores

---

# 📈 Model Performance

## Classification Report

```text
Accuracy: 98%

Precision: 92%
Recall: 81%
F1 Score: 85%
```

## ROC-AUC Score

```text
0.9647
```

## Cross Validation

```text
5-Fold Cross Validation Performed
```

---

# 📉 Confusion Matrix

Example:

```text
[[1914   18]
 [  18   50]]
```

Interpretation:

- 1914 Correct Non-Failures
- 50 Correct Failures
- 18 False Positives
- 18 False Negatives

---

# 🔍 Feature Importance

Most influential features:

1. Torque
2. Rotational Speed
3. Tool Wear
4. Air Temperature
5. Process Temperature
6. Machine Type

---

# 🤖 Explainable AI (SHAP)

SHAP (SHapley Additive exPlanations) was used to explain model predictions.

Benefits:

- Understand model decisions
- Identify important features
- Improve transparency
- Increase trust in predictions

---

# 🚀 REST API

Built using FastAPI.

## Endpoints

### Home

```http
GET /
```

Response:

```json
{
  "message": "Production Line Performance Checker"
}
```

---

### Predict Failure

```http
POST /predict
```

Request:

```json
{
  "Type": 1,
  "Air_temperature": 300,
  "Process_temperature": 310,
  "Rotational_speed": 1500,
  "Torque": 40,
  "Tool_wear": 10
}
```

Response:

```json
{
  "machine_failure": 0,
  "failure_probability": 0.0213
}
```

---

# 🐳 Docker Support

## Build Docker Image

```bash
docker build -t production-line-checker .
```

## Run Container

```bash
docker run -p 8000:8000 production-line-checker
```

Open:

```text
http://localhost:8000/docs
```

---

# ☁️ Cloud Deployment

Deployed on Render.

Production URL:

```text
https://production-line-performance-checker.onrender.com
```

Swagger UI:

```text
https://production-line-performance-checker.onrender.com/docs
```

---

# 📂 Project Structure

```text
Production-Line-Performance-Checker/
│
├── app/
│   └── main.py
│
├── data/
│   └── ai4i2020.csv
│
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── feature_importance.png
│   ├── roc_curve.png
│   └── shap_summary.png
│
├── src/
│   ├── train.py
│   └── explain.py
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .dockerignore
```

---

# 🛠️ Tech Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-Learn

### Explainability

- SHAP

### Visualization

- Matplotlib

### Backend

- FastAPI
- Uvicorn

### Deployment

- Docker
- Render

### Version Control

- Git
- GitHub

---

# 📚 Key Learnings

Through this project I learned:

- End-to-end machine learning workflows
- Feature engineering and preprocessing
- Model evaluation and validation
- Explainable AI using SHAP
- API development with FastAPI
- Docker containerization
- Cloud deployment with Render
- Git and GitHub workflows

---

# 🔮 Future Improvements

- XGBoost implementation
- Hyperparameter tuning
- CI/CD using GitHub Actions
- Monitoring and logging
- Kubernetes deployment
- Real-time streaming predictions
- Interactive frontend dashboard

---

# 👨‍💻 Author

**Anish Gutti**

GitHub:
https://github.com/Anish-185

Project Repository:
https://github.com/Anish-185/Production-Line-Performance-Checker

---

## ⭐ If you found this project useful, consider giving it a star on GitHub!
