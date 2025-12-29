# Credit Card Fraud Detection System

An end-to-end credit card fraud detection system built using classical machine learning, deep learning, ensemble techniques, and deployed as a REST API using FastAPI.

This project demonstrates the complete ML lifecycle — from data preprocessing and model training to real-time inference through an API.

---

## 🚀 Features

- Decision Tree and Random Forest baseline models
- CNN (Conv1D) for deep learning-based fraud detection
- Stacking classifier to combine multiple model predictions
- Threshold-based risk decisioning
- REST API built using FastAPI
- Interactive API testing using Swagger UI

---

## 🧠 Models Used

- **Decision Tree**
- **Random Forest**
- **Convolutional Neural Network (CNN)**
- **Stacking Classifier (Logistic Regression as meta-model)**

The ensemble approach improves robustness by combining heterogeneous models.

---

## 🏗️ Project Structure

```
credit-card-fraud/
│
├── api/
│ ├── app.py # FastAPI application
│ ├── random_forest.pkl
│ ├── decision_tree.pkl
│ ├── stacking_model.pkl
│ ├── scaler.pkl
│ └── cnn_model.keras
│
├── notebook/
│ └── model_training.ipynb # Data preprocessing & model training
│
├── requirements.txt
└── README.md
```


---

## 📊 Dataset

- Dataset: **Credit Card Fraud Detection (Kaggle)**
- Features `V1–V28` are PCA-transformed due to confidentiality
- Highly imbalanced dataset


---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```
git clone https://github.com/<your-username>/credit-card-fraud.git
cd credit-card-fraud
```

### 2️⃣ Create and activate virtual environment
```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```

### ▶️ Run the API
```
cd api
uvicorn app:app --reload
```

### 🧪 Notes on Model Behavior

- Fraud detection prioritizes recall over precision

- Stacking classifier is conservative and requires model agreement

- Synthetic inputs are for demonstration; real fraud validation was performed using dataset samples

### 🔮 Future Improvements

- User authentication and transaction history

- Real-world feature engineering pipeline

- Dockerized deployment

- Frontend dashboard
