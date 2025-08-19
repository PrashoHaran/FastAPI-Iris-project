# 🌸 ML FastAPI – Iris Flower Classification  

This project is a **FastAPI-based ML inference service** that predicts the species of an Iris flower given its sepal and petal measurements.  

---

## 📘 Problem Description  
The **Iris dataset** is a classic dataset in machine learning containing measurements of **sepal length, sepal width, petal length, and petal width** for three species of Iris flowers (*Setosa, Versicolor, Virginica*).  
The task is to build a machine learning model that classifies a flower into one of the three species based on these four features.  

---

## 🤖 Model Choice & Justification  
We used a **Logistic Regression** classifier (with preprocessing) because:  
- It’s a **simple yet powerful model** for multiclass classification.  
- Works well on small datasets like Iris.  
- Lightweight and **fast to serve in real-time inference** via API.  

The model was trained using **scikit-learn**, and then saved (`model.pkl`) for serving in FastAPI.  

---

## 📂 Project Structure  
```
ML-FASTAPI-IRIS/
│
├── app/
│   └── main.py          # FastAPI app
│
├── models/
│   └── model.pkl        # Saved trained model
│
├── train.py             # Training script
├── requirements.txt     # Dependencies
├── README.md            # Documentation
└── .gitignore
```

---

## ⚙️ How to Run the Application  

### 1. Clone the Repository  
```bash
git clone https://github.com/<your-username>/ML-FASTAPI-IRIS.git
cd ML-FASTAPI-IRIS
```

### 2. Create Virtual Environment & Install Dependencies  
```bash
python -m venv .venv
.venv\Scripts\activate      # On Windows
source .venv/bin/activate   # On Mac/Linux

pip install -r requirements.txt
```

### 3. Train the Model  
```bash
python train.py
```
This will save the trained model into `models/model.pkl`.

### 4. Start FastAPI Server  
```bash
uvicorn app.main:app --reload
```

### 5. Open Interactive API Docs  
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  

---

## 📌 Example API Usage  

### Health Check  
```http
GET /
```
**Response:**
```json
{
  "status": "ok",
  "message": "Iris Model API is running"
}
```

### Prediction  
```http
POST /predict
Content-Type: application/json

{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**
```json
{
  "prediction": "setosa",
  "confidence": 0.98
}
```

### Model Info  
```http
GET /model-info
```
**Response:**
```json
{
  "model_type": "LogisticRegression",
  "problem_type": "classification",
  "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
  "classes": ["setosa", "versicolor", "virginica"]
}
```

---

## 🏆 Deliverables  
- **main.py** → FastAPI application (`app/main.py`)  
- **model.pkl** → Saved model (`models/model.pkl`)  
- **requirements.txt** → Dependencies  
- **README.md** → Documentation  
