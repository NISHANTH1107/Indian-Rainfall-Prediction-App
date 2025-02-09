# India Rainfall Prediction App 🌧️

A machine learning application that predicts monthly rainfall across different states in India using historical data from 1901-2015.

## 🌐 Live Demo
Try out the application here: [India Rainfall Predictor](https://rainfall-prediction-india.streamlit.app)

## 📌 Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)

## ✨ Features
- Predicts next month's rainfall based on previous 3 months' data
- Covers all states and union territories in India
- Interactive visualization of rainfall patterns
- User-friendly interface built with Streamlit
- Random Forest model with high accuracy
- Visual representation of predictions

## 🛠️ Installation

1. Clone the repository
```bash
git clone https://github.com/NISHANTH1107/Indian-Rainfall-Prediction-App.git
cd Indian-Rainfall-Prediction-App
```

2. Create a Virtual Environment (optional)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install Requirements
```bash
pip install -r requirements.txt
```

## Usage
1. Train the model
```bash
python indian_rainfall_prediction.py
```

2. Run Streamlit
```bash
streamlit run app.py
```

3. Open your browser and navigate to (http://localhost:8501)

## 📊 Dataset
The dataset used in this project is "Rainfall in India 1901-2015" which contains monthly rainfall data for different states and union territories of India. The data has been collected over a period of more than 100 years, making it a robust source for prediction.

### Data Features:

- SUBDIVISION: States/Union Territories
- YEAR: 1901-2015
- Monthly rainfall data (JAN to DEC)
- Annual rainfall

## 🤖 Model Architecture

- Algorithm: Random Forest Regressor
- Features: Rolling window of 3 months
- Train-Test Split: 80-20
- Hyperparameters:

- n_estimators: 100
- max_depth: 10
- n_jobs: 1



