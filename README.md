# 🔍 Customer Churn Prediction using ANN & Streamlit

This project uses an Artificial Neural Network (ANN) to predict customer churn based on the [Churn_Modelling dataset](https://www.kaggle.com/datasets/filippoo/deep-learning-az-ann). It includes both a training script and a Streamlit web app for interactive predictions.

---

## 📁 Project Files

```
├── model_training.py         # Trains and saves the ANN model
├── app.py                    # Streamlit app for real-time churn prediction
├── trained_ann_model.keras   # Trained ANN model (generated after training)
├── geo_encoder.pkl           # Pickled LabelEncoder for 'Geography'
├── gender_encoder.pkl        # Pickled LabelEncoder for 'Gender'
├── requirements.txt          # List of required Python packages
└── README.md                 # Project documentation (this file)
```

---

## 🚀 Features

- Downloads dataset directly using `kagglehub`
- Preprocesses customer data (encoding + scaling)
- Trains an ANN model with multiple dense and dropout layers
- Saves model and encoders for future use
- Provides a **Streamlit interface** for real-time prediction

---

## 🔧 Installation

Install required packages:

```bash
pip install -r requirements.txt
```

Ensure your **Kaggle API key** is set up in `~/.kaggle/kaggle.json`.  
[How to get Kaggle API Key →](https://www.kaggle.com/account)

---

## 🧪 Step 1: Train the Model

Run the following command to train and save the model:

```bash
python model_training.py
```

This will:
- Download the dataset
- Encode and scale data
- Train the model
- Save the trained `.keras` model and encoder `.pkl` files

---

## 🌐 Step 2: Launch the Streamlit App

After training, run:

```bash
streamlit run app.py
```

This opens a browser window where you can input customer details and get a churn prediction.

---

## 📦 Example Input Values

| Field              | Example |
|-------------------|---------|
| Credit Score       | 600     |
| Geography          | France  |
| Gender             | Female  |
| Age                | 40      |
| Tenure             | 3       |
| Balance            | 60000   |
| Num of Products    | 2       |
| Has Credit Card    | 1       |
| Is Active Member   | 1       |
| Estimated Salary   | 50000   |

---

## 💻 Model Summary

- Input: 10 features
- Layers:
  - Dense(64) → ReLU
  - Dense(128) → ReLU
  - Dropout(0.2)
  - Dense(256) → ReLU
  - Dropout(0.3)
  - Dense(512) → ReLU
  - Dense(1) → Sigmoid

Loss: `binary_crossentropy`  
Optimizer: `adam`

---

## 📄 requirements.txt

```
tensorflow
scikit-learn
pandas
kagglehub
kaggle
streamlit
```

---

## 📌 Dataset

- Source: [Kaggle](https://www.kaggle.com/datasets/filippoo/deep-learning-az-ann)
- File used: `Churn_Modelling.csv`

---

## 📜 License

Free to use for educational and research purposes.
