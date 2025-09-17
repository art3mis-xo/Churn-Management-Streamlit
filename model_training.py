# dataset link: https://www.kaggle.com/datasets/filippoo/deep-learning-az-ann?resource=download
#model_training.py
# This script trains a neural network model for churn prediction using the Churn_Modelling dataset.
# It preprocesses the data, trains the model, and saves it for later use.
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

file_name = "Churn_Modelling.csv"
# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "filippoo/deep-learning-az-ann",
  file_name,
)
#print("=========================================1 loaded dataset")
print("First 5 records:\n", df.head())
#print("=========================================1.5")
print(df['Geography'].unique())
# Preprocess the data
df = df.drop(["RowNumber", "CustomerId", "Surname"],axis=1)
#print("=========================================2")
print("Any null values:\n",df.isna().sum())
#print("=========================================3")
print(df.describe())
#print("=========================================4")
print(df.info())
#print("=========================================5")
print(df.columns)
#print("=========================================6")
# le = LabelEncoder()
# df["Geography"] = le.fit_transform(df['Geography'])
# df["Gender"] = le.fit_transform(df['Gender'])
geo_encoder = LabelEncoder()
gender_encoder = LabelEncoder()
df["Geography"] = geo_encoder.fit_transform(df["Geography"])
df["Gender"] = gender_encoder.fit_transform(df["Gender"])

with open("geo_encoder.pkl", "wb") as f:
    pickle.dump(geo_encoder, f)
with open("gender_encoder.pkl", "wb") as f:
    pickle.dump(gender_encoder, f)
#print("=========================================7")
sc = StandardScaler()
df[["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]] = sc.fit_transform(
    df[["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]]
)
#print("=========================================8")
x = df.drop("Exited",axis=1)
y = df['Exited']
#print("=========================================9")
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)
#print("=========================================10")

def model_creation_ann():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(10,)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

my_model = model_creation_ann()
my_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
history = my_model.fit(x_train, y_train, epochs=10)
results = my_model.evaluate(x_test, y_test)
#print("=========================================11")
print("test loss = ", results[0])
print("text acc: ", results[1])
#print("=========================================12")
prediction = my_model.predict(x_test[:3])
print(prediction)
print(x_test[:3])
print(type(x_test))
#print("=========================================13")
my_model.save("trained_ann_model.keras")
#print("=========================================14")
