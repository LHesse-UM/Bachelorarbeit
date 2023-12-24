import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
from datetime import datetime
from joblib import dump
import time
import pickle
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

from datetime import datetime
from joblib import load
from sklearn.exceptions import DataConversionWarning

train_test_val_path = "./trainingsdaten/train_data.pkl"
data_path = "./trainingsdaten/"

if not os.path.exists(train_test_val_path):
    print("Erzeugen der Datensplits...")

    data = pd.read_csv('./trainingsdaten/trainingsdaten.csv')
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])

    time_pairs = [
        ("16:54:32.105","16:54:33.889"),
        ("16:54:35.068","16:54:37.002"),
        ("16:54:38.378","16:54:40.211"),
        ("16:54:42.082","16:54:43.769"),
        ("16:54:45.002","16:54:46.835"),
        ("16:54:49.099","16:54:50.786"),
        ("16:54:52.801","16:54:54.587"),
        ("16:54:55.868","16:54:57.698"),
        ("16:54:59.618","16:55:01.604"),
        ("16:55:03.080","16:55:04.864"),
        ("16:55:06.686","16:55:08.520"),
        ("16:55:10.043","16:55:11.827"),
        ("16:55:14.189","16:55:16.073"),
        ("16:55:17.697","16:55:19.482"),
        ("16:55:21.201","16:55:23.086"),
        ("16:55:24.856","16:55:26.890"),
        ("16:55:29.104","16:55:31.134"),
        ("16:55:32.415","16:55:34.198"),
        ("16:55:36.362","16:55:38.247"),
        ("16:55:39.870","16:55:41.705"),
        ("16:55:43.574","16:55:45.360"),
        ("16:55:47.821","16:55:49.755"),
        ("16:55:50.495","16:55:52.131"),
        ("16:55:55.575","16:55:57.361"),
        ("16:55:59.232","16:56:00.967"),
        ("16:56:03.377","16:56:05.314"),
        ("16:56:07.626","16:56:09.462"),
        ("16:56:11.675","16:56:13.413"),
        ("16:56:15.724","16:56:17.560"),
        ("16:56:19.727","16:56:21.463"),
        ("16:56:23.723","16:56:25.360"),
        ("16:56:27.527","16:56:29.211"),
        ("16:56:30.343","16:56:31.928"),
        ("16:56:33.552","16:56:35.138"),
        ("16:56:37.208","16:56:38.840"),
        ("16:56:39.576","16:56:41.262"),
        ("16:56:42.936","16:56:44.621"),
        ("16:56:45.213","16:56:46.799"),
        ("16:56:48.666","16:56:50.202"),
        ("16:56:50.892","16:56:52.426"),
        ("16:56:54.246","16:56:55.882"),
        ("16:56:56.619","16:56:58.203"),
        ("16:57:00.123","16:57:01.707"),
        ("16:57:02.643","16:57:04.229"),
        ("16:57:05.902","16:57:07.485"),
        ("16:57:08.815","16:57:10.602"),
        ("16:57:11.980","16:57:13.566"),
        ("16:57:16.126","16:57:17.911"),
        ("16:57:18.746","16:57:20.532"),
        ("16:57:23.137","16:57:24.728"),
        ("16:57:26.300","16:57:27.785"),
        ("16:57:29.015","16:57:30.554"),
        ("16:57:32.423","16:57:33.861"),
        ("16:57:34.990","16:57:36.475"),
        ("16:57:38.050","16:57:39.687"),
        ("16:57:40.719","16:57:42.204"),
        ("16:57:44.176","16:57:45.762"),
        ("16:57:46.355","16:57:47.892"),
        ("16:57:50.053","16:57:51.690"),
        ("16:57:52.328","16:57:53.865"),
        ("16:57:56.079","16:57:57.666"),
        ("16:57:58.602","16:58:00.136"),
        ("16:58:02.551","16:58:04.033"),
        ("16:58:04.869","16:58:06.307")
    ]

    for index, row in data.iterrows():
        for start, end in time_pairs:
            if pd.to_datetime(start) <= row['Timestamp'] < pd.to_datetime(end):
                data.at[index, 'Label'] = 1

    # Daten für Features und Labels aufteilen
    X = data.iloc[:, :-2].values
    y = data.iloc[:, -1].values

    # Daten normalisieren
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Daten in das erforderliche Format umwandeln
    X = X.reshape((-1, 20, 64))  # jede Sequenz hat 20 Zeitschritte und jeder Zeitschritt hat 64 Werte

    # Daten in Trainings- und Testsets aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    
    # speichern der Daten um Zeit zu sparen
    with open(data_path + 'train_data.pkl', 'wb') as file:
        pickle.dump((X_train, y_train), file)

    with open(data_path +'test_data.pkl', 'wb') as file:
        pickle.dump((X_test, y_test), file)

    with open(data_path + 'val_data.pkl', 'wb') as file:
        pickle.dump((X_val, y_val), file)


# Oeffnen der geseicherten Trainings-/Test-/Valdaten
with open(data_path + 'train_data.pkl', 'rb') as file:
    X_train, y_train = pickle.load(file)

with open(data_path + 'test_data.pkl', 'rb') as file:
    X_test, y_test = pickle.load(file)

with open(data_path + 'val_data.pkl', 'rb') as file:
    X_val, y_val = pickle.load(file)


print("Daten geladen...")

# Schritt 2: Modell aufbauen
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(20, 64)))  # Anpassen an deine Daten
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid-Aktivierung für binäre Klassifikation

# Modell kompilieren
print("Starte Training...")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Schritt 3: Modell trainieren
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

metrics = model.evaluate(X_test, y_test, return_dict=True)
print(metrics)

#dump(scaler, 'scaler2.joblib')
model.save(model, 'model2')
