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
from sklearn.utils import resample
from datetime import datetime
from joblib import load
from sklearn.exceptions import DataConversionWarning

train_test_val_path = "./trainingsdaten/train_data.pkl"
data_path = "./trainingsdaten/"

if not os.path.exists(train_test_val_path):
    print("Erzeugen der Datensplits...")

    data = pd.read_csv('./promstehend3.csv')
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%H:%M:%S.%f')

 
    data['Timestamp'] = data['Timestamp'].dt.time


    time_pairs = [
        ("16:05:27.472", "16:05:28.351"),
        ("16:05:33.330", "16:05:34.103"),
        ("16:05:40.475", "16:05:41.304"),
        ("16:05:42.524", "16:05:43.310"),
        ("16:05:46.325", "16:05:47.252"),
        ("16:05:49.302", "16:05:50.134"),
        ("16:05:52.908", "16:05:53.687"),
        ("16:05:55.346", "16:05:56.671"),
        ("16:06:02.030", "16:06:03.006"),
        ("16:06:04.562", "16:06:05.392"),
        ("16:06:10.118", "16:06:10.950"),
        ("16:06:16.699", "16:06:17.520"),
        ("16:06:25.122", "16:06:25.991"),
        ("16:06:55.390", "16:06:56.214"),
        ("16:06:58.504", "16:06:59.383"),
        ("16:07:05.825", "16:07:06.660"),
        ("16:07:12.938", "16:07:13.715"),
        ("16:07:21.698", "16:07:22.575"),
        ("16:07:26.908", "16:07:27.841"),
        ("16:07:28.234", "16:07:29.057"),
        ("16:07:33.060", "16:07:33.927"),
        ("16:07:45.330", "16:07:46.260"),
        ("16:07:47.332", "16:07:48.211"),
        ("16:08:00.776", "16:08:01.640"),
        ("16:08:28.459", "16:08:29.379"),
        ("16:08:33.775", "16:08:34.549"),
        ("16:08:40.588", "16:08:41.494"),
        ("16:08:41.908", "16:08:42.845"),
        ("16:08:48.536", "16:08:49.424"),
        ("16:08:51.609", "16:08:52.540"),
        ("16:08:53.279", "16:08:54.945"),
        ("16:08:56.311", "16:08:57.390"),
        ("16:08:58.476", "16:08:59.318"),
        ("16:09:00.629", "16:09:01.505"),
        ("16:09:03.548", "16:09:04.481"),
        ("16:09:05.019", "16:09:05.798"),
        ("16:09:09.011", "16:09:09.885"),
        ("16:09:43.428", "16:09:44.368"),
        ("16:09:57.119", "16:09:58.051"),
        ("16:10:12.363", "16:10:13.284"),
        ("16:10:14.560", "16:10:15.838"),
        ("16:10:18.323", "16:10:19.111"),
        ("16:10:21.927", "16:10:22.789"),
        ("16:10:26.162", "16:10:27.034"),
        ("16:10:51.219", "16:10:52.162"),
        ("16:11:32.988", "16:11:33.917"),
        ("16:11:37.669", "16:11:38.654"),
        ("16:11:39.757", "16:11:40.634"),
        ("16:11:44.795", "16:11:45.712"),
        ("16:11:46.359", "16:11:47.236"),
        ("16:11:56.332", "16:11:57.267"),
        ("16:11:58.383", "16:11:59.656"),
        ("16:12:00.156", "16:12:00.971"),
        ("16:12:01.758", "16:12:02.701"),
        ("16:12:05.569", "16:12:06.600"),
        ("16:12:32.541", "16:12:33.474"),
        ("16:12:56.672", "16:12:57.558"),
        ("16:13:03.006", "16:13:03.931"),
        ("16:13:05.997", "16:13:08.112"),
        ("16:13:08.338", "16:13:09.022"),
        ("16:13:10.868", "16:13:11.755"),
        ("16:13:12.092", "16:13:13.023"),
        ("16:13:17.596", "16:13:18.432"),
        ("16:13:19.026", "16:13:20.162"),
        ("16:13:23.868", "16:13:24.776"),
        ("16:13:25.027", "16:13:26.440"),
    ]
    '''
    time_pairs = [
        ("11:46:00.018","11:46:01.892"),
        ("11:46:02.460","11:46:04.152"),
        ("11:46:05.167","11:46:06.384"),
        ("11:46:11.396","11:46:12.900"),
        ("11:46:13.650","11:46:14.408"),
        ("11:46:15.170","11:46:16.590"),
        ("11:46:17.900","11:46:18.962"),
        ("11:46:19.355","11:46:20.384"),
        ("11:46:21.405","11:46:22.431"),
        ("11:46:23.700","11:46:24.740"),
        ("11:46:25.500","11:46:26.734"),
        ("11:46:27.802","11:46:28.200"),
        ("11:46:33.600","11:46:34.970"),
        ("11:46:38.000","11:46:39.100"),
        ("11:46:41.800","11:46:42.400"),
        ("11:46:43.780","11:46:44.400"),
        ("11:46:45.650","11:46:46.400"),
        ("11:46:47.670","11:46:48.861"),
        ("11:46:50.065","11:46:51.222"),
        ("11:46:53.352","11:46:53.800"),
        ("11:47:03.400","11:47:04.000"),
        ("11:47:07.300","11:47:07.960"),
        ("11:47:09.600","11:47:09.800"),
        ("11:47:10.420","11:47:11.670"),
        ("11:47:12.500","11:47:12.850"),
        ("11:47:27.400","11:47:27.800"),
        ("11:47:30.450","11:47:31.100"),
        ("11:47:31.650","11:47:32.944"),
        ("11:47:33.700","11:47:34.670"),
        ("11:47:40.870","11:47:41.999"),
        ("11:47:42.900","11:47:44.000"),
        ("11:47:46.210","11:47:46.897"),
        ("11:47:57.200","11:47:58.510"),
        ("11:47:59.250","11:48:00.100"),
        ("11:48:01.333","11:48:02.200"),
        ("11:48:06.300","11:48:08.236"),
        ("11:48:09.320","11:48:10.364"),
        ("11:48:14.760","11:48:15.632"),
        ("11:48:16.800","11:48:17.386"),
        ("11:48:18.170","11:48:18.893"),
        ("11:48:21.200","11:48:22.617"),
        ("11:48:23.300","11:48:24.056"),
        ("11:48:24.700","11:48:25.566"),
        ("11:48:31.415","11:48:32.198"),
        ("11:48:33.600","11:48:34.045"),
        ("11:48:38.858","11:48:39.846"),
        ("11:48:47.550","11:48:48.200"),
        ("11:48:54.578","11:48:55.510"),
        ("11:48:56.700","11:48:57.405"),
        ("11:49:21.598","11:49:23.000"),
        ("11:49:23.800","11:49:25.286"),
        ("11:49:26.318","11:49:27.333"),
        ("11:49:31.971","11:49:33.300"),
        ("11:49:33.986","11:49:34.790"),
        ("11:49:35.689","11:49:36.579"),
        ("11:49:37.745","11:49:38.922"),
        ("11:49:39.255","11:49:40.155"),
        ("11:49:40.800","11:49:41.815"),
        ("11:49:42.881","11:49:43.927"),
        ("11:49:44.603","11:49:45.675"),
        ("11:49:46.697","11:49:47.188"),
        ("11:49:51.972","11:49:53.199"),
        ("11:49:56.114","11:49:56.900"),
        ("11:49:59.400","11:50:00.189"),
        ("11:50:01.306","11:50:01.893"),
        ("11:50:02.850","11:50:03.743"),
        ("11:50:04.958","11:50:05.907"),
        ("11:50:07.128","11:50:08.153"),
        ("11:50:10.000","11:50:10.100"),
        ("11:50:18.500","11:50:19.096"),
        ("11:50:21.815","11:50:22.744"),
        ("11:50:24.670","11:50:25.533"),
        ("11:50:43.666","11:50:44.647"),
        ("11:50:45.600","11:50:46.061"),
        ("11:50:47.568","11:50:48.412"),
        ("11:50:58.630","11:50:59.675"),
        ("11:50:59.950","11:51:00.954"),
        ("11:51:01.874","11:51:02.414"),
        ("11:51:03.299","11:51:03.938"),
        ("11:51:23.550","11:51:24.350"),
        ("11:51:33.824","11:51:34.315"),
        ("11:51:34.800","11:51:35.500"),
        ("11:51:44.000","11:51:45.215"),
        ("11:51:49.062","11:51:49.700"),
        ("11:51:59.152","11:52:00.386"),
        ("11:52:01.158","11:52:02.003"),
        ("11:52:03.317","11:52:04.096"),
        ("11:52:07.900","11:52:09.025"),
        ("11:52:10.734","11:52:11.660"),
        ("11:52:15.960","11:52:16.999"),
        ("11:52:17.389","11:52:18.358"),
        ("11:52:18.890","11:52:19.745"),
        ("11:52:31.852","11:52:33.160"),
        ("11:52:33.893","11:52:34.817"),
        ("11:52:39.704","11:52:40.777"),
        ("11:52:41.752","11:52:42.500"),
        ("11:52:55.066","11:52:55.959"),
        ("11:52:56.768","11:52:57.415"),
    ]
    if len(X[index]) >= 8: 
                    
                    for j in range(0, len(X[index]), 8): 
                        X[index, j:j+8] = X[index, j:j+8][::-1]
    '''
    

    X = data.iloc[:, :-2].values
    for index, row in data.iterrows():
        for start, end in time_pairs:
            start_datetime = pd.to_datetime(start, format='%H:%M:%S.%f')

            
            start_time = start_datetime.time()
            end_datetime = pd.to_datetime(end, format='%H:%M:%S.%f')
            end_time = end_datetime.time()
        
            if start_time <= row['Timestamp'] <= end_time:
                data.at[index, 'Label'] = 1
    
                

    # Daten für Features und Labels aufteilen
    #X = data.iloc[:, :-2].values
 
    # Unter-Sampling der überrepräsentierten Klasse
    data2 = pd.read_csv('./coesfeld.csv')
    data2['Timestamp'] = pd.to_datetime(data2['Timestamp'], format='%H:%M:%S.%f')

 
    data2['Timestamp'] = data2['Timestamp'].dt.time
    class_0 = data[data['Label'] == 0]
    class_0 = pd.concat([class_0, data2])
    class_1 = data[data['Label'] == 1]

    # Stelle sicher, dass Klasse 1 die unterrepräsentierte Klasse ist

    class_0_downsampled = class_0

    data_balanced = pd.concat([class_0_downsampled, class_1])

    
    # Jetzt Daten für Features und Labels aufteilen
    X = data_balanced.iloc[:, :-2].values
    y = data_balanced.iloc[:, -1].values

    unique_values, counts = np.unique(y, return_counts=True)
    value_counts = dict(zip(unique_values, counts))
    print(unique_values)
    print(value_counts) 

    # Daten normalisieren
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Daten in das erforderliche Format umwandeln
    X = X.reshape((-1, 20, 64))

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


# Öffnen der geseicherten Trainings-/Test-/Valdaten
with open(data_path + 'train_data.pkl', 'rb') as file:
    X_train, y_train = pickle.load(file)

with open(data_path + 'test_data.pkl', 'rb') as file:
    X_test, y_test = pickle.load(file)

with open(data_path + 'val_data.pkl', 'rb') as file:
    X_val, y_val = pickle.load(file)


print("Daten geladen...")

# Modell aufbauen
model = Sequential()
model.add(LSTM(10, return_sequences=False, input_shape=(20, 64)))  
model.add(Dense(1, activation='sigmoid'))  # Sigmoid-Aktivierung für binäre Klassifikation

# Modell kompilieren
print("Starte Training...")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')])

# Modell trainieren
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Modell evaluieren
metrics = model.evaluate(X_val, y_val, return_dict=True)
print(metrics)

#dump(scaler, 'scaler2.joblib')
# Model speichern
model.save('model.keras')






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

Modell aufbauen
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(20, 64))) 
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid-Aktivierung für binäre Klassifikation

# Modell kompilieren
print("Starte Training...")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

 Modell trainieren
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

metrics = model.evaluate(X_test, y_test, return_dict=True)
print(metrics)

#dump(scaler, 'scaler2.joblib')
model.save(model, 'model2')
