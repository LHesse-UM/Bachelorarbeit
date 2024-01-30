import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import cv2
import numpy as np
import serial
import time
from datetime import datetime
import csv
from joblib import load
import warnings
from sklearn.exceptions import DataConversionWarning


# Dieser Code ist für das Testen des Modells zuständig. 
# Er ist nahezu äquivalent zu collectData.py, kann jedoch in Echtzeit das Modell testen, anstatt die Daten zu speichern


# Aufbau der seriellen Schnittstelle
# Port und Baudrate ggfs. anpassen
ser = serial.Serial('COM4', 115200)


# Einladen des Modells
model = load('model.joblib')
scaler = load('scaler.joblib')

# Variablendefinition
matritzen = []

warnings.filterwarnings('ignore')

def get_sensor_data(timestamp):
    sensor_data = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        # Auslesen der Daten
        data = ser.readline().decode('utf-8').strip()
        if data:

            # Vorverarbeiten der Daten
            value_groups = data.split(";")
            distances = [int(group.split(",")[0]) if group.split(",")[0] else 0 for group in value_groups]
            distances = distances[:-1]
        
            if len(distances) == 64:

                # Umwandlung der Daten in ein 8*8-Format
                my_sensor_data = np.array(distances).reshape((8, 8))
                block_size_x = sensor_data.shape[1] // 8
                block_size_y = sensor_data.shape[0] // 8
                
                for i in range(8):
                    for j in range(8):
                        value = my_sensor_data[i, j]

                        # Störwerte rausfiltern
                        if value > 2000 or value == 0:
                            value = 0
                            my_sensor_data[i, j] = 0
                        else: 

                        # Farbwert berechnen
                        color_value = int(value * 255 / 2000)
                        sensor_data[i*block_size_y:(i+1)*block_size_y, j*block_size_x:(j+1)*block_size_x] = [color_value, color_value, color_value]

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5  
                        font_color = (255-color_value, 255-color_value, 255-color_value) 
                        font_thickness = 1  

                        text = f"{value}"
                        
                        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                        text_x = j * block_size_x + (block_size_x - text_size[0]) // 2
                        text_y = i * block_size_y + (block_size_y + text_size[1]) // 2

                       
                        cv2.putText(sensor_data, text, (text_x, text_y), font, font_scale, font_color, font_thickness)
                        
                matritzen.append(my_sensor_data.flatten())
                # Überprüfe, ob die Queue 20 Matrizen enthält
                if len(matritzen) == 20:
                        
                    row = np.concatenate(matritzen)
                    new_sample_scaled = scaler.transform(row.reshape(1, -1))  # Reshape zu (1, 1280)

                    
                    # Reshape für das RNN-Modell
                    new_sample_scaled = new_sample_scaled.reshape(1, 20, 64)


                    # Modellprediction erhalten und ausgeben
                    prediction = model.predict(new_sample_scaled,verbose=0)
                    predicted_class = (prediction > 0.5).astype(int)
                    #print(predicted_class) 
                    print(prediction)

                    
                    matritzen.pop(0)
                
                            
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        
    return sensor_data

# Initialisieren der Kamera
# Ggfs. Zahl anpassen, wenn keine Webcam genutzt wird
cap = cv2.VideoCapture(1)


while True:

    # Zeitstempel erhalten
    now = datetime.now()
    timestamp = now.strftime("%H:%M:%S") + ".{:03d}".format(int(now.microsecond / 1000))

    # Kamerabild
    ret, camera_frame = cap.read()

    # Sensorbild
    sensor_frame = get_sensor_data(timestamp)
    
    
    #Einfügen des Timestamps in das Sensorbild
    cv2.putText(sensor_frame, timestamp, (10, sensor_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    if sensor_frame.shape[0] != camera_frame.shape[0]:
        sensor_frame = cv2.resize(sensor_frame, (sensor_frame.shape[1], camera_frame.shape[0]))
    
    combined_frame = np.hstack((camera_frame, sensor_frame))

    
    cv2.imshow('Camera and Sensor Data', combined_frame)

    # Beenden des Skrips mit 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
