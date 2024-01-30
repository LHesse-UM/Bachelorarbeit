import serial
import time

# Dieser Code ist für das Testen der Sensorkonfiguration 

# Port und Baudrate anpassen
ser = serial.Serial('COM4', 115200)

try:
    while True:
        
        data = ser.readline().decode('utf-8').strip()
        
        if data:
            # Ausgabe der Daten
            print(data)
           

except KeyboardInterrupt:
    
    ser.close()
