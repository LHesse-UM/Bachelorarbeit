import serial
import time

ser = serial.Serial('COM4', 115200)

try:
    while True:
        
        data = ser.readline().decode('utf-8').strip()
        
        if data:
            print(data)
           

except KeyboardInterrupt:
    
    ser.close()
