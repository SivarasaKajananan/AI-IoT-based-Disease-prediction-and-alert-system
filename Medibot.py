import smbus2
import time
import Adafruit_ADS1x15
import serial
import pynmea2
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import autosklearn.classification
import pandas as pd
from sklearn.model_selection import train_test_split

# Firebase initialization
cred = credentials.Certificate('path/to/serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': '########.firebaseio.com/'
})
ref = db.reference('medibot')

# I2C bus for MLX90614
bus = smbus2.SMBus(1)
MLX90614_ADDRESS = 0x5A
MLX90614_TEMP_REGISTER = 0x07

# ADC for Pulse sensor
adc = Adafruit_ADS1x15.ADS1115()
GAIN = 1

# Serial for GPS
gps_serial = serial.Serial('/dev/ttyAMA0', 9600, timeout=1)

# Auto-Sklearn model initialization
automl = autosklearn.classification.AutoSklearnClassifier()

df = pd.read_csv('disease_dataset.csv')
X = df.drop('disease', axis=1)
y = df['disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
automl.fit(X_train, y_train)

def read_temp():
    temp_raw = bus.read_word_data(MLX90614_ADDRESS, MLX90614_TEMP_REGISTER)
    temp_celsius = temp_raw * 0.02 - 273.15
    return temp_celsius

def read_pulse():
    pulse_value = adc.read_adc(0, gain=GAIN)
    return pulse_value

def read_gps():
    while True:
        data = gps_serial.readline().decode('ascii', errors='replace')
        if data.startswith('$GPGGA'):
            msg = pynmea2.parse(data)
            return {'latitude': msg.latitude, 'longitude': msg.longitude}

def upload_data(data):
    ref.push(data)

def send_sos(temperature, pulse, gps_data):
    sos_data = {
        'type': 'SOS',
        'temperature': temperature,
        'pulse': pulse,
        'latitude': gps_data['latitude'],
        'longitude': gps_data['longitude'],
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    ref.child('sos_alerts').push(sos_data)

def main():
    while True:
        temperature = read_temp()
        pulse = read_pulse()
        gps_data = read_gps()
        
        # Prepare data for prediction
        input_data = pd.DataFrame({
            'temperature': [temperature],
            'pulse': [pulse],
            # Add more features if necessary
        })
        
        prediction = automl.predict(input_data)[0]
        
        data = {
            'temperature': temperature,
            'pulse': pulse,
            'latitude': gps_data['latitude'],
            'longitude': gps_data['longitude'],
            'prediction': prediction
        }
        
        upload_data(data)

        
        if temperature > 38.0 or pulse > 100:  
            send_sos(temperature, pulse, gps_data)
        
        time.sleep(60)  

if __name__ == '__main__':
    main()
