import urllib.request, json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.src.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras

# load json and create model
json_file = open('compiled-data/BTCUSDT.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("compiled-data/BTCUSDT.h5")
print("Loaded model from disk")

early_stopping = EarlyStopping(monitor='loss',
                                   patience=3,
                                   mode='min')
# evaluate loaded model on test data
optimizer = keras.optimizers.Adam(learning_rate=0.01)
loaded_model.compile(loss = "mean_squared_error", optimizer = optimizer, metrics = ["mae"])

#importing test data and reshape for usage
# with urllib.request.urlopen("http://binancepredalgo.000webhostapp.com/Binance/model/BTCUSDT.json") as url:
#     data = json.load(url)
#
# df = pd.DataFrame(data)
# df.head()
# df = df.drop(['a','q','f','l','m','M'], axis=1)
#
# df.columns=["p","T"]
# df.head()

with open("../model/BTCUSDT.json") as datafile:
    data = json.load(datafile)
df = pd.DataFrame(data)

# convert cloumns to respective data types
df['Open'] = df['Open'].astype(float)
df['High'] = df['High'].astype(float)
df['Low'] = df['Low'].astype(float)
df['Close'] = df['Close'].astype(float)
df['Volume_(Coin)'] = df['Volume_(Coin)'].astype(float)
df['Volume_(Currency)'] = df['Volume_(Currency)'].astype(float)

df.head()
df = df.drop(['Open','High','Low','Volume_(Coin)','Volume_(Currency)'], axis=1)

# updating the header
df.columns=['Timestamp','Close']
df.head()

scaler = MinMaxScaler()
time_step = 40
n_forecast = 15  # length of output sequences (forecast period)

X_input = df.iloc[-time_step:].Close.values
X_input = scaler.fit_transform(X_input.reshape(-1,1))
X_input = np.reshape(X_input, (1,time_step,1))

LSTM_prediction = scaler.inverse_transform(loaded_model.predict(X_input).reshape(-1,1))

# print(LSTM_prediction)

# organize the results in a data frame
df_past = df
df_past.rename(columns={'Close': 'price', 'Timestamp': 'timestamp'}, inplace=True)
df_past['timestamp'] = pd.to_datetime(df_past['timestamp'] * 1000000, unit='ns')
df_past['forecast'] = np.nan
df_past['forecast'].iloc[-1] = df_past['price'].iloc[-1]

df_future = pd.DataFrame(columns=['timestamp', 'forecast'])
df_future['timestamp'] = pd.date_range(df_past['timestamp'].iloc[-1], periods=n_forecast, freq='1M')
df_future['forecast'] = LSTM_prediction.flatten()

results = pd.DataFrame(df_future, columns=['timestamp', 'forecast'])

plt.plot(results)
plt.show()

# Serializing json and save
results.to_json('../model/BTCUSTD-forecast.json', orient='records')
