import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

#importing training data and reshape for usage
with open("../model/BTCUSDT.json") as datafile:
    data = json.load(datafile)
df = pd.DataFrame(data)

training_data = df.iloc[:, 1].values

scaler = MinMaxScaler()

training_data = scaler.fit_transform(training_data.reshape(-1, 1))

x_training_data = []
y_training_data = []

timesteps = int(len(training_data) / 2)

for i in range(timesteps, len(training_data)):
    x_training_data.append(training_data[i-timesteps:i, 0])
    y_training_data.append(training_data[i, 0])

x_training_data = np.array(x_training_data)
y_training_data = np.array(y_training_data)

x_training_data = np.reshape(x_training_data, (x_training_data.shape[0],x_training_data.shape[1],1))

#building model
rnn = tf.keras.models.Sequential()
rnn.add(tf.keras.layers.LSTM(units = 45, return_sequences = True, input_shape = (x_training_data.shape[1], 1)))
rnn.add(tf.keras.layers.Dropout(0.2))

for i in [True, True, False]:
    rnn.add(tf.keras.layers.LSTM(units = 45, return_sequences = i))
    rnn.add(tf.keras.layers.Dropout(0.2))

epochs = 100
batch_size = int(len(training_data) / epochs)

rnn.add(tf.keras.layers.Dense(units = 1))
rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')
rnn.fit(x_training_data, y_training_data, epochs = epochs, batch_size = batch_size)

# serialize model to JSON
model_json = rnn.to_json()
with open("compiled-data/BTCUSDT.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
rnn.save_weights("compiled-data/BTCUSDT.h5")
print("Saved model to disk")