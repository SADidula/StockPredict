import urllib.request, json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from tensorflow import keras
from keras.callbacks import EarlyStopping

# importing training data and reshape for usage
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
# df.info()

length_data = len(df)     # rows that data has
split_ratio = 0.7           # %70 train + %30 validation
length_train = round(length_data * split_ratio)
length_validation = length_data - length_train

train_data = df[:length_train].iloc[:,:2]
train_data['Timestamp'] = pd.to_datetime(train_data['Timestamp'] * 1000000, unit='ns')

validation_data = df[length_train:].iloc[:,:2]
validation_data['Timestamp'] = pd.to_datetime(validation_data['Timestamp'] * 1000000, unit='ns')  # converting to date time object

dataset_train = train_data.Close.values
dataset_train = np.reshape(dataset_train, (-1,1))

scaler = MinMaxScaler(feature_range = (0,1))

# scaling dataset
dataset_train_scaled = scaler.fit_transform(dataset_train)

X_train = []
y_train = []

time_step = 40

for i in range(time_step, length_train):
    X_train.append(dataset_train_scaled[i - time_step:i, 0])
    y_train.append(dataset_train_scaled[i, 0])

# convert list to array
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
y_train = np.reshape(y_train, (y_train.shape[0],1))

y_train = scaler.inverse_transform(y_train) # scaling back from 0-1 to original

dataset_validation = validation_data.Close.values
dataset_validation = np.reshape(dataset_validation, (-1,1))
scaled_dataset_validation = scaler.fit_transform(dataset_validation)

# Creating X_test and y_test
X_test = []
y_test = []
n_forecast = 15  # length of output sequences (forecast period)

for i in range(time_step, length_validation):
    X_test.append(scaled_dataset_validation[i-time_step:i,0])
    y_test.append(scaled_dataset_validation[i,0])

# Converting to array
X_test, y_test = np.array(X_test), np.array(y_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))  # reshape to 3D array
y_test = np.reshape(y_test, (-1,1))  # reshape to 2D array

y_train = scaler.fit_transform(y_train)

model_lstm = Sequential()
model_lstm.add(LSTM(32,return_sequences=True,input_shape = (X_train.shape[1],1))) #64 lstm neuron block
model_lstm.add(Dropout(0.2))

for i in [False]:
    model_lstm.add(LSTM(units = 32, return_sequences = i))
    # model_lstm.add(Dropout(0.2))

model_lstm.add(Dense(n_forecast, activation='linear'))

optimizer = keras.optimizers.Adam(learning_rate=0.01)

# prediction results

# under units = 32, init_dropout = 0.2, depleted last dropout, 2 neurons
# 0.01 = loss 0.0020, mae 0.0317, result = 4468.4585

# under units = 32, init_dropout = 0.2, continuous_dropout = 0.2, 2 neurons
# 0.01 = loss 0.0026, mae 0.0367, result = 4467.277

# under units = 32, init_dropout = 0.6, continuous_dropout = 0.3, 2 neurons
# 0.01 = loss 0.0032, mae 0.0428, result = 4465.4565

# under units = 32, init_dropout = 0.2, continuous_dropout = 0.2, 3 neurons
# 0.01 = loss 0.0033, mae 0.0438, result = 4467.346

# under units = 50, init_dropout = 0.6, continuous_dropout = 0.3, 3 neurons
# 0.01 = loss 0.0037, mae 0.0452, result = 4467.0635

# under init_units = 32, cont_units = 16, init_dropout = 0.6, continuous_dropout = 0.3, 2 neurons
# 0.01 = loss 0.0047, mae 0.0508, result = 4465.7676

# under units = 50, init_dropout = 0.6, continuous_dropout = 0.3, 5 neurons
# 0.00001 = loss 0.0045, mae = 0.05, result = 4467.353
# 0.01 = loss 0.005, mae = 0.0510, result = 4468.75

# under units = 32, init_dropout = 0.75, continuous_dropout = 0.5, 2 neurons
# 0.01 = loss 0.0054, mae 0.055, result = 4466.745

# under units = 50, init_dropout = 0.6, continuous_dropout = 0.4, 6 neurons
# 0.01 = loss 0.0055, mae 0.0584, result = 4470

early_stopping = EarlyStopping(monitor='loss',
                                   patience=3,
                                   mode='min')

model_lstm.compile(loss = "mean_squared_error", optimizer = optimizer, metrics = ["mae"])
model_lstm.fit(X_train, y_train, epochs = 100, batch_size = 5, callbacks=[early_stopping], shuffle=False)

X_input = df.iloc[-time_step:].Close.values
X_input = scaler.fit_transform(X_input.reshape(-1,1))
X_input = np.reshape(X_input, (1,time_step,1))

LSTM_prediction = scaler.inverse_transform(model_lstm.predict(X_input).reshape(-1,1))

print(LSTM_prediction)

# serialize model to JSON
model_json = model_lstm.to_json()
with open("compiled-data/BTCUSDT.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_lstm.save_weights("compiled-data/BTCUSDT.h5")
print("Saved model to disk")