import math
import urllib.request, json
import numpy as np
import pandas as pd
import warnings

from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from tensorflow import keras
from keras.callbacks import EarlyStopping

# importing training data and reshape for usage
with open("../model/XLMUSDT.json") as datafile:
    data = json.load(datafile)
df = pd.DataFrame(data)

# convert cloumns to respective data types
df['Timestamp'] = pd.to_datetime(df['Timestamp'] * 1000000, unit='ns')
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

#Create a new data frame with only the 'Close column'
data = df.filter(['Close'])

#Convert the dataframe to numpy array
dataset = data.values

#Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8 )

#Scale the data before it presents to neural netrwork as it good practice by preprocessing the datae
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#Create the training data set
#Crete the scaled training set
train_data = scaled_data[0:training_data_len , :]

#Split the data into x_train and _train data sets
x_train = []
y_train = []

for i in range(120, len(train_data)):
  x_train.append(train_data[i-120:i, 0])
  y_train.append(train_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

model_lstm = Sequential()
model_lstm.add(LSTM(32,return_sequences=True,input_shape= (x_train.shape[1], 1))) #64 lstm neuron block
model_lstm.add(Dropout(0.2))

for i in [False]:
    model_lstm.add(LSTM(units = 32, return_sequences = i))
    # model_lstm.add(Dropout(0.2))

model_lstm.add(Dense(15, activation='linear'))

optimizer = keras.optimizers.Adam(learning_rate=0.01)

early_stopping = EarlyStopping(monitor='loss',
                                   patience=3,
                                   mode='min')

model_lstm.compile(loss = "mean_squared_error", optimizer = optimizer, metrics = ["mae"])
model_lstm.fit(x_train, y_train, epochs = 100, batch_size = 5, callbacks=[early_stopping], shuffle=False)

#Create the testing data set
#Create a new array containing scaled values from index 1616 to 2170
test_data = scaled_data[training_data_len - 120: , :]

#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(120, len(test_data)):
  x_test.append(test_data[i-120:i, 0])

x_test = np.array(x_test)

#Get the model predicted price values
predictions = model_lstm.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Get the root mean squared error (RSME)
rsme = np.sqrt( np.mean( predictions - y_test )**2 )

#plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()