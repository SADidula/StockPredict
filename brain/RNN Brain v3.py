# import libraries
from sklearn.preprocessing import MinMaxScaler
import IPython.display
import json
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from keras.preprocessing import timeseries_dataset_from_array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from tensorflow import keras

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

# epoch to date time
df['Timestamp'] = pd.to_datetime(df['Timestamp'] * 1000000, unit='ns')

# sorting by date
df.sort_values('Timestamp', inplace=True)
df = df[df['Timestamp'] >= '2017']
df = df.dropna(axis=0, thresh=7)
date_time = pd.to_datetime(df.pop('Timestamp'), format='%d.%m.%Y %H:%M:%S')
timestamp_s = date_time.map(datetime.datetime.timestamp)

day = 24 * 60 * 60
year = (365.2425) * day
df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

# df.info()

# divide data to (70%, 20%, 10%)
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n * 0.7)]
val_df = df[int(n * 0.7):int(n * 0.9)]
test_df = df[int(n * 0.9):]

num_features = df.shape[1]

# normalize data
train_mean = train_df.mean()
train_std = train_df.std()
train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# data windowing
class WindowGenerator:
    pass

class WindowGenerator:
    def __init__(self, input_width, label_width, shift,
                 train_df=train_df, val_df=val_df, test_df=test_df,
                 label_columns=None):

        # initialize data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # work out the label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # work out the window parameters
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    # split window function
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # setting dataset shape
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    WindowGenerator.split_window = split_window

    # convert dataframe to dataset
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32, )

        ds = ds.map(self.split_window)

        return ds

    # convert all dataframes to datasets
    WindowGenerator.make_dataset = make_dataset

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result

    def plot(self, model=None, plot_col='Close', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12,8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)

            if model is not None:
                predictions = model(inputs)

                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

                # print(predictions)


            if n == 0:
                plt.legend()

        plt.xlabel('Time [hours]')
        plt.show()

lstm_model = Sequential()
lstm_model.add(LSTM(32, return_sequences=True))  # 64 lstm neuron block
lstm_model.add(Dropout(0.2))

for i in [True]:
    lstm_model.add(LSTM(units=32, return_sequences=i))

lstm_model.add(Dense(15, activation='linear'))

OPTIMIZER = keras.optimizers.Adam(learning_rate=0.01)
MAX_EPOCHS = 100
BATCH_SIZE = 5

def compile_and_fit(model, window, patience=3):
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   mode='min')
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=OPTIMIZER, metrics = ["mae"])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=window.val,
                        callbacks=[early_stopping],
                        shuffle=False)

    return history

# main work
# creating window
wide_window = WindowGenerator(
    input_width=32, label_width=32, shift=1,
    label_columns=['Close']
)

# wide_window
history = compile_and_fit(lstm_model, wide_window)
IPython.display.clear_output()
lstm_model.save('compiled-data/BTCUSDT.h5')

wide_window.plot(lstm_model.predict)
