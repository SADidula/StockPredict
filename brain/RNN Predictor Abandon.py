import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
# from tensorflow.python.keras.models import model_from_json

# load json and create model
json_file = open('compiled-data/BTCUSDT.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("compiled-data/BTCUSDT.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

#importing test data and reshape for usage
with open("../model/BTCUSDT.json") as datafile:
    data = json.load(datafile)
df = pd.DataFrame(data)

scaler = MinMaxScaler()

test_data = df.iloc[:, 1].values
unscaled_test_data = df.iloc[:, 1].values

x_test_data = np.reshape(test_data, (-1, 1))
x_test_data = scaler.fit_transform(x_test_data)

final_x_test_data = []
# test_data_len = int(len(test_data) / 2)
test_data_len = int(len(test_data) / 2)

for i in range(test_data_len, len(x_test_data)):
    final_x_test_data.append(x_test_data[i-test_data_len:i, 0])

final_x_test_data = np.array(final_x_test_data)
final_x_test_data = np.reshape(final_x_test_data, (final_x_test_data.shape[0], final_x_test_data.shape[1], 1))

predictions = loaded_model.predict(final_x_test_data)

unscaled_predictions = scaler.inverse_transform(predictions)
# plt.clf() #This clears the first prediction plot from our canvas
# plt.plot(unscaled_predictions)

plt.plot(unscaled_predictions, color = '#135485', label = "Predictions")
plt.plot(unscaled_test_data, color = 'black', label = "Real Data")
plt.title('Facebook Stock Price Predictions')
plt.show()


