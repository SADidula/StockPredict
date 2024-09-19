# Abandon due to not efficiently handling large sum of data
# forecasting efficiency is low

import json
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import mean_squared_error
from math import sqrt

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
df.set_index('Timestamp')

# f = plt.figure()
# ax1 = f.add_subplot(121)
# ax1.set_title('1st diff')
# ax1.plot(df['p'].astype(float))
#
# ax2 = f.add_subplot(122)
# plot_acf(df['p'].astype(float).dropna(), ax=ax2)
# plt.show()

def adfuller_test(stock_prices):
    result=adfuller(stock_prices)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )

    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
    else:
        print("weak evidence against null hypothesis,indicating it is non-stationary ")

adfuller_test(df['Close'].diff().dropna())

# df['p'].astype(float).diff().dropna().plot()
# autocorrelation_plot(df['p'].astype(float).diff().dropna())
#
# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(df['p'].astype(float).diff().dropna(),lags=40,ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(df['p'].astype(float).diff().dropna(),lags=40,ax=ax2)

# For non-seasonal data
#p=1, d=1, q=2

stock_y_axis = df['Close']
# stock_y_axis = stock_y_axis.replace([np.inf, -np.inf], np.nan).dropna()

stock_x_axis = pd.to_datetime(df['Timestamp'] * 1000000, unit='ns')
# stock_x_axis = stock_x_axis.replace([np.inf, -np.inf], np.nan).dropna()

# model=ARIMA(endog=stock_y_axis,exog=stock_x_axis,order=(1,1,2))
# model_fit=model.fit()
# model_fit.summary()

model=sm.tsa.statespace.SARIMAX(stock_y_axis,order=(1, 1, 2),seasonal_order=(1,1,2,4))
results=model.fit()

future_dates=[df.index[-1]+ x for x in range(0,len(stock_y_axis))]
future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)

future_datest_df.tail()

future_df=pd.concat([df,future_datest_df])

future_df['forecast'] = results.predict(start = 0, end = len(stock_y_axis))

true_data_len = len(stock_x_axis)
predicted_data_len = len(future_df['forecast'])
sum_data_len = true_data_len + (predicted_data_len - true_data_len)

for index in range(sum_data_len):
    print('predicted=%f, expected=%f' % (future_df['forecast'][index], future_df['Close'][index]))

rmse = sqrt(mean_squared_error(future_df['Close'], future_df['forecast']))
print('Test RMSE: %.3f' % rmse)

# future_df['p'].astype(float).plot(figsize=(12,8))
# future_df['forecast'].plot(figsize=(12, 8))

# plt.show()