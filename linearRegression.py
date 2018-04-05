import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('TCS.csv')
df['HL_PCT'] = (df['High'] - df['Close']) / df['Close']
df['PCT_CHG'] = (df['Close'] - df['Open']) / df['Open']
df['Volume'] = df['Total Trade Quantity']
df = df[['Date', 'Close', 'HL_PCT', 'PCT_CHG', 'Volume']]
df['Forecast'] = df.Close.shift(-30)
df.set_index('Date', inplace=True)
df.fillna(method='ffill', inplace=True)
train_test_dataframe = df[:-30]

prediction_dataframe = df[-30:]

train_X_dataframe = np.array(train_test_dataframe.iloc[:int(0.9*train_test_dataframe.count()[0]),:-1])
test_X_dataframe = np.array(train_test_dataframe.iloc[int(0.9*train_test_dataframe.count()[0]):,:-1])
train_y_dataframe = np.array(train_test_dataframe.iloc[:int(0.9*train_test_dataframe.count()[0]),-1])
test_y_dataframe = np.array(train_test_dataframe.iloc[int(0.9*train_test_dataframe.count()[0]):,-1])

lreg =  linear_model.LinearRegression()
lreg.fit(train_X_dataframe, train_y_dataframe)

print(lreg.score(test_X_dataframe,test_y_dataframe))