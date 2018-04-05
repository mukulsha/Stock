import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Import data
data = pd.read_csv('daily/TCS.csv')
# Set column to forecast
forecast_col = 'Close'
# Set time to forecast
forecast = 30
print(data.tail())
data.dropna(inplace = True)
print(data.tail())
data['Label'] = data[forecast_col].shift(-forecast)
# Drop date variable
data = data.drop(['Date'], 1)
# Dimensions of dataset
n = data.shape[0]
p = data.shape[1]
# Make data a numpy array
data = data.values

# Training and test data
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end
test_end = n
print(data)

data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

# Scale data
scaler = MinMaxScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)
# Build X and y
X_train = data_train[:, 1:]
print(X_train)
y_train = data_train[:, 0]
print(y_train)
X_test = data_test[:, 1:]
print(X_test)
y_test = data_test[:, 0]
print(y_test)