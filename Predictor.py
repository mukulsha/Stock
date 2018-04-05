import pandas as pd
import math, datetime
from datetime import datetime, timedelta
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style


class Predictor:

    forecast_col = "Close"

    df = None
    interval = None
    forecast = None
    company_name = None
    max_stock_price = None
    percentage_gain = None
    investment = 1

    clf = None

    def __init__(self, company_name, interval = "daily", forecast = 30):

        style.use('ggplot')
        df = pd.read_csv(interval + '/' + company_name + '.csv')
        self.df = df
        self.interval = interval
        self.forecast = forecast
        self.company_code = company_name

        clf = LinearRegression()

    def get_last_date(self):
        return self.df.iloc[-1].name

    def get_last_open(self):
        return self.df.iloc[-1].Open

    def get_last_high(self):
        return self.df.iloc[-1].High

    def get_last_low(self):
        return self.df['Low'][-1]

    def get_last_last(self):
        return self.df.iloc[-1].Last

    def get_last_close(self):
        return self.df.iloc[-1].Close

    def get_last_total_trade_quantity(self):
        return self.df.iloc[-1]['Total Trade Quantity']

    def get_last_turnover(self):
        return self.df.iloc[-1]['Turnover (Lacs)']

    def get_count(self):
        return self.df.count()[1]

    def get_interval(self):
        return self.interval

    def set_interval(self, interval):
        self.interval = interval

    def get_forecast(self):
        return self.forecast

    def set_forecast(self, forecast):
        self.interval = forecast

    def get_percentage_gain(self):
        return self.percentage_gain

    def set_percentage_gain(self, percentage_gain):
        self.percentage_gain = percentage_gain

    def get_investment(self):
        return self.investment

    def set_investment(self, investment):
        self.investment = investment

    def get_returns(self, investment = 1):
        return investment * self.get_percentage_gain()

    def get_accuracy(self):
        df = pd.read_csv(self.interval + '/' + self.company_code + '.csv')
        df = pd.read_csv('TCS.csv')
        df['HL_PCT'] = (df['High'] - df['Close']) / df['Close']
        df['PCT_CHG'] = (df['Close'] - df['Open']) / df['Open']
        df['Volume'] = df['Total Trade Quantity']
        df = df[['Date', 'Close', 'HL_PCT', 'PCT_CHG', 'Volume']]
        df.fillna(-99999, inplace=True)
        df.set_index('Date', inplace=True)
        df['Label'] = df[self.forecast_col].shift(-self.forecast)
        X = np.array(df.drop(['Label'], 1))
        X = preprocessing.scale(X)

        x_lately = X[-self.forecast:]
        X = X[:-self.forecast]

        df.dropna(inplace=True)
        y = np.array(df['Label'])
        clf = LinearRegression()
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
        clf.fit(X_train, y_train)

        return clf.score(X_test, y_test)

    def get_daily_prediction(self):
        df = pd.read_csv(self.interval + '/' + self.company_code + '.csv')
        df['HL_PCT'] = (df['High'] - df['Close']) / df['Close']
        df['PCT_CHG'] = (df['Close'] - df['Open']) / df['Open']
        df['Volume'] = df['Total Trade Quantity']
        df = df[['Date', 'Close', 'HL_PCT', 'PCT_CHG', 'Volume']]
        df.fillna(-99999, inplace=True)
        df.set_index('Date', inplace=True)
        df['Label'] = df[self.forecast_col].shift(-self.forecast)
        X = np.array(df.drop(['Label'], 1))
        X = preprocessing.scale(X)

        x_lately = X[-self.forecast:]
        X = X[:-self.forecast]

        df.dropna(inplace=True)
        y = np.array(df['Label'])

        clf = LinearRegression()
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
        clf.fit(X_train, y_train)

        forecast_set = clf.predict(x_lately)
        df['Forecast'] = np.nan
        last_date = df.iloc[-1].name
        last_price = df['Close'][-1]
        last_unix = datetime.strptime(last_date, "%Y-%m-%d")
        next_unix = last_unix + timedelta(days=1)

        for i in forecast_set:
            next_date = (next_unix)
            next_unix += timedelta(days=1)
            df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

        df['Close'].plot()
        df['Forecast'].plot()
        temp = df['Forecast']
        temp.dropna(inplace = True)
        self.max_stock_price = max(temp)
        self.set_percentage_gain(self.max_stock_price / last_price)
        # plt.legend(loc=4)
        # plt.xlabel('Date')
        # plt.ylabel('Price')
        # plt.show()
# tcs = Predictor('TCS')
# print(tcs.get_accuracy())