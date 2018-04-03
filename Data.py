import pandas as pd
import datetime as dt
from datetime import datetime, timedelta
import csv
import os
import quandl
from Predictor import *


class Data:
    company_list = ""

    def __init__(self):
        self.company_list = pd.read_csv("company_names.csv")
        quandl.ApiConfig.api_key = "AL5rA7kD-SXUJiSFx2eM"

    #       MAKE DIRECTORY IF DOESNT EXIST      #

    @staticmethod
    def make_path(interval):
        if not os.path.exists(interval):
            os.mkdir(interval)

    #       GET COMPANY NAME        #

    @staticmethod
    def get_company_name(company_code):
        company_code = "NSE/" + company_code
        with open('NSE_company_names.csv', 'r') as company_list:
            for symbol in csv.reader(company_list):
                if symbol[0] == company_code:
                    return symbol[1]

    #       GET COMPANY CODE        #

    @staticmethod
    def get_company_code(company_name):
        with open('NSE_company_names.csv', 'r') as company_list:
            for symbol in csv.reader(company_list):
                if symbol[1] == company_name:
                    return symbol[0]

    @staticmethod
    def get_company_ticker(company_code):
        return "NSE/" + company_code

    @staticmethod
    def get_company_data(company, interval = "daily"):

        ticker = Data.get_company_ticker(company)
        Data.make_path(interval)

        # replace if else block for any downloaded csv #
        # df = pd.read_csv(interval + '/' + str(company) + '.csv') #
        if not os.path.exists(interval + '/' +company + '.csv'):
            df = quandl.get(ticker, collapse=interval)
            df.to_csv(interval + '/' + str(company) + '.csv')
        else:
            df = pd.read_csv(interval + '/' + str(company) + '.csv')
            last_date = df.iloc[-1]["Date"]
            last_date = datetime.strptime(last_date, "%Y-%m-%d")
            try:
                if last_date.date() != dt.datetime.today():
                    df = quandl.get(ticker, collapse=interval)
                    df.to_csv(interval + '/' + str(company) + '.csv')
            except ConnectionError:
                df = pd.read_csv(interval + '/' + str(company) + '.csv')
        return df

    @staticmethod
    def get_offline_company_data(company, interval ="daily"):
        return pd.read_csv(interval + '/' + str(company) + '.csv')
# Data.get_company_data("TCS")


Data.get_offline_company_data("WIPRO")
wipro =  Predictor("WIPRO").get_daily_prediction()
print("percentage gain")
print(wipro.get_percentage_gain())
print("investment")
print(wipro.get_investment())
wipro.set_investment(1000)
print("investment")
print(wipro.get_investment())
print("returns")
# print(wipro.get_returns())
# print(wipro.get_returns(100))

# wipro.get_daily_prediction()

# hello World