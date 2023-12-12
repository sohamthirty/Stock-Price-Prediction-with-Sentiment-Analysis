import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from pprint import pprint
import yfinance as yf
import datetime as dt
from itertools import combinations
import pickle
from prettytable import PrettyTable


class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_layers, output_size):
        super(DynamicLSTM, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.LSTM(input_size, hidden_sizes[0], num_layers=1, batch_first=True))
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.append(nn.LSTM(hidden_sizes[i-1], hidden_sizes[i], num_layers=1, batch_first=True))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x, _ = layer(x)
        x = self.output_layer(x[:, -1, :])  # Take the output from the last time step
        return x





def load_data_lstm(choosen_stock):     
    yf.pdr_override() # Override pandas datareader with yfinance
    y_symbols = [choosen_stock]
    
    # State the dates
    startdate = dt.datetime(2018, 1, 1) # start date
    enddate = dt.datetime(2023, 12, 12) # end date

    # Retrieve historical stock price data for the specified symbols and date range
    df = yf.download(y_symbols, start=startdate, end=enddate) 
    df = df.reset_index() # Reset the index to make 'Date' a regular column
    df['Stock'] = choosen_stock # add 'Stock' column
    df = df[['Date', 'Stock', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']] # Reorder the columns
    df['Date'] = pd.to_datetime(df['Date'])
        
    # add new row
    new_row = [{'Date':pd.to_datetime('2023-12-12T00:00:00.000000000'), 'Stock': choosen_stock , 'Adj Close': 0.0, 'Close': 0.0 , 'High': 0.0,'Low': 0.0, 'Open':0.0, 'Volume':0.0}]
    df = pd.concat([df, pd.DataFrame(new_row)], ignore_index=True)
    new_row = [{'Date':pd.to_datetime('2023-12-13T00:00:00.000000000'), 'Stock': choosen_stock , 'Adj Close': 0.0, 'Close': 0.0 , 'High': 0.0,'Low': 0.0, 'Open':0.0, 'Volume':0.0}]
    df = pd.concat([df, pd.DataFrame(new_row)], ignore_index=True)
    new_row = [{'Date':pd.to_datetime('2023-12-14T00:00:00.000000000'), 'Stock': choosen_stock , 'Adj Close': 0.0, 'Close': 0.0 , 'High': 0.0,'Low': 0.0, 'Open':0.0, 'Volume':0.0}]
    df = pd.concat([df, pd.DataFrame(new_row)], ignore_index=True)
    new_row = [{'Date':pd.to_datetime('2023-12-15T00:00:00.000000000'), 'Stock': choosen_stock , 'Adj Close': 0.0, 'Close': 0.0 , 'High': 0.0,'Low': 0.0, 'Open':0.0, 'Volume':0.0}]
    df = pd.concat([df, pd.DataFrame(new_row)], ignore_index=True)
    new_row = [{'Date':pd.to_datetime('2023-12-16T00:00:00.000000000'), 'Stock': choosen_stock , 'Adj Close': 0.0, 'Close': 0.0 , 'High': 0.0,'Low': 0.0, 'Open':0.0, 'Volume':0.0}]
    df = pd.concat([df, pd.DataFrame(new_row)], ignore_index=True)
    
    return df, startdate, dt.datetime(2023, 12, 18)


def prepare_data_multivariate(df, choosen_stock, startdate, enddate, features, look_back, predict_type='year'):
    # Choose specific stock
    data = df #df[df["Stock"] == choosen_stock]

    device = torch.device('cpu')

    # Test split
    if predict_type=='year':
        test_data = data[data["Date"].dt.year == 2019]
    elif predict_type=='month':
        test_data = data[(data["Date"].dt.year == 2019) & (data["Date"].dt.month.isin([1]))]
    elif predict_type=='days':
        test_data = data[data["Date"].dt.year == 2019][0:20] 
    else: # Specific
        test_data = data[(data["Date"] >= dt.datetime(2023, 11, 12)) & (data["Date"] <= enddate)]
        
    # Train split
    train_data = data[(data["Date"] >= startdate) & (data["Date"] <= dt.datetime(2023, 11, 11))]
    
    # Feature selection and engineering
    train_data = train_data[features + ["Date"]].values
    test_data = test_data[features + ["Date"]].values
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data[:, :-1] = scaler.fit_transform(train_data[:, :-1])
    test_data[:, :-1] = scaler.transform(test_data[:, :-1])
    
    # Create sequences for LSTM input
    def create_sequences(dataset, look_back=1):
        X, Y, dates = [], [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:(i + look_back), :-1])
            Y.append(dataset[i + look_back, 0])
            dates.append(dataset[i + look_back, -1])  # Assuming the last column is 'Date'
        return np.array(X), np.array(Y), np.array(dates)
    train_X, train_Y, train_dates = create_sequences(train_data, look_back)
    test_X, test_Y, test_dates = create_sequences(test_data, look_back)

    # Convert data to PyTorch tensors
    train_X = torch.Tensor(train_X.astype(np.float32)).to(device)
    train_Y = torch.Tensor(train_Y).to(device)
    test_X = torch.Tensor(test_X.astype(np.float32)).to(device)
    test_Y = torch.Tensor(test_Y).to(device)
    
    return train_X, train_Y, train_dates, test_X, test_Y, test_dates, scaler, test_data




# def get_preds(test_X, test_data, test_dates, scaler, model):
def get_preds(test_X, test_data, test_dates, test_predict_inverse, test_Y_inverse,model):
    test_predict = model(test_X).view(-1).cpu().detach().numpy()
    formatted_dates = [test_date.strftime('%Y-%m-%d') for test_date in test_dates]
    formatted_test_Y = ["{:.4f}".format(price) for price in test_Y_inverse.flatten()]
    formatted_test_predict = ["{:.4f}".format(price) for price in test_predict_inverse.flatten()]
    
    formatted_dates = formatted_dates[-14:-4]
    formatted_test_predict = formatted_test_predict[-14:-4]

    actual_prices = formatted_test_Y[0:] + ['']
    actual_prices = actual_prices[-10:]
    table = PrettyTable()
    table.field_names = ["Date", "Actual Price", "Predicted Price"]
    for date, actual_price, predicted_price in zip(formatted_dates, actual_prices, formatted_test_predict):
        # Add a separator line before the last row
        if date == formatted_dates[-1]:
            table.add_row(["-" * 10, "-" * 12, "-" * 16])
        table.add_row([date, round(float(actual_price), 2) if actual_price != '' else 'TBA', round(float(predicted_price), 2)])
    
    df = pd.DataFrame([row for row in table.rows], columns=table.field_names)
    df = df[df["Date"] != "----------"]
    df["Actual Price"] = df["Actual Price"].replace('TBA', 0)


    return table, df