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
import openai


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





def load_data(choosen_stock):    
    yf.pdr_override() # Override pandas datareader with yfinance
    y_symbols = [choosen_stock]
    
    # State the dates
    startdate = dt.datetime(2018, 1, 1) # start date
    enddate = dt.datetime(2023, 11, 30) # end date # +1
    
    # Retrieve historical stock price data for the specified symbols and date range
    df = yf.download(y_symbols, start=startdate, end=enddate) 
    df = df.reset_index() # Reset the index to make 'Date' a regular column
    df['Stock'] = choosen_stock # add 'Stock' column
    df = df[['Date', 'Stock', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']] # Reorder the columns
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df, startdate, enddate


def prepare_data_multivariate(df, choosen_stock, startdate, enddate, features, look_back, predict_type='year'):
    # Choose specific stock
    data = df #df[df["Stock"] == choosen_stock]

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
    train_X = torch.Tensor(train_X.astype(np.float32))
    train_Y = torch.Tensor(train_Y)
    test_X = torch.Tensor(test_X.astype(np.float32))
    test_Y = torch.Tensor(test_Y)
    
    return train_X, train_Y, train_dates, test_X, test_Y, test_dates, scaler, test_data


def get_preds(test_X, test_data, test_dates, scaler, model):
    
    test_predict = model(test_X).view(-1).cpu().detach().numpy()
    # Inverse Scaling
    # --> 1.test_predict
    test_data1 = test_data[:, 1:-1]
    # Ensure the second array has the same number of rows as the first array
    test_data1 = test_data1[:test_predict.reshape(-1, 1).shape[0], :]
    # Append the arrays
    test_data1 = np.hstack((test_predict.reshape(-1, 1), test_data1)) 
    test_predict_inverse = scaler.inverse_transform(test_data1)[:,0]

    # --> 2.test_Y
    test_data2 = test_data[:, :-1]
    test_data2 = test_data2[:test_predict.reshape(-1, 1).shape[0], :]
    test_Y_inverse = scaler.inverse_transform(test_data2)[:,0]


    # Formatting the prices to a desired decimal form
    actual_prices = ["{:.4f}".format(price) for price in test_Y_inverse.flatten()]
    predicted_prices = ["{:.4f}".format(price) for price in test_predict_inverse.flatten()]
    formatted_dates = [test_dates.strftime('%Y-%m-%d') for test_dates in test_dates]


    # Display
    # Remove the first value and shift up the remaining values
    actual_prices = actual_prices[1:] + ['']
    table = PrettyTable()
    table.field_names = ["Date", "Actual Price", "Predicted Price"]
    for date, actual_price, predicted_price in zip(formatted_dates, actual_prices, predicted_prices):
        # Add a separator line before the last row
        if date == formatted_dates[-1]:
            table.add_row(["-" * 10, "-" * 12, "-" * 16])
        table.add_row([date, round(float(actual_price), 2) if actual_price != '' else 'TBA', round(float(predicted_price), 2)])

    return table


def chatgpt():
    openai.api_key = "***"

    messages = [
    {"role": "system", "content": "You are a kind helpful assistant."},]

    while True:
        message = input("User : ")
        if message:
            messages.append(
                {"role": "user", "content": message},
            )
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )
        
        reply = chat.choices[0].message.content
        print(f"ChatGPT: {reply}")
        messages.append({"role": "assistant", "content": reply})

    return
