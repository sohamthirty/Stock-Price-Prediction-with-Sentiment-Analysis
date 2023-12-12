# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 02:36:59 2019

@author: Admin

To run the code:
cd app
streamlit run app.py
"""

import streamlit as st
import matplotlib.pyplot as plt, pandas as pd, numpy as np
from PIL import Image


from matplotlib.pyplot import rc
from pandas_datareader import data as pdr
from datetime import datetime
import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)
from dateutil.parser import parse
from scipy.stats import iqr
from datetime import timedelta, date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pickle
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from functions import *
from openai import OpenAI

def set_pub():
    rc('font', weight='bold')    # bold fonts are easier to see
    rc('grid', c='0.5', ls='-', lw=0.5)
    rc('figure', figsize = (10,8))
    plt.style.use('bmh')
    rc('lines', linewidth=1.3, color='b')

@st.cache_data()
def loadData(ticker, start, end): 
     df_stockdata = pdr.get_data_yahoo(ticker, start= str(start), end = str(end) )['Adj Close']   
     df_stockdata.index = pd.to_datetime(df_stockdata.index)
     return df_stockdata

def get_data_yahoo(ticker, start, end):
    data = pdr.get_data_yahoo(ticker, start= str(start), end = str(end))
    return st.dataframe(data)

        

def plotData(ticker, start, end):
    
    df_stockdata = loadData(ticker, start, end)
    df_stockdata.index = pd.to_datetime(df_stockdata.index)

    
    
    set_pub()
    fig, ax = plt.subplots(2,1)

    
    ma1_checkbox = st.checkbox('Moving Average 1')
    
    ma2_checkbox = st.checkbox('Moving Average 2')
    
    ax[0].set_title('Adj Close Price %s' % ticker, fontdict = {'fontsize' : 15})
    ax[0].plot(df_stockdata.index, df_stockdata.values,'g-',linewidth=1.6)
    ax[0].set_xlim(ax[0].get_xlim()[0] - 10, ax[0].get_xlim()[1] + 10)
    ax[0].set_xlabel('Time Frame')
    ax[0].set_ylabel('Price in USD')
    ax[0].grid(True)
    
    if ma1_checkbox:
        days1 = st.slider('Business Days to roll MA1', 5, 120, 30)
        ma1 = df_stockdata.rolling(days1).mean()
        ax[0].plot(ma1, 'b-', label = 'MA %s days'%days1)
        ax[0].legend(loc = 'best')
    if ma2_checkbox:
        days2 = st.slider('Business Days to roll MA2', 5, 120, 30)
        ma2 = df_stockdata.rolling(days2).mean()
        ax[0].plot(ma2, color = 'magenta', label = 'MA %s days'%days2)
        ax[0].legend(loc = 'best')
    
    ax[1].set_title('Daily Total Returns %s' % ticker, fontdict = {'fontsize' : 15})
    ax[1].plot(df_stockdata.index[1:], df_stockdata.pct_change().values[1:],'r-')
    ax[1].set_xlim(ax[1].get_xlim()[0] - 10, ax[1].get_xlim()[1] + 10)
    ax[1].set_xlabel('Time Frame')
    ax[1].set_ylabel('Percentage Change')
    plt.tight_layout()
    ax[1].grid(True)
    st.pyplot(fig)
    


def prediction_app(start, end):
            START = pd.to_datetime(start)
            TODAY = date.today().strftime("%Y-%m-%d")

            stocks = ('GOOG', 'AMZN', 'MSFT', 'GME', 'META', 'NFLX','NVDA','TSLA')
            selected_stock = st.selectbox('Select dataset for prediction', stocks)

            @st.cache_data
            def load_data(ticker):
                data = yf.download(ticker, START, TODAY)
                data.reset_index(inplace=True)
                return data
                
            #data_load_state = st.text('Loading data...')
            data = load_data(selected_stock)
            #data_load_state.text('Loading data... done!')


            prophet_model = st.checkbox('FB Prophet')
            if prophet_model:
                n_years = st.slider('Years of prediction:', 1, 4)
                period = n_years * 365

            
                # Predict forecast with Prophet.
                df_train = data[['Date','Close']]
                df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

                m = Prophet()
                m.fit(df_train)
                future = m.make_future_dataframe(periods=period)
                forecast = m.predict(future)

                # Show and plot forecast
                st.subheader('Forecast data')
                st.write(forecast.tail())
                    
                st.write(f'Forecast plot for {n_years} years')
                fig1 = plot_plotly(m, forecast)
                st.plotly_chart(fig1)

            lstm_model = st.checkbox('Multivariate LSTM')
            if lstm_model:
                data, START, TODAY = load_data_lstm(selected_stock)
                
                features = ['Close', 'High', 'Low', 'Open', 'Volume']
                look_back = 5 # No. of Lags to consider
                predict_type = 'predict' #predict_type.lower()

                device = torch.device('cpu')

                train_X, train_Y, train_dates, test_X, test_Y, test_dates, scaler, test_data = prepare_data_multivariate(data, selected_stock, START, TODAY, features=features, look_back=look_back, predict_type=predict_type )

                path = r"Saved Params copy/{}_params.pkl".format(selected_stock)
                stock_model = r"Saved Models copy/{}_model".format(selected_stock) 

                file = open(path,'rb')
                object_file = pickle.load(file)
                print(device)

                model = DynamicLSTM(object_file['input_size'], object_file['hidden_size'], object_file['num_layers'], object_file['output_size']).to(device)
                model.load_state_dict(torch.load(stock_model,map_location=torch.device('cpu')))
                model.eval()



                # table, df = get_preds(test_X, test_data, test_dates, scaler, model)
                table, df = get_preds(object_file['test_X'], object_file['test_data'], object_file["test_dates"], object_file['test_predict_inverse'],object_file['test_Y_inverse'], model)
                st.subheader('Predicted values')
                st.write(table)

                dates = df['Date']

                fig = plt.figure(figsize=(15,12))
                plt.plot(dates.iloc[:-1], df['Actual Price'].iloc[:-1], label='True', linewidth=2)
                plt.plot(dates.iloc[:-1], df['Predicted Price'].iloc[:-1], label='Predicted', linewidth=2)
                plt.plot(dates.iloc[-1],df['Predicted Price'].iloc[-1], label='Forecast',marker = '*', color= 'orange', markersize=20)

                #for i in range(0,len(dates)): 
                plt.annotate('TBA', (dates[len(dates)], df['Actual Price'][len(dates)] + 0.5 ))
                plt.annotate(df['Predicted Price'][len(dates)], (dates[len(dates)], df['Predicted Price'][len(dates)] + 0.2 ))


                plt.rcParams.update({'font.size': 15})
                plt.title("Test vs. Predicted Prices")
                plt.xlabel("Date")
                plt.ylabel("Price")
                plt.legend()
                plt.xticks(rotation=45)
                plt.grid(True)
                st.pyplot(fig)


def chatgpt_bot():
    #st.title("ChatGPT-like clone")
    client = OpenAI(api_key = st.secrets["OPENAI_API_KEY"]) # Add your API key

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Any question about the stock market?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            ):
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

	
def show_news(ticker):
    stock = yf.Ticker(ticker)
    all_news = stock.news
    titles = []
    for news in all_news:
        titles.append(news['title'])
    title_df = pd.DataFrame(titles, columns=['Top Headlines'])
    st.dataframe(title_df, width=800)




    

''' # Stock Market Analytics
   #### (stock prices from *yahoo finance*) '''

sp500_list = pd.read_csv('SP500_list.csv')

ticker = st.selectbox('Select any stock ticker from S&P 500', sp500_list['Symbol'], index = 30).upper()
pivot_sector = True

image = Image.open('imageforapp.png')



st.sidebar.image(image, caption='', 
                 use_column_width=True)
st.sidebar.header('A stock analysis app')
st.sidebar.subheader('Choose the option to visualize')

ticker_meta = yf.Ticker(ticker)
        
series_info  = pd.Series(ticker_meta.info,index = reversed(list(ticker_meta.info.keys())))
series_info = series_info.loc[['symbol', 'shortName', 'financialCurrency','exchange',
                                'marketCap', 'quoteType', 'industry', 'longBusinessSummary', 
                                'fullTimeEmployees']]

# Rename the columns
new_col_names = {
    'symbol': 'Ticker Symbol',
    'shortName': 'Stock Name',
    'financialCurrency': 'Financial Currency',
    'exchange': 'Exchange',
    'marketCap': 'Market Cap',
    'quoteType': 'Quote Type',
    'industry': 'Industry',
    'longBusinessSummary':'About',
    'fullTimeEmployees':'Company Size'
}
series_info = series_info.rename(new_col_names)


if pivot_sector:
    sector = sp500_list[sp500_list['Symbol'] == ticker]['Sector']
    sector = sector.values[0]
    series_info['Sector'] = sector

'''#### About:'''
series_info['About']

series_info = series_info.drop('About')

series_info.name = 'Stock'            
st.dataframe(series_info, width=800)

if pivot_sector:
    todays_news = st.sidebar.checkbox("Today's News", value = True)
    if todays_news:
        st.title("Today's Stock News")
        show_news(ticker)

'''#### Choose the timeframe of the stock'''
start = st.text_input('Enter the start date in yyyy-mm-dd format:', '2018-01-01')
end = st.text_input('Enter the end date in yyyy-mm-dd format:', '2019-01-01')

try:
    start = parse(start).date()
    #print('The start date is valid')
    control_date1 = True
except ValueError:
    st.error('Invalid Start date')
    control_date1 = False
 
    
try:
    end = parse(end).date()
    #print('The end date is valid')
    control_date2 = True
except ValueError:
    st.error('Invalid End date')
    control_date2 = False

def check_dates():
    return control_date1 & control_date2


if start <= datetime(1970,1,1,0,0).date():
    st.error('Please insert a date posterior to 1st January 1970')
    pivot_date = False
else:
    pivot_date = True
    
if check_dates() and pivot_date == True:
    
        
    if len(loadData(ticker, start, end)) > 0: # if the ticker is invalid the function returns an empty series
        

        principal_graphs_checkbox = st.sidebar.checkbox('Stock Trends')
        if principal_graphs_checkbox:
            '''# Stock Trends'''
            plotData(ticker, start, end)
        
                    
        historical_prices_checkbox = st.sidebar.checkbox('Historical prices and volumes')
        if historical_prices_checkbox:
            st.title('Historical prices and volumes')
            get_data_yahoo(ticker, start, end)
        
        see_future = st.sidebar.checkbox('Future Close Price Prediction')
        if see_future:
            st.title('Future Close Price Prediction')
            prediction_app(start, end)
        
        chatgpt = st.sidebar.checkbox('ChatBot help')
        if chatgpt:
            st.title('Ask ChatGPT about the stock market!')
            chatgpt_bot()

        
        
    else:
        st.error('Invalid ticker')
    

    




