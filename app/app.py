# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 02:36:59 2019

@author: Admin
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
    data = pdr.get_data_yahoo(ticker, start= str(start), end = str(end) )
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
    plt.tight_layout()
    ax[1].grid(True)
    st.pyplot(fig)
    


def prediction_app():
            START = start
            TODAY = date.today().strftime("%Y-%m-%d")

            stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
            selected_stock = st.selectbox('Select dataset for prediction', stocks)

            prophet_model = st.checkbox('FB Prophet')
            if prophet_model:
                n_years = st.slider('Years of prediction:', 1, 4)
                period = n_years * 365

                @st.cache_data
                def load_data(ticker):
                    data = yf.download(ticker, START, TODAY)
                    data.reset_index(inplace=True)
                    return data
                
                data_load_state = st.text('Loading data...')
                data = load_data(selected_stock)
                data_load_state.text('Loading data... done!')

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
                pass

    

''' # Stock Market Analytics
   #### (stock prices from *yahoo finance*) '''

sp500_list = pd.read_csv('SP500_list.csv')

ticker = st.selectbox('Select any stock ticker from S&P 500', sp500_list['Symbol'], index = 30).upper()
pivot_sector = True
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
        
     
        image = Image.open('imageforapp.png')



        st.sidebar.image(image, caption='',

                 use_column_width=True)
        st.sidebar.header('A stock analysis app')
        st.sidebar.subheader('Choose the option to visualize')
        
        ticker_meta = yf.Ticker(ticker)
        
        series_info  = pd.Series(ticker_meta.info,index = reversed(list(ticker_meta.info.keys())))
        series_info = series_info.loc[['symbol', 'shortName', 'financialCurrency','exchange',
                                        'marketCap', 'quoteType']]
        if pivot_sector:
            sector = sp500_list[sp500_list['Symbol'] == ticker]['Sector']
            sector = sector.values[0]
            series_info['sector'] = sector
        
           
        series_info.name = 'Stock'            
        st.dataframe(series_info)

        
        principal_graphs_checkbox = st.sidebar.checkbox('Moving Average', value = True)
        if principal_graphs_checkbox:
            plotData(ticker, start, end)
        
                    
        historical_prices_checkbox = st.sidebar.checkbox('Historical prices and volumes')
        if historical_prices_checkbox:
            st.title('Historical prices and volumes')
            get_data_yahoo(ticker, start, end)
        
        see_future = st.sidebar.checkbox('Future Close Price Prediction')
        if see_future:
            st.title('Future Close Price Prediction')
            prediction_app()

        
        
    else:
        st.error('Invalid ticker')
    

    




