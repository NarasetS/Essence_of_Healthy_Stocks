import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
import datetime
import calendar


#### Calculate ROIC #####
def yfinance_average_ROI(ticker,property) :
    try:
        netfixedcapital = yf.Ticker(ticker).balance_sheet.dropna(axis = 1, how = 'all')
        netfixedcapital = netfixedcapital.T
        netfixedcapital = netfixedcapital['Working Capital'] + netfixedcapital['Total Non Current Assets']

        earninngs = yf.Ticker(ticker).financials.dropna(axis = 1, how = 'all')
        earninngs = earninngs.T
        earninngs = earninngs[property]


        data = pd.DataFrame()
        data['Invested Capital'] = netfixedcapital
        data[property] = earninngs
        data = data.dropna()
        data['avg_MF_ROC'] = data[property]/data['Invested Capital']
        avg_MF_ROC = data['avg_MF_ROC'].mean()
        avg_MF_ROC = np.round(avg_MF_ROC,4)

        return avg_MF_ROC
    except:
        return None
    
    
def yfinance_normalised_OI_index(ticker,property):
    ### Get data from .financials and .quaterly_financials ####
    try:
        data_financials = yf.Ticker(ticker).financials.loc[yf.Ticker(ticker).financials.index == property].dropna(axis = 1, how = 'all')
        data_financials = data_financials.T
        # print(data_financials)
        
        data_quaterly_financials = yf.Ticker(ticker).quarterly_financials.loc[yf.Ticker(ticker).quarterly_financials.index == property].dropna(axis = 1, how = 'all')
        list_col = data_quaterly_financials.columns
        data_quaterly_financials = data_quaterly_financials.reset_index()
        data_quaterly_financials['ttm'] = 0
        if len(list_col) >=4 :
            ttm_period = 4
        else : ttm_period = len(list_col)
        for i in range(ttm_period): 
            data_quaterly_financials['ttm'] = data_quaterly_financials['ttm'] + data_quaterly_financials[list_col[i]]
        data_quaterly_financials = data_quaterly_financials.set_index('index')
        data_quaterly_financials = data_quaterly_financials.T
        # print(data_quaterly_financials.loc[data_quaterly_financials.index == 'ttm'])
        # print(data_quaterly_financials)
              
        data = pd.concat([data_quaterly_financials.loc[data_quaterly_financials.index == 'ttm'],data_financials])

        minval = data[property].astype('float64').min()
        maxval = data[property].astype('float64').max()
        numofyear = len(data[property])
        data[property] = ( data[property] - minval ) / (maxval - minval)
        
        
        #### Calculate beta ###
        data.reset_index(inplace=True)
        data = data.rename(columns = {'index':'time'})
        data.reset_index(inplace=True)
        data = data.sort_values('index',ascending=False)
        data = data.drop(columns = 'index')
        data.reset_index(inplace=True,drop = True)
        beta, alpha = np.polyfit(np.array(data[property]).astype('float64'), np.array(data.index).astype('float64'),1)
        # data[property].plot()
        beta = np.round(beta,4)
        return beta , numofyear
    except: 
        return
    
def normalize(df):
  x = df.copy()
  for i in x.columns[0:]:
    x[i] = x[i]/x[i][0]
  return x

def daily_return(df):
    df_daily_return = df.copy()
    for i in df.columns[0:]:
        for j in range(1, len(df)):
            # df_daily_return.iloc[j][i] = ((df.iloc[j][i] - df.iloc[j-1][i])/df.iloc[j-1][i]) * 100
            df_daily_return.loc[j,i] = ((df.loc[j,i] - df.loc[j-1,i])/df.loc[j-1,i]) * 100
        # df_daily_return.iloc[0][i] = 0
        df_daily_return.loc[0,i] = 0
    return df_daily_return

def finding_beta(market,ticker,date_start,date_end):
    try:
        m = yf.Ticker(market).history(start=date_start, end=date_end, interval="1d")
        s = yf.Ticker(ticker).history(start=date_start, end=date_end, interval="1d")
        price_current = s.iloc[-1]['Close']
        price_past = s.iloc[0]['Close']
        # print(price_past)
        # print(price_current)
        # print(s)

        m.reset_index(inplace=True)
        s.reset_index(inplace=True)
        m['1d'] = m['Date'].dt.date
        s['1d'] = s['Date'].dt.date
        m = m.drop(columns=['Date'])
        s = s.drop(columns=['Date'])
        m = m.set_index('1d')
        s = s.set_index('1d')

        data = pd.DataFrame()
        data['market'] = m['Close']
        data['ticker'] = s['Close']
        
        data = data.loc[~(data['market'].isna() | data['ticker'].isna())]
        data.reset_index(inplace=True,drop=True)

        data = normalize(data)
        # data.plot()
        data_daily_return = daily_return(data)
        # data_daily_return.plot.scatter(x='market',
        #               y='ticker',
        #               c='DarkBlue',)
        beta, alpha = np.polyfit(data_daily_return['market'], data_daily_return['ticker'], 1)
        return beta, price_current, price_past
    except:
        return None
    
def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year,month)[1])
    return datetime.date(year, month, day)
