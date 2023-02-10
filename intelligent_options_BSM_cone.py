import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import time
import datetime
from datetime import datetime
from datetime import timedelta
from scipy.stats import lognorm
#force the layout to be wide
st.set_page_config(layout="wide")
st.title("Stock Price Compare to BSM Price Cone")
#create 2 columns
cols = st.columns(2)
#returns the x locaiton where y >=A 
def first_x(x, y, A):
    first_index = np.argwhere(y >= A)
    if first_index.size == 0:
        return np.nan
    else:
        return x[first_index[0][0]]
#Grabs the last years worth of ticker data
def get_ticker_closing_price(ticker):
    today = datetime.today()
    year_ago = today - timedelta(days=365)
    data = yf.download(ticker, start=year_ago, end=today)
    closing_prices = data['Close']
    closing_prices.rename(ticker, inplace=True)
    return closing_prices

Tick = cols[0].text_input("Enter Ticker Symbol: ")
risk_free_rate = cols[0].number_input("Enter Risk Free Rate (%): ", value=3.5)
minimum_value = cols[0].number_input("Enter Minimum Value: ", value=0)
median_value = cols[0].number_input("Enter Median Value: ", value=0)
maximum_value = cols[0].number_input("Enter Maximum Value: ", value=0)
trading_days=365

if st.button("Run"):
	stock_price_hist=get_ticker_closing_price(Tick)
	#grab the most recent price
	Initial_price = stock_price_hist.iloc[-1]

	ticker=yf.Ticker(Tick)
	#get the options chain for 45 days out
	option_idx=np.abs((pd.to_datetime(ticker.options))-(datetime.now()+timedelta(45))).argmin()
	option_date=ticker.options[option_idx]
	opt = ticker.option_chain(option_date).puts
	#grab the option strike price closest to current share price
	temp=opt[opt['strike']<Initial_price]
	IV=temp.iloc[-1]["impliedVolatility"]

	#create an array for 1Y with an element for each day
	days=np.arange(0,trading_days+1,1)
	price_min=[]
	price_max=[]
	#set the BSM cones to start at todays price
	price_min.append(Initial_price)
	price_max.append(Initial_price)
	price_max[0]=Initial_price
	#calculate the implied volatility by dividing by the sqrt of time
	iv_array=IV/np.sqrt(trading_days/days)
	#calculate the price step using a linear approximation with the current price grown at the risk free rate
	price_step=Initial_price*(risk_free_rate/100)/trading_days
	median_price=days*price_step+Initial_price
	median_price[0]=Initial_price
	#Calculate the upper and lower 1 sigma bound of price for each day
	for i in range(1,trading_days+1,1):
	    #samples the distribution in 1 cent increments up to the 3x the current price
	    x=np.arange(0,3*Initial_price,0.01)
	    #calculate the lognormal distribution given the current median price and volatility
	    prob=lognorm.cdf(x,iv_array[i], 0,median_price[i])
	    #grab the price at the 1 sigma lower bound
	    low=first_x(x,prob,0.16)
	    price_min.append(low)
	    #grab the price at the 1 sigma upper bound
	    high=first_x(x,prob,0.84)
	    price_max.append(high)
	
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	temp_label=Tick+ " Price"
	#plot the historical 1Y stock price data
	plt.plot(stock_price_hist,label=temp_label)
	#convert the days to dates in the future to make the plot easier to visualize
	start_date=datetime.now()
	dates = np.array([start_date + timedelta(days=int(day)) for day in days])
	#plot the BSM cone
	plt.plot(dates,median_price, color="black",label="Median BSM",linestyle='dashed')
	plt.plot(dates, price_min,color="black",label="Min BSM",linestyle='dashed')
	plt.plot(dates, price_max,color="black",label="Max BSM",linestyle='dashed')
	last=dates[len(dates)-1]
	#plot the valuation range
	plt.scatter(last,maximum_value,label="Max Value", marker="v", color="red")
	plt.scatter(last,median_value,label="Ave Value",marker="x", color="green")
	plt.scatter(last,minimum_value,label="Min Value", marker="^", color="red")
	plt.legend(loc="lower left", fontsize = 8)
	plt.xticks(fontsize = 8)
	plt.yticks(fontsize = 8)
	cols[1].write(fig)