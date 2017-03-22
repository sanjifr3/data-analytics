import urllib2
import pandas as pd
import datetime
from bs4 import BeautifulSoup
import ast

# Open URL and read it
url = urllib2.urlopen("https://poloniex.com/public?command=returnChartData&currencyPair=BTC_ETH&start=1435699200&end=9999999999&period=14400")
content = url.read()

soup = BeautifulSoup(content,"lxml")
#print (soup.prettify())
url_dict = ast.literal_eval(soup.body.string)
url_df = pd.DataFrame(url_dict)
url_df['date'] =  url_df['date'].apply(lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))

#print url_df

#print url_df[url_df['date'] == 1439006400]

url_df.to_csv("btc_to_eth_historical_data.csv", index=False)


# Open url and re


url = urllib2.urlopen("https://poloniex.com/public?command=returnChartData&currencyPair=USDT_ETH&start=1435699200&end=9999999999&period=14400")
content = url.read()

soup = BeautifulSoup(content,"lxml")
url_dict = ast.literal_eval(soup.body.string)
url_df = pd.DataFrame(url_dict)
url_df['date'] =  url_df['date'].apply(lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))
url_df.to_csv("usd_to_eth_historical_data.csv", index=False)



