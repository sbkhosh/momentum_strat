#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import re
import urllib
import itertools
import asyncio
import grequests
import concurrent.futures
import bs4 as bs
import requests
import glob
import datetime
import lxml.html as lh
import dateutil
import quandl
import plotly.graph_objects as go
import yfinance as yf
import pickle
from numpy.random import rand
from matplotlib import style
from heapq import nlargest
from pandas import DataFrame
from datetime import datetime
from aiohttp import ClientSession
from random import random

pd.options.mode.chained_assignment = None 

def get_headers(df):
    return(df.columns.values)

def read_data(path):
    df = pd.read_csv(path,sep=',')
    return(df)

def view_data(df):
    print(df.head(20))
  
def get_info(df):
    df.info()
    df.describe()
    
def format_output(df):
    df.columns = [''] * len(df.columns)
    df = df.to_string(index=False)
    return(df)
   
def show_cmtx(df):
    corr_matrix = df.corr()
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_matrix, cmap=cmap, center=0, square=True, linewidths=0.5)
    plt.show()
    
def omit_by(dct, predicate=lambda x: x!=0):
    return({k:v for k,v in dct.items() if predicate(v)})
    
def check_missing_data(df):
    res = df.isnull().sum().sort_values(ascending=False)
    print(res)
    
    if(sum(res.values) != 0):
        kv_nz = omit_by(res)
        for el in kv_nz.keys():        
            print(df[df[str(el)].isnull()])

def write_to(df,name,flag):
    try:
        if(flag=="csv"):
            df.to_csv(str(name)+".csv",index=False)
        elif(flag=="html"):
            df.to_html(str(name)+"html",index=False)
    except:
        print("No other types supported")
        
def grouping(df,index):
    grp = df.groupby([str(index)]).agg(\
                                      bs_sum=pd.NamedAgg(column='base salary', aggfunc='sum'),\
                                      bs_mean=pd.NamedAgg(column='base salary', aggfunc='mean'),\
                                      bs_max=pd.NamedAgg(column='base salary', aggfunc='max'),\
                                      job_cnt=pd.NamedAgg(column='job title', aggfunc='count'),\
    )
    grp.reset_index(inplace=True)
    return(grp)
    
def get_data(tickers):
    quandl.ApiConfig.api_key = "M3S6cLgQ3b_czSDmKJxD"
    df = quandl.get_table('WIKI/PRICES', ticker = tickers, \
                          qopts = { 'columns': ['ticker', 'date', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume' ] }, \
                          date = { 'gte': params['std'], 'lte': params['end'] }, \
                          paginate=True)   
    df = df.set_index('date')
    idx = df.index.values[::-1]
    df = df.reindex(idx,axis='index')
    # df = df.pivot(columns='ticker')
    return(df)

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.find('a', {'class': 'external text', 'rel': True}).text
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    with open("sp500tickers.txt", 'w') as f:
        for item in tickers:
            f.write("%s\n" % item)        
    return(tickers)

def viz_data(df,cols):
    df.reset_index(inplace=True)
    df.set_index('Date', inplace=True)
    grouped = df.groupby('Ticker')[cols]
    ncols=len(df.groupby('Ticker').groups.keys())
    nrows = int(np.ceil(grouped.ngroups/ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,4), sharey=False)
    for (key, ax) in zip(grouped.groups.keys(), axes.flatten()):
        grouped.get_group(key).plot(ax=ax,title=key)
    ax.legend()
    plt.show()

def viz_shade(df,cols,dy):
    df_viz = df[cols]
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    x_axis = df_viz.index.get_level_values(0)
    
    ax.fill_between(x_axis, df_viz['boll_up_'+str(dy)], df_viz['boll_dn_'+str(dy)], color='grey')
    ax.plot(x_axis, df_viz['adj_close'], color='blue', lw=2)
    ax.plot(x_axis, df_viz['ma_'+str(dy)], color='black', lw=2)    
    plt.show()

def process_dfs(tickers,dy):
    dfs = []
    weights = [0.4,0.6]
    for i,el in enumerate(tickers):
        df = yf.Ticker(str(el)).history(period="1Y")[['Close','Volume']]
        df['Ticker'] = str(el)
        df['weight'] = weights[i]
        dfs.append(df)
      
    res = pd.concat(dfs,axis=0)
    res.reset_index(inplace=True)
    res = res.set_index(['Date','Ticker'])
    res['pct_change'] = res.groupby(level='Ticker')['Close'].transform(lambda x: x.pct_change())
    res['mean'] = res.groupby(level='Ticker')['Close'].transform(lambda x: x.mean())

    res['mean_weight'] = res['mean'] * res['weight']
    
    res['ma_'+str(dy)] = res.groupby(level='Ticker')['Close'].transform(lambda x: x.rolling(window=dy).mean())
    res['std_'+str(dy)] = res.groupby(level='Ticker')['Close'].transform(lambda x: x.rolling(window=dy).std())
    res['boll_up_'+str(dy)] = res.groupby(level='Ticker')['ma_'+str(dy)].transform(lambda x: x) + \
                                                 2.0 * res.groupby(level='Ticker')['std_'+str(dy)].transform(lambda x: x)
    res['boll_dn_'+str(dy)] = res.groupby(level='Ticker')['ma_'+str(dy)].transform(lambda x: x) - \
                                                 2.0 * res.groupby(level='Ticker')['std_'+str(dy)].transform(lambda x: x)
    res['cum_ret'] = res.groupby(level='Ticker')['pct_change'].transform(lambda x: (1.0+x).cumprod())
    return(res)
  
    # res.dropna(inplace=True)
    # total_relative_returns = (np.exp(portfolio_log_ret.cumsum()) - 1)
   
if __name__=='__main__':
    tickers = save_sp500_tickers()
    df = process_dfs(['AAPL','AMZN'],20)
    viz_data(df,['mean','mean_weight'])
