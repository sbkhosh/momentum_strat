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
from numpy.random import rand
from matplotlib import style
from heapq import nlargest
from pandas import DataFrame
from datetime import datetime
from aiohttp import ClientSession
from random import random

pd.options.mode.chained_assignment = None 

params = {
    'scrape': True,
    'company': 'AAPL',
    'std': '2015-12-31',
    'end': '2018-12-31',
    'boll_days': 20,
    'init_cap': 100000.0
}

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

def process_cols(df,dy):
    df['ma_'+str(dy)] = df['adj_close'].rolling(window=dy).mean() 
    df['std_'+str(dy)] = df['adj_close'].rolling(window=dy).std() 
    df['boll_up_'+str(dy)] = df['ma_'+str(dy)] + 2 * df['std_'+str(dy)] 
    df['boll_dn_'+str(dy)] = df['ma_'+str(dy)] - 2 * df['std_'+str(dy)] 
    return(df)

def viz_data(df,cols):
    df_viz = df[cols]
    df_viz.plot(figsize=(16,9))
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

def strat_gen(df,dy,pltflg):
    # when price touches lower bollinger band => buy signal
    # when price touches upper bollinger band => sell signal

    df['ma_'+ str(dy)] = df['adj_close'].rolling(window=dy).mean() 
    df['std_'+ str(dy)] = df['adj_close'].rolling(window=dy).std() 
    df['boll_up_'+ str(dy)] = df['ma_'+str(dy)] + 2 * df['std_'+str(dy)] 
    df['boll_dn_'+ str(dy)] = df['ma_'+str(dy)] - 2 * df['std_'+str(dy)]
   
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0.0
    signals['close'] = df['adj_close']
    signals['ma_'+ str(dy)] = df['ma_'+ str(dy)]
    signals['boll_up_'+ str(dy)] = df['ma_'+str(dy)] + 2 * df['std_'+str(dy)] 
    signals['boll_dn_'+ str(dy)] = df['ma_'+str(dy)] - 2 * df['std_'+str(dy)]

    conditions  = [ signals['close'][dy:] <= signals['boll_dn_'+ str(dy)][dy:], 
                    list(map(bool,signals['boll_dn_'+ str(dy)][dy:] < signals['close'][dy:]) and
                         map(bool, signals['close'][dy:] < signals['boll_up_'+ str(dy)][dy:])), 
                    signals['close'][dy:] >= signals['boll_up_'+ str(dy)][dy:] ]
    choices     = [ 1, 0, -1 ]    
    signals['signal'][dy:] = np.select(conditions, choices)
    # signals['positions'] = signals['signal'].diff()

    if(pltflg):
        fig = plt.figure(figsize=(16,9))
        ax1 = fig.add_subplot(111, ylabel='Price')
        signals['close'].plot(ax=ax1, color='r', lw=1)
        signals[['boll_up_'+ str(dy), 'boll_dn_'+ str(dy)]].plot(ax=ax1, lw=1)
        ax1.plot(signals.loc[signals.signal == 1.0].index, signals.close[signals.signal == 1.0],
                 '^', markersize=5, color='m')
        ax1.plot(signals.loc[signals.signal == -1.0].index, signals.close[signals.signal == -1.0],
                 'v', markersize=5, color='k')
        plt.show()
        fig.savefig('strat_gen.pdf')
    return(signals)

def strat_bktst(signals,pltflg):
    initial_capital= params['init_cap']

    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions[params['company']] = 100*signals['signal']    
    portfolio = positions.multiply(signals['close'], axis=0)
    pos_diff = positions.diff()
    portfolio['holdings'] = (positions.multiply(signals['close'], axis=0)).sum(axis=1)
    portfolio['cash'] = initial_capital - (pos_diff.multiply(signals['close'], axis=0)).sum(axis=1).cumsum()   
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()

    if(pltflg):
        fig = plt.figure(figsize=(16,9))
        ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')
        portfolio['total'].plot(ax=ax1, lw=2.)
        ax1.plot(portfolio.loc[signals.signal == 1.0].index, 
                 portfolio.total[signals.signal == 1.0],
                 '^', markersize=5, color='m')
        
        ax1.plot(portfolio.loc[signals.signal == -1.0].index, 
                 portfolio.total[signals.signal == -1.0],
                 'v', markersize=5, color='k')
        plt.show()
        fig.savefig('strat_bktst.pdf')
    return(portfolio)

def strat_gen_1(df,pltflg):
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0.0
    signals[['adj_open', 'adj_high', 'adj_low', 'adj_close']] = df[['adj_open', 'adj_high', 'adj_low', 'adj_close']]

    conditions  = [ signals['adj_open'] == signals['adj_high'],
                    signals['adj_close'] == signals['adj_high'] ]
    choices     = [ 1, -1 ]    
    signals['signal'] = np.select(conditions, choices)

    if(pltflg):
        fig = plt.figure(figsize=(16,9))
        ax1 = fig.add_subplot(111, ylabel='Price')
        signals['adj_close'].plot(ax=ax1, color='r', lw=1)
        ax1.plot(signals.loc[signals.signal == 1.0].index, signals.adj_close[signals.signal == 1.0],
                 '^', markersize=5, color='m')
        ax1.plot(signals.loc[signals.signal == -1.0].index, signals.adj_close[signals.signal == -1.0],
                 'v', markersize=5, color='k')
        plt.show()
        fig.savefig('strat_gen_1.pdf')
    return(signals)

def strat_bktst_1(signals,pltflg):
    initial_capital= params['init_cap']

    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions[params['company']] = 100*signals['signal']    
    portfolio = positions.multiply(signals['adj_close'], axis=0)
    pos_diff = positions.diff()
    portfolio['holdings'] = (positions.multiply(signals['adj_close'], axis=0)).sum(axis=1)
    portfolio['cash'] = initial_capital - (pos_diff.multiply(signals['adj_close'], axis=0)).sum(axis=1).cumsum()   
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()

    if(pltflg):
        fig = plt.figure(figsize=(16,9))
        ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')
        portfolio['total'].plot(ax=ax1, lw=2.)
        ax1.plot(portfolio.loc[signals.signal == 1.0].index, 
                 portfolio.total[signals.signal == 1.0],
                 '^', markersize=5, color='m')
        
        ax1.plot(portfolio.loc[signals.signal == -1.0].index, 
                 portfolio.total[signals.signal == -1.0],
                 'v', markersize=5, color='k')
        plt.show()
        fig.savefig('strat_bktst_1.pdf')
    return(portfolio)

def stats(signals,portfolio):
    returns = portfolio['returns']
    sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
    window = 252
    rolling_max = signals['close'].rolling(window, min_periods=1).max()
    daily_drawdown = signals['close']/rolling_max - 1.0
    max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()
    daily_drawdown.plot()
    max_daily_drawdown.plot()
    plt.show()
    days = (signals.index[-1] - signals.index[0]).days
    cagr = ((((signals['close'][-1]) / signals['close'][1])) ** (365.0/days)) - 1
    print(sharpe_ratio,cagr)

def get_candlestick(df):
    df.reset_index(inplace=True)
    fig = go.Figure(data=[go.Candlestick(x=df['date'],
                                         open=df['adj_open'],
                                         high=df['adj_high'],
                                         low=df['adj_low'],
                                         close=df['adj_close'])])

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()

    # f, ax = plt.subplots(figsize=(16,9))
    # ohlc=df[['adj_open', 'adj_high', 'adj_low', 'adj_close']]
    # # plot the candlesticks
    # candlestick_ohlc(ax, ohlc.values, width=.6, colorup='green', colordown='red')
    # # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    
if __name__=='__main__':
    tickers = ['AAPL']
    df_raw = get_data(tickers)

    signals = strat_gen_1(df_raw,True)
    portfolio = strat_bktst_1(signals,True)    

    # days = params['boll_days']
    # signals = strat_gen(df_raw,days,True)
    # portfolio = strat_bktst(signals,True)
    # stats(signals,portfolio)
    
    # df = process_cols(df_raw,days)
    # cols = ['adj_close', 'ma_'+str(days),'boll_up_'+str(days),'boll_dn_'+str(days)]
    # viz_shade(df,cols,days)
    # viz_data(df,cols)
    
