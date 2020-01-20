#!/usr/bin/python3

import bs4 as bs
import datetime as dt
import os
import pickle
import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import quandl
import sklearn
import scipy.stats as scs
import statsmodels.api as sm
from dateutil import parser
from pandas.plotting import register_matplotlib_converters
from matplotlib import style
from sklearn import preprocessing, svm
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,VotingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.externals import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, \
    RobustScaler, PowerTransformer, QuantileTransformer, Normalizer
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from pprint import pprint
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX


def tsplot(y, lags=None, figsize=(15, 10), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
        plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
        plt.show()

pd.options.mode.chained_assignment = None 

# [ 'lin', 'svm-lin', 'svm-poly', 'tree', 'forest', 'xgbr', 'nn', 'comb', 'voting']

params = {
    'check_shape': False,
    'test_size': 0.2,
    'grid_search': True,
    'regrs': 'xgbr',
}

def build_grid_rf():
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 20, stop = 720, num = 20)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    # bootstrap = [True, False]

    grid = {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf
    }
    return(grid)

def build_grid_xgbr():
    # learning_rate = [ float(x) for x in np.linspace(start = 0.1, stop = 1.0, num = 100) ]
    # n_estimators = [ 100 ]
    # max_depth = [ 2 ]
    # min_child_weight = [ 1 ]
    # nthread = [ 1 ] 
    # subsample = [ 0.15 ]

    learning_rate = [float(x) for x in np.linspace(start = 0.1, stop = 1.0, num = 181)]
    n_estimators = [int(x) for x in np.linspace(start = 25, stop = 150, num = 6)]
    max_depth = [int(x) for x in np.linspace(1, 10, num = 5)]
    min_child_weight = [int(x) for x in np.linspace(1, 20, num = 5)]
    nthread = [ 1 ] 
    subsample = [float(x) for x in np.linspace(start = 0.05, stop = 0.5, num = 5)]
    
    
    grid = {'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'nthread': nthread,
            'subsample': subsample
    }
    
    return(grid)

def build_grid_nn():
    hidden_layer_sizes = [ (64,64), (16,16), (8,8), (4,4) ]
    activation = ['relu', 'tanh']
    solver = ['adam']
    alpha = [float(x) for x in np.linspace(start = 0.001, stop = 0.01, num = 10)]
    learning_rate_init = [float(x) for x in np.linspace(start = 0.001, stop = 0.01, num = 9)]
    learning_rate = ['constant', 'adaptive'] 
    max_iter = [int(x) for x in np.linspace(50, 300, num = 6)]
    tol = [0.0001]
    momentum = [0.1,0.5,0.9]
    beta_1=[0.1,0.5,0.9]
    beta_2=[0.111,0.555,0.999]
    n_iter_no_change=[5,10]

    grid = {'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation,
            'solver': solver,
            'alpha': alpha,
            'learning_rate_init': learning_rate_init,
            'learning_rate': learning_rate,           
            'max_iter': max_iter,
            'tol': tol,
            'momentum': momentum,
            'beta_1': beta_1,
            'beta_2': beta_2,
            'n_iter_no_change': n_iter_no_change
    }
    return(grid)

def grid_search(reg,X_train,y_train,random_grid):
    print('#############################################')
    print('parameters in use before grid search')
    print('#############################################')

    pprint(reg.get_params())
    prms = reg.get_params()
    
    reg_random = RandomizedSearchCV(estimator=reg,param_distributions=random_grid,n_iter=100,cv=5, \
                                    verbose=2,random_state=42,n_jobs=-1)
    reg_random.fit(X_train,y_train)
    prms = reg_random.best_params_

    print('#############################################')
    print('best parameters after grid search')
    print('#############################################')
    pprint(reg_random.best_params_)

    return(prms)

def set_grid_search(regrs,reg):
    if(regrs=='rf'):
        random_grid = build_grid_rf()
        prms = grid_search(reg,X_train,y_train,random_grid)
        reg_prms = RandomForestRegressor(n_estimators=prms['n_estimators'],max_features=prms['max_features'],max_depth=prms['max_depth'],\
                                         min_samples_split=prms['min_samples_split'],min_samples_leaf=prms['min_samples_leaf'],random_state=24361,\
                                         n_jobs=-1) 
    elif(regrs=='xgbr'):
        random_grid = build_grid_xgbr()
        prms = grid_search(reg,X_train,y_train,random_grid)
        reg_prms = XGBRegressor(learning_rate=prms['learning_rate'], max_depth=prms['max_depth'], min_child_weight=prms['min_child_weight'], \
                                n_estimators=prms['n_estimators'], nthread=prms['nthread'], subsample=prms['subsample'],random_state=24361,\
                                n_jobs=-1)
    elif(regrs=='nn'):
        random_grid = build_grid_nn()
        prms = grid_search(reg,X_train,y_train,random_grid)
        reg_prms = MLPRegressor(hidden_layer_sizes=prms['hidden_layer_sizes'],activation=prms['activation'],solver=prms['solver'],\
                                alpha=prms['alpha'],learning_rate_init=prms['learning_rate_init'],learning_rate=prms['learning_rate'],
                                max_iter=prms['max_iter'],tol=prms['tol'],momentum=prms['momentum'],beta_1=prms['beta_1'],beta_2=prms['beta_2'],\
                                n_iter_no_change=prms['n_iter_no_change'],random_state=24361)
    return(reg_prms)


def check_shape(X_train, X_test, y_train, y_test):
    print('X_train.shape, X_test.shape = ', X_train.shape, X_test.shape)
    print('y_train.shape, y_test.shape = ', y_train.shape, y_test.shape)

def write_to(df,name,flag):
    try:
        if(flag=="csv"):
            df.to_csv("stock_dfs/" + str(name)+".csv")
        elif(flag=="html"):
            df.to_html(str(name)+"html")
    except ValueError:
        print("No other types supported")

def get_tickers(tickers):
    for ticker in tickers:
        ticker = str(ticker)
        try: 
            print(ticker)
            quandl.ApiConfig.api_key = "M3S6cLgQ3b_czSDmKJxD"
            df = quandl.get("WIKI/" + ticker, start_date = "2015-12-31", end_date = "2018-12-31")
            write_to(df,str(ticker),"csv")
        except ValueError:
            print("Error")
            print(ticker)
           
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
    
def read_data(ticker):
    out = pd.DataFrame()
    filename = 'stock_dfs/' + str(ticker) + '.csv'
    df = pd.read_csv(filename)[['Date', 'Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
    df.columns = ['date','open','high','low','close','volume']
    df = df.set_index(['date'])
    return(df)

def view_data(df,flag):
    try:
        if(flag==0):
            print(df.head())
        elif(flag==1):
            print(df.tail())
    except:
        print("No valid flag choice")

def get_mom(df):
    mtum = pd.DataFrame(index=df.index)
    
    mtum['close'] = df.close
    mtum['volume'] = df.volume
    
    mtum['vel'] = mtum.close.diff(1)
    mtum['mom'] = mtum.vel * df.volume
    mtum['acc'] = mtum.vel.diff(1)
    mtum.dropna(inplace=True)
    
    return(mtum)

def get_k_param(df):
    max_close = np.max(df['close'])
    min_close = np.min(df['close'])
    ntot = len(df)

    ups = np.sum(df['vel'] > 0)
    dws = ntot - ups

    kprm = float(ups)/float(ntot)
    return(kprm)

def scaler_def(index):
    if(index == 0):
        scl = StandardScaler()
    elif(index == 1):
        scl = MinMaxScaler()
    elif(index == 2):
        scl = MaxAbsScaler()
    elif(index == 3):
        scl = RobustScaler(quantile_range=(25, 75))
    elif(index == 4):
        scl = PowerTransformer(method='yeo-johnson')
    elif(index == 5):
        scl = PowerTransformer(method='box-cox')
    elif(index == 6):
        scl = QuantileTransformer(output_distribution='normal')
    elif(index == 7):
        scl = QuantileTransformer(output_distribution='uniform')
    elif(index == 8):
        scl = Normalizer()
    else:
        print('not a correct scaler defined')
    return(scl)

def regressors(regrs):   
    if(regrs == 'lin'):
        reg = LinearRegression(n_jobs=-1)
    elif(regrs == 'svm-lin'):
        reg = svm.SVR(kernel='linear',gamma='auto')
    elif(regrs == 'svm-poly'):
        reg = svm.SVR(kernel='poly',gamma='auto')
    elif(regrs == 'tree'):
        reg = DecisionTreeRegressor(random_state=24361)
    elif(regrs == 'forest'):
        reg = RandomForestRegressor(n_estimators=100,random_state=24361,n_jobs=-1) 
    elif(regrs == 'xgbr'):
        reg = XGBRegressor(learning_rate=0.0471, max_depth=2, min_child_weight=1, \
                           n_estimators=100, subsample=0.15)
        # reg = XGBRegressor(learning_rate=LR_LR, max_depth=2, min_child_weight=1, \
        #                    n_estimators=100, subsample=SUB_SUB)
    elif(regrs == 'nn'):
        reg = MLPRegressor(hidden_layer_sizes=(3,3), activation='tanh', solver='adam', alpha=0.001, \
                           learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=50, \
                           tol=0.0001, momentum=0.5, nesterovs_momentum=True, validation_fraction=0.1, \
                           beta_1=0.1, beta_2=0.555, epsilon=1e-08, n_iter_no_change=50, random_state=24361)
    elif(regrs == 'comb'):
        xgbr = XGBRegressor(learning_rate=0.045, max_depth=2, min_child_weight=1,n_estimators=100, subsample=0.15)
        frst = RandomForestRegressor(n_estimators=100,random_state=24361,n_jobs=-1)
        dtr = DecisionTreeRegressor(random_state=24361)
        nn = MLPRegressor(hidden_layer_sizes=(64,64), activation='tanh', solver='adam', learning_rate_init=0.045)
        reg = StackingRegressor(regressors=[xgbr,nn],meta_regressor=xgbr)
    elif(regrs == 'voting'):
        frst = RandomForestRegressor(n_estimators=100,random_state=24361,n_jobs=-1)
        dtr = DecisionTreeRegressor(random_state=24361)
        reg = VotingClassifier(estimators=[('frst',frst),('dtr',dtr)],voting='hard')

    return(reg)

def ml_train_test(dtf):   
    cols = dtf.columns.values
    df = dtf[cols]
    
    # df[cols] = preprocessing.scale(df[cols])
    # scaler = scaler_def(0)
    # df[cols] = scaler.fit_transform(df[cols])

    X = np.array(df.drop(['vel'],axis=1))
    y = np.array(df['vel'])  

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=params['test_size'],random_state=47252)
    if(params['check_shape']):
        check_shape(X_train, X_test, y_train, y_test)

    return(X_train, X_test, y_train, y_test)

def model(X_train,X_test,y_train,y_test):
    if(params['grid_search']):
        reg_init = regressors(params['regrs'])
        reg = set_grid_search(params['regrs'],reg_init)
    else:
        reg = regressors(params['regrs'])
       
    reg.fit(X_train,y_train)        
    predictions = reg.predict(X_test)       
    accuracy = r2_score(y_test,predictions)

    print(accuracy)

def plot_vmom_acc(df, figsize=(15, 10)):
    ts0 = df['close']
    ts1 = df['vel']
    ts2 = df['mom']
    ts3 = df['acc']

    # with plt.style.context(style):    
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)

    ts_ax = plt.subplot2grid(layout, (0, 0))
    vel_ax = plt.subplot2grid(layout, (0, 1))
    mom_ax = plt.subplot2grid(layout, (1, 0))
    acc_ax = plt.subplot2grid(layout, (1, 1))
        
    ts0.plot(ax=ts_ax,grid=True)
    ts_ax.set_title('price')
    
    ts1.plot(ax=vel_ax,grid=True)
    vel_ax.set_title('velocity')
    
    ts2.plot(ax=mom_ax,grid=True)
    mom_ax.set_title('momentum')
    
    ts3.plot(ax=acc_ax,grid=True)
    acc_ax.set_title('acceleration')
    
    plt.tight_layout()
    plt.show()

def trends(df):
    ts0 = df['close'].rolling(12).mean()
    ts1 = df['close'].diff()
    ts2 = detr.diff(12)

    fig = plt.figure(figsize=figsize)
    layout = (3, 1)

    ts0_ax = plt.subplot2grid(layout, (0, 0))
    ts1_ax = plt.subplot2grid(layout, (0, 1))
    ts2_ax = plt.subplot2grid(layout, (1, 0))
        
    ts0.plot(ax=ts0_ax,grid=True)
    ts0_ax.set_title('moving average')
    
    ts1.plot(ax=ts1_ax,grid=True)
    ts1_ax.set_title('detrended')
    
    ts2.plot(ax=ts2_ax,grid=True)
    ts2_ax.set_title('seasomal detrended')
       
    plt.tight_layout()
    plt.show()
    
def sarimax_model():
    model = SARIMAX(df['close'],order=(1, 1, 1),seasonal_order=(1, 1, 1, 12))
    result = model.fit()
    print(result.summary().tables[1])

    
if __name__ == '__main__':
    tickers = ['AAPL','CSCO', 'AMZN']
    idx = 0
    ticker = tickers[idx]
    # get_tickers(tickers)

    prices = read_data(ticker)
    mom = get_mom(prices)

    print(get_k_param(mom))
    
    # X_train, X_test, y_train, y_test = ml_train_test(mom)
    # model(X_train,X_test,y_train,y_test)
    
