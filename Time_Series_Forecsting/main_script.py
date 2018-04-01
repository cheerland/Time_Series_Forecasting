# -*- coding: utf-8 -*-

'''This is the main script for time series forecasting project.'''

import pandas as pd
from matplotlib import pyplot
import numpy as np
import math
import helper_func as hf 
import stat_models as sm
import eda_plots as ep 


#reading aerosol optical depth data for godavari and talcher coalfields
godavari_df = pd.read_csv('Godavari Coalfield.csv')
godavari_series = pd.Series.from_csv('Godavari Coalfield.csv', header = 0)
talcher_df = pd.read_csv('Talcher Coalfield.csv')
talcher_series = pd.Series.from_csv('Talcher Coalfield.csv', header = 0)

#plotting intial line plots
ep.first_plot(godavari_series)
ep.first_plot(talcher_series)
#plotting initial kde plots
ep.first_plot(godavari_series, 'kde')
ep.first_plot(talcher_series, 'kde')

#print summary of godavari and talcher aerosol optical depth data
print(godavari_series.describe())
print(talcher_series.describe())

#plot timeseries data yearwise
ep.yearwise_div(godavari_series, name = 'Godavari')
ep.yearwise_div(talcher_series, name = 'Talcher')

#plot lag plots 
pd.tools.plotting.lag_plot(talcher_series)
pd.tools.plotting.lag_plot(godavari_series)

#plot autocorrelation plots
ep.correlation_plots(talcher_series, nlags = 10, method = 'ols')
ep.correlation_plots(godavari_series, nlags = 10, method = 'ols')

#plot rolling statistics 
ep.plot_rol(talcher_series, name = 'Talcher')
ep.plot_rol(godavari_series, name = 'Godavari')

#tests stationarity of timeseries data
talcher_ts = sm.test_stationarity(talcher_series, lag = 'AIC')
godavari_ts = sm.test_stationarity(godavari_series, lag = 'AIC')

#calculate moving averages for the timeseries
godavari_ma = hf.moving_avg(godavari_series)
talcher_ma = hf.moving_avg(talcher_series)

#calculate exponential weighted moving averages 
godavari_emwa = hf.emwa(godavari_series)
talcher_emwa = hf.emwa(talcher_series)

#calculates adjacent differences for a timeseries
talcher_diff = hf.differencing(talcher_series)
godavari_diff = hf.differencing(godavari_series)

#decomposes a timeseries
tlchr_dec = hf.decompose(talcher_series)
gdvri_dec = hf.decompose(godavari_series)

#plot decomposed timeseries
ep.decomp_plot(talcher_series, tlchr_dec.trend, tlchr_dec.seasonal, tlchr_dec.residual, name = 'Talcher')
ep.decomp_plot(godavari_series, gdvri_dec.trend, gdvri_dec.seasonal, gdvri_dec.residual, name = 'Godavari')

#AutoRegresive Integrated Moving Average (ARIMA) implementation
godavari_arima, talcher_arima = [], []
godavari_arima_rsqar, talcher_arima_rsqar = [], []
godavari_arima_rmse, talcher_arima_rmse = [], []
godavari_arima_ljung, talcher_arima_ljung = [], []
gdvri_writer_arima = pd.ExcelWriter('Fitted ARIMA Godavari.xlsx')
tlchr_writer_arima = pd.ExcelWriter('Fitted ARIMA Talcher.xlsx')
gdvri_arima_ljung = pd.ExcelWriter('Ljung ARIMA Godavari.xlsx')
tlchr_arima_ljung = pd.ExcelWriter('Ljung ARIMA Talcher.xlsx')
p, i = 5, 0
while p >= 0:
    _ = [[godavari_arima.append(sm.ARIMA(godavari_series, a=p, i=j, m=k)) for j in range(5)] for k in range(5)]
    _2 = [[talcher_arima.append(sm.ARIMA(talcher_series, a=p, i=j, m=k)) for j in range(5)] for k in range(5)]
    _3 = [godavari_arima_rsqar.append(sm.rsquared(godavari_series, godavari_arima[j])) for j in range(25)]
    _4 = [talcher_arima_rsqar.append(sm.rsquared(talcher_series, talcher_arima[j])) for j in range(25)]
    _5 = [godavari_arima_rmse.append(sm.rmse(godavari_series, godavari_arima[j])) for j in range(25)]
    _6 = [talcher_arima_rmse.append(sm.rmse(talcher_series, talcher_arima[j])) for j in range(25)]
    _7 = [godavari_arima_ljung.append(sm.ljung(godavari_series, godavari_arima[j])) for j in range(25)]
    _8 = [talcher_arima_ljung.append(sm.ljung(talcher_series, talcher_arima[j])) for j in range(25)]
    for n in range(25):
        godavari_arima.to_frame.to_excel(gdvri_writer_arima, '{}th file'.format(n))
        talcher_arima.to_frame.to_excel(tlchr_writer_arima, '{}th file'.format(n))
        godavari_arima_ljung.to_frame.to_excel(gdvri_arima_ljung,'{}th file'.format(n))
        talcher_arima_ljung.to_frame.to_excel(tlchr_arima_ljung,'{}th file'.format(n))
   p -= 1
   i += 1

#Markov Chain Monte Carlo implementation
gdvri_writer_proh = pd.ExcelWriter('Fitted MCMC Godavari.xlsx')
tlchr_writer_proh = pd.ExcelWriter('Fitted MCMC Talcher.xlsx')
gdvri_proh_ljung = pd.ExcelWriter('Ljung MCMC Godavari.xlsx')
tlchr_proh_ljung = pd.ExcelWriter('Ljung MCMC Talcher.xlsx')        
godavari_proh, godavari_proh_frcst = sm.prophet(godavari_df, interval_width = 0.95)
talcher_proh, talcher_proh_frcst = sm.prophet(talcher_df, interval_width = 0.95)
godavari_proh_rsqar = sm.rsquared(godavari_series, godavari_proh) 
talcher_proh_rsqar = sm.rsquared(talcher_series, talcher_proh)
godavari_proh_rmse = sm.rmse(godavari_series, godavari_proh) 
talcher_proh_rmse = sm.rmse(talcher_series, talcher_proh)
godavari_proh_ljung = sm.ljung(godavari_series, godavari_proh) 
talcher_proh_ljung = sm.ljung(talcher_series, talcher_proh)
godavari_proh_ra = sm.residualanalysis(godavari_series, godavari_proh) 
talcher_proh_ra = sm.residualanalysis(talcher_series, talcher_proh)
godavari_proh.to_excel(gdvri_writer_proh, 'Godavari')
talcher_proh.to_excel(tlchr_writer_arima, 'Talcher')
godavari_proh_ljung.to_excel(gdvri_proh_ljung, 'Godavari')
tlchr_proh_ljung.to_excel(tlchr_proh_ljung, 'Talcher')
