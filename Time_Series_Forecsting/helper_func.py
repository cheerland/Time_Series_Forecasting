# -*- coding: utf-8 -*-

'''This script contains different helper functions to perform different manipulations on a time series data'''


import pandas as pd
from matplotlib import pyplot
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


#calculates and returns moving average on a timeseries 
def moving_avg(timeseries):
    mov_avg = pd.rolling_mean(timeseries, 12)
    series_diff = timeseries-mov_avg
    series_diff.dropna(inplace = True)
    return series_diff

#calculates and returns exponential weighted moving average on a timesries
def emwa(timeseries):
    b = np.arange(0, len(timeseries))
    emw_avg = pd.ewma(timeseries, halflife =12)
    pyplot.plot(b, timeseries.values, color = 'blue')
    pyplot.plot(b,emw_avg.values, color = 'red')
    emwa_diff = timeseries-emw_avg
    return emwa_diff
 
#calculates and returns adjacent differences on a timeseries 
def differencing(timeseries):
    b = np.arange(0, len(timeseries))
    series_diff = timeseries-timeseries.shift()
    pyplot.plot(b, series_diff.values)
    series_diff.dropna(inplace = True)
    return series_diff

#decomposes a timeseries into trends, residual and seasonal components
def decompose(timeseries, freq = 12):
    decomposition = seasonal_decompose(talcher_series, freq = freq)
    return decomposition

