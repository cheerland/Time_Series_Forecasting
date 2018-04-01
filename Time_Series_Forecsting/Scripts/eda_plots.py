# -*- coding: utf-8 -*-

'''This script contains all the necessary functions to plot different types of plots required during the exploratory data analysis of a time series data '''

import pandas as pd
from matplotlib import pyplot
import numpy as np
from statsmodels.tsa.stattools import acf, pacf


#plots a simple lineplot or a special plot 
def first_plot(timeseries, kind = None):
    a = range(len(timeseries))
    if kind == None:
        pyplot.plot(a,timeseries.values)
        pyplot.title('Area Averaged Time Series of Talcher')
        pyplot.axis([0,200,0,1.5])
        pyplot.savefig('Talcher TS plot.png', dpi = 1000, format = 'png')
    else:
        timeseries.plot(kind = kind)
    	pyplot.title('Distribution of Time Series data of Talchar')
    	pyplot.savefig('Talcher TS Distribution plot.png', dpi = 1000, format = 'png')

#plots yearwise timeseries for given timeseries
def yearwise_div(timeseries, name):
    groups = timeseries.groupby(pd.TimeGrouper('A'))
    years = pd.DataFrame()
    for name, group in groups :
        years[name.year] = group.values
    
    #yearwwise plots of given timeseries
    years.plot(subplots = True, legend = False)
    pyplot.title('{} yearwise TS plot'.format(name))
    pyplot.savefig('{} yearwise TS plot.png'.format(name), dpi = 1000, format = 'png')
       
    years.ix[0:12, [2001,2002,2003,2004]].plot(subplots = True, legend = True)
    pyplot.axis([0,15,0,1.0])
    pyplot.savefig('Yearwise TS plot I.png', dpi = 1000, format = 'png')
    
    years.ix[0:12, [2005,2006,2007,2008]].plot(subplots = True, legend = True)
    pyplot.axis([0,15,0,1.5])
    pyplot.savefig('Yearwise TS plot II.png', dpi = 1000, format = 'png')
    
    years.ix[0:12, [2009,2010,2011,2012]].plot(subplots = True, legend = True)
    pyplot.axis([0,15,0,1.0])
    pyplot.savefig('Yearwise TS plot III.png', dpi = 1000, format = 'png')
    
    years.ix[0:12, [2013,2014,2015,2016]].plot(subplots = True, legend = True)
    pyplot.axis([0,15,0,1.0])
    pyplot.savefig('Yearwise TS plot IV.png', dpi = 1000, format = 'png')

#plots rolling statistics 
def plot_rol(timeseries, window = 12, name):
	b = np.arange(0, len(timeseries))
    #Determining Rolling Statistics
    rolmean = pd.rolling_mean(timeseries, window = window)
    rolstd = pd.rolling_std(timeseries, window = window)
    orig = pyplot.plot(b, timeseries.values, color = 'blue', label = 'Original')
    mean = pyplot.plot(b, rolmean.values, color = 'red', label = 'Rolling Mean')
    std = pyplot.plot(b, rolstd.values, color = 'green', label = 'Rolling Std')
    pyplot.legend(loc = 'best')
    pyplot.title('Rolling Mean and Standard Deviation for {}'.format(name))
    pyplot.savefig('Original, Rolling Mean and Standard Deviation plot.png', dpi = 1000,
                   format = 'png')
    pyplot.show(block = False)

#plots decomposed timeseries
def decomp_plot(timeseries, trend, seasonal, residual, name):
	b = np.arange(0, len(timeseries))
	pyplot.subplot(411)
    pyplot.plot(b, timeseries.values, label = 'Original')
    pyplot.legend(loc = 'best')
    pyplot.subplot(412)
    pyplot.plot(b, trend.values, label = 'Trend')
    pyplot.legend(loc = 'best')
    pyplot.subplot(413)
    pyplot.plot(b, seasonal.values, label = 'Seasoanal')
    pyplot.legend(loc = 'best')
    pyplot.subplot(414)
    pyplot.plot(b, residual.values, label = 'Residual')
    pyplot.legend(loc = 'best')
    pyplot.tight_layout()
    pyplot.title('{} decomposed plot'.format(name))

#plots correlation and autocorrelation plots
def correlation_plots(timeseries, nlags = 10, method = 'ols'):
    lag_acf = acf(timeseries, nlags = nlags)
    lag_pacf = pacf(timeseries, nlags = nlags, method = method)
    
    #plot ACF
    pyplot.subplot(121)
    pyplot.plot(lag_acf)
    pyplot.axhline(y = 0, linestyle = '--', color = 'black')
    pyplot.axhline(y = -1.96/(np.sqrt(len(timeseries))), linestyle = '--', color = 'black')
    pyplot.axhline(y = 1.96/(np.sqrt(len(timeseries))), linestyle = '--', color = 'black')
    pyplot.title('Autocorrelation Plot')
    
    #plot PACF
    pyplot.subplot(122)
    pyplot.plot(lag_pacf)
    pyplot.axhline(y = 0, linestyle = '--', color = 'black')
    pyplot.axhline(y = -1.96/(np.sqrt(len(timeseries))), linestyle = '--', color = 'black')
    pyplot.axhline(y = 1.96/(np.sqrt(len(timeseries))), linestyle = '--', color = 'black')
    pyplot.title('Partial Autocorrelation Plot')
    pyplot.tight_layout()
    pyplot.savefig('Correlation Plots.png', dpi = 1000, format = 'png')
