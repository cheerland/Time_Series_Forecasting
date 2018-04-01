# -*- coding: utf-8 -*-

'''This script contains the necessary functions to perform statistical tests, implement ARIMA, SARIMA and MCMC time series modeling, and check the accuracy
of the fitted model using different evaluation metrics '''


import pandas as pd
from matplotlib import pyplot
import numpy as np
import math
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from fbprophet import Prophet


#Perform Dicky-Fuller Test
def test_stationarity(timeseries, lag = 'AIC'):  
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag = lag)
    dfoutput = pd.Series(dftest[0:4], index = ['Test Static', 'P value', 'No. of Lags Used',
                         'No. of Observations used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput
 
#implements auto-regressive integrated moving average model on the timeseries
#plots and returns ARIMA fitted values
def ARIMA(timeseries, a = 0, i = 0, m = 0):
    series_diff = timeseries-timeseries.shift()
    series_diff.dropna(inplace = True)
    b = np.arange(0, len(series_diff))
    model = ARIMA(timeseries, order = (a,i,m))
    results_ARIMA = model.fit(disp = -1)
    pyplot.plot(b, series_diff.values, color = 'blue')
    pyplot.plot(b, results_ARIMA.fittedvalues, color = 'red')
    pyplot.title('ARIMA Fitted  Values')
    pyplot.savefig('ARIMA Fitted  Values.png', dpi = 1000, format = 'png')
    print('AIC Value:',results_ARIMA.aic)
    return results_ARIMA.fittedvalues

#implements markov chain monte carlo and forecasts on the fitted model based on given period and frequency
def prophet(timedf, interval_width = 0.9,periods = 36, freq = 'MS', uncertainty = False):
    model = Prophet(interval_width = interval_width) # set the uncertainty interval to 95% (the Prophet default is 80%)
    model_df = timedf.rename((columns={'Time': 'ds', 'AOD 550 nm': 'y'}))
    model_fit = model.fit(model_df)
    future_dates = model.make_future_dataframe(periods = periods, freq = freq) #creates future dates
    forecast = model.predict(future_dates)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    model.plot(forecast, uncertainty = uncertainty)
    model.plot_components(forecast)
    return model_fit, forecast

#implements seasonal arima on the time series and returns the fitted model
def SARIMA(timeseries, a = 0, i = 0, m = 0):
    mod = sm.tsa.SARIMAX(timeseries, trend='n', order=(a,i,m), seasonal_order=(1,1,1,12))
    results = mod.fit()
    print (results.summary())
    return results

#performs scaling on the fitted model, timeseries1 should be original and timeseries2 shoud be fitted model
def scaling_back(timeseries1, timeseries2):
    predictions_diff = pd.Series(timeseries2, copy=True)
    predictions = pd.Series(timeseries1.ix[0], index=timeseries1.index)
    predictions = predictions.add(predictions_diff.cumsum(),fill_value=0)
    b = np.arange(0, len(timeseries1))
    c = np.arange(0, len(predictions))
    pyplot.subplot(121)
    pyplot.plot(b, timeseries1)
    pyplot.title('Talcher Original Data')
    pyplot.subplot(122)
    pyplot.plot(c, predictions)
    pyplot.title('Talcher Fitted Data')
    pyplot.tight_layout()
    pyplot.savefig('Talcher Scaledback  Values.png', dpi = 1000, format = 'png')
    return predictions

# performs R Squared calculation, takes original series and fitted model as input
def rsquared(timeseries1, timeseries2):
	timeseries2 = pd.Series(timeseries2, copy=True)
    model_residuals = math.sqrt(pd.Series.mean((timeseries1 - timeseries2)**2))
    mean_residuals = math.sqrt(pd.Series.mean(timeseries1 - pd.Series.mean(timeseries1)**2))
    rsquare = 1- model_residuals/mean_residuals
    return rsquare

# Root Mean Squared calculation, takes original series and fitted model as input 
def rmse(timeseries1, timeseries2):
	timeseries2 = pd.Series(timeseries2, copy=True)
    model_rmse = math.sqrt(pd.Series.mean((timeseries1 - timeseries2)**2))
    return model_rmse

#performs residual analysis and plots auto-correlated and partial auto-correlated plots
#takes original series and fitted model as input
def residualanalysis(timeseries1, timeseries2, method = 'ols', plots = True):
    timeseries2 = pd.Series(timeseries2, copy=True)
    residual = timeseries1-timeseries2
    lag_acf = acf(residual, nlags = 50)
    lag_pacf = pacf(residual, nlags = 50, method = method)
    if plots == True:
    	pyplot.subplot(121)
    	pyplot.plot(lag_acf)
    	pyplot.axhline(y = 0, linestyle = '--', color = 'black')
    	pyplot.axhline(y = -1.96/(np.sqrt(len(residual))), linestyle = '--', color = 'black')
    	pyplot.axhline(y = 1.96/(np.sqrt(len(residual))), linestyle = '--', color = 'black')
    	pyplot.title('Residual Autocorrelation Plot')
        #plot partial auto-correlation plot
    	pyplot.subplot(122)
    	pyplot.plot(lag_pacf)
    	pyplot.axhline(y = 0, linestyle = '--', color = 'black')
    	pyplot.axhline(y = -1.96/(np.sqrt(len(residual))), linestyle = '--', color = 'black')
    	pyplot.axhline(y = 1.96/(np.sqrt(len(residual))), linestyle = '--', color = 'black')
    	pyplot.title('Residual Partial Autocorrelation Plot')
    	pyplot.tight_layout()
    	pyplot.savefig('Correlation Plots.png', dpi = 1000, format = 'png')
    return residual

#performs Ljungbox Test to daigonise fitted model and returns a dataframe with results
#takes original series and fitted model as input
def ljung(timeseries1, timeseries2, lags = 10):
    timeseries2 = pd.Series(timeseries2, copy=True)
    lbvalue, pvalue = acorr_ljungbox((timeseries1-timeseries2), lags=lags, boxpierce=False)
    df = pd.DataFrame({'Test_Statistic':lbvalue, 'P_value':pvalue })
    return df

