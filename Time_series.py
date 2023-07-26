# %%
import pandas as pd
import numpy as np

# %%
import warnings
warnings.filterwarnings("ignore") # To prevent kernel from showing any warning

# %%
dataset = pd.read_csv("./Ak-new-csv.csv")
dataset.head()

# %%
dataset.shape

# %%
dataset.isnull().sum()

# %%
dataset.columns = dataset.columns.str.lower()

# %%
dataset.columns

# %%
series_data = dataset.loc[:, ['date', 'total', 'amount']]
series_data

# %%
date_arrays = []
for i in series_data["date"].values:
    d = i.split("/")
    vals = d[1:]
    date_join = "/".join(vals)
    date_arrays.append(date_join)
series_data["month_year"] = date_arrays

# %%
series_data

# %%
mean_t = series_data['total'].mean()
std_t = series_data['total'].std()
def fillna(value):
    if value != np.nan:
        value = np.random.normal(loc=mean_t, scale=std_t)
    return np.abs(int(value))

# %%
series_data.loc[:, 'total'] = series_data.loc[:, 'total'].apply(lambda x: fillna(x))

# %%
series_data.isnull().sum()

# %%
series_data

# %%
series_data = series_data.groupby("month_year")["total", "amount"].mean()
series_data = series_data.reset_index()
series_data.head()

# %%
series_data

# %%
series_data['total'] = series_data['total'].astype(np.int64)
series_data['amount'] = series_data['amount'].astype(np.int64)

# %%
amount_series = series_data.drop('total', axis=1)
attendance_series = series_data.drop('amount', axis=1)

# %%
amount_series.head()

# %%
attendance_series.head()

# %%
amount_series = amount_series.set_index('month_year')


# %%
attendance_series = attendance_series.set_index('month_year')

# %%
amount_series

# %%
attendance_series

# %% [markdown]
# ## INCOME

# %%
import matplotlib.pyplot as plt

# %%
amount_series = amount_series.sort_index()

# %%
index =  pd.to_datetime(amount_series.index)

# %%
amount_series.index = index

# %%
fig, ax = plt.subplots(figsize=(8, 5))
amount_series.plot(kind='line', ax=ax, grid=True)
ax.set_title("Time series of Monthly income")
plt.show()

# %% [markdown]
# **Time Series Decomposition:**   
# A time series is usually composed of the following components:
#    > **1) Trend :** This component usually is increasing, decreasing, or constant.  
#    > **2) Seasonality :** This is the periodic behavior of the time series that occurs within a year.   
#    > **3) Residual :** This is what remains of the time series after the trend and seasonality are removed.  
# 
# The basic approach to seasonal decomposition splits the time series into above components.

# %%
from statsmodels.tsa.seasonal import seasonal_decompose

# %%
decompose_add = seasonal_decompose(amount_series.values, period=12)
decompose_add.plot()
plt.show()

# %% [markdown]
# **Stationary Time Series :**   
# A stationary time series is one whose statistical properties such as mean, variance, autocorrelation, etc. are all constant over time. Most statistical forecasting methods are based on the assumption that the time series can be rendered approximately stationary (i.e., "stationarized") through the use of mathematical transformations. A stationarized series is relatively easy to predict
# 
# Sign of obvious trends, seasonality, or other systematic structures in the series are indicators of a non-stationary series. A more accurate method would be to use a statistical test, such as the Dickey-Fuller test.
# 
# **ADFuller Test:**  
# If Test statistic < Critical Value and p-value < 0.05 – then series is stationary

# %%
from statsmodels.tsa.stattools import adfuller

# %%
# functon for adf test
def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    print ('----------------------------------------------')
    adftest = adfuller(timeseries)
    adf_output = pd.Series(adftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in adftest[4].items():
        adf_output['Critical Value (%s)'%key] = value
    print (adf_output)


# %%
# calling adf function and passing series
adf_test(amount_series.values)

# %% [markdown]
# **The p-value obtained is less than significance level of 0.05 and the ADF statistic is less than any of the critical values. Hence the zero ordered differenced series is stationary and d=0**

# %% [markdown]
# 
# 
# ---
# 
# 
# 
# ---
# 
# 

# %% [markdown]
# **Autocorrelation and Partial Autocorrelation Function:**  
# Autocorrelation and partial autocorrelation are plots that graphically summarize the impact of observations at prior timesteps on the observations we are trying to predict.
# 
# **ACF plot gives the q value and PACF gives the p value**  
# Look for tail of pattern in either ACF or PACF. If tail is crossing the blue region then it will give us potential p and q values.

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# %%
# if the series is not stationary then make sure to pass differenced series instead of original series
plot_acf(amount_series);
plot_pacf(amount_series);

# %% [markdown]
# 
# 1.   From ACF plot we can see that q value 5
# 2.   From PACF plot we can see that p value 5
# 
# 
# 

# %% [markdown]
# ### Split dataset to Train and Test

# %%
amount_series

# %%
amount_series.index = pd.to_datetime(amount_series.index, format='%m/%Y')
amount_series = amount_series.sort_index()

# %%
train_df = amount_series.loc[:'2022-05-01']
test_df = amount_series.loc['2022-06-01':]

# %%
train_df.shape

# %%
test_df.shape

# %% [markdown]
# ### Time series forecasting

# %%
!pip install prophet --q

# %%
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly


# %%
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# %% [markdown]
# #### **Exponential Smoothing:**  
# Exponential smoothing is a time series forecasting method for univariate data. There are three main types of exponential smoothing time series forecasting methods.
# A simple method that assumes no systematic structure, an extension that explicitly handles trends, and the most advanced approach that add support for seasonality.

# %% [markdown]
# ##### **Single Exponential Smoothing**
# 
# Single Exponential Smoothing, SES for short, also called Simple Exponential Smoothing, is a time series forecasting method for univariate data without a trend or seasonality.

# %%
single_exp = SimpleExpSmoothing(train_df).fit()
single_exp_train_pred = single_exp.fittedvalues
single_exp_test_pred = single_exp.forecast(9)

# %%
train_df['amount'].plot(style='--', color='gray', legend=True, label='train_df')
test_df['amount'].plot(style='--', color='r', legend=True, label='test_df')
single_exp_test_pred.plot(color='b', legend=True, label='Prediction')

# %%
print('Train RMSE:',mean_squared_error(train_df, single_exp_train_pred)**0.5)
print('Test RMSE:',mean_squared_error(test_df, single_exp_test_pred)**0.5)
print('Train MAPE:',mean_absolute_percentage_error(train_df, single_exp_train_pred))
print('Test MAPE:',mean_absolute_percentage_error(test_df, single_exp_test_pred))

# %% [markdown]
# ##### **Double Exponential Smoothing**
#    
# Double Exponential Smoothing is an extension to Exponential Smoothing that explicitly adds support for trends in the univariate time series

# %%
double_exp = ExponentialSmoothing(train_df, trend=None, initialization_method='heuristic', seasonal='add', damped_trend=False).fit()
double_exp_train_pred = double_exp.fittedvalues
double_exp_test_pred = double_exp.forecast(9)

# %%
train_df['amount'].plot(style='--', color='gray', legend=True, label='train_df')
test_df['amount'].plot(style='--', color='r', legend=True, label='test_df')
double_exp_test_pred.plot(color='b', legend=True, label='prediction')
plt.show()

# %%
print('Train RMSE:',mean_squared_error(train_df, double_exp_train_pred)**0.5)
print('Test RMSE:',mean_squared_error(test_df, double_exp_test_pred)**0.5)
print('Train MAPE:',mean_absolute_percentage_error(train_df, double_exp_train_pred))
print('Test MAPE:',mean_absolute_percentage_error(test_df, double_exp_test_pred))

# %% [markdown]
# ##### **Triple Exponential Smoothing**
# Triple Exponential Smoothing is an extension of Exponential Smoothing that explicitly adds support for seasonality to the univariate time series. Also known as Holt-Winters Exponential Smoothing.

# %%
hw_model = ExponentialSmoothing(train_df['amount'],
                          trend    ='add',
                          initialization_method='heuristic',
                          seasonal = "add",
                          damped_trend=True).fit()
hw_train_pred =  hw_model.fittedvalues
hw_test_pred =  hw_model.forecast(9)

# %%
train_df['amount'].plot(style='--', color='gray', legend=True, label='train_df')
test_df['amount'].plot(style='--', color='r', legend=True, label='test_df')
hw_test_pred.plot(color='b', legend=True, label='prediction')
plt.show()

# %%
print('Train RMSE:',mean_squared_error(train_df, hw_train_pred)**0.5)
print('Test RMSE:',mean_squared_error(test_df, hw_test_pred)**0.5)
print('Train MAPE:',mean_absolute_percentage_error(train_df, hw_train_pred))
print('Test MAPE:',mean_absolute_percentage_error(test_df, hw_test_pred))

# %% [markdown]
# #### **ARIMA**
# 
# A popular and widely used statistical method for time series forecasting is the ARIMA model. ARIMA is an acronym that stands for AutoRegressive Integrated Moving Average. The parameters of the ARIMA model are defined as follows:
# 
# >**p** : The number of lag observations included in the model, also called the lag order.   
# >**d** : The number of times that the raw observations are differenced, also called the degree of differencing.   
# >**q** : The size of the moving average window, also called the order of moving average.

# %%
ar = ARIMA(train_df, order=(5,0,5)).fit()
ar_train_pred = ar.fittedvalues
ar_test_pred = ar.forecast(9)

# %%
train_df['amount'].plot(style='--', color='gray', legend=True, label='train_df')
test_df['amount'].plot(style='--', color='r', legend=True, label='test_df')
ar_test_pred.plot(color='b', legend=True, label='prediction')
plt.show()

# %%
print('Train RMSE:',mean_squared_error(train_df, ar_train_pred)**0.5)
print('Test RMSE:',mean_squared_error(test_df, ar_test_pred)**0.5)
print('Train MAPE:',mean_absolute_percentage_error(train_df, ar_train_pred))
print('Test MAPE:',mean_absolute_percentage_error(test_df, ar_test_pred))

# %%
from tqdm import tqdm_notebook
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product

# %%
def optimize_SARIMA(parameters_list, d, D, s, exog):
    """
        Return dataframe with parameters, corresponding AIC and SSE

        parameters_list - list with (p, q, P, Q) tuples
        d - integration order
        D - seasonal integration order
        s - length of season
        exog - the exogenous variable
    """

    results = []

    for param in tqdm_notebook(parameters_list):
        try:
            model = SARIMAX(exog, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue

        aic = model.aic
        results.append([param, aic])

    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)x(P,Q)', 'AIC']
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)

    return result_df

# %%
p = range(0, 5, 1)
d = 0
q = range(0, 5, 1)
P = range(0, 5, 1)
D = 0
Q = range(0, 5, 1)
s = parameters = product(p, q, P, Q)
parameters_list = list(parameters)
print(len(parameters_list))

# %%
result_df = optimize_SARIMA(parameters_list, 0, 0, 5, train_df)


# %%
result_df

# %%
sar =  SARIMAX(train_df, order=(2, 0, 1), seasonal_order=(1, 0, 3, 12)).fit(dis=-1)
sar_train_pred = sar.fittedvalues
sar_test_pred = sar.forecast(9)

# %%
train_df['amount'].plot(style='--', color='gray', legend=True, label='train_df')
test_df['amount'].plot(style='--', color='r', legend=True, label='test_df')
sar_test_pred.plot(color='b', legend=True, label='prediction')
plt.show()

# %%
print('Train RMSE:',mean_squared_error(train_df, sar_train_pred)**0.5)
print('Test RMSE:',mean_squared_error(test_df, sar_test_pred)**0.5)
print('Train MAPE:',mean_absolute_percentage_error(train_df, sar_train_pred))
print('Test MAPE:',mean_absolute_percentage_error(test_df, sar_test_pred))

# %%
sar.plot_diagnostics(figsize=(15,12));

# %% [markdown]
# #### **Prophet**
# 
# The Prophet library is an open-source library designed for making forecasts for univariate time series datasets. It is easy to use and designed to automatically find a good set of hyperparameters for the model in an effort to make skillful forecasts for data with trends and seasonal structure by default.
# 
# The DataFrame must have a specific format. The first column must have the name ‘ds‘ and contain the date-times. The second column must have the name ‘y‘ and contain the observations.

# %%
# converting the original dataframe into required format by prophet
prophet_df = amount_series.copy()
prophet_df.reset_index(inplace=True)
prophet_df.columns=['ds','y']
prophet_df.head()

# %%
# train test split data
prophet_train_df = prophet_df.iloc[:-9]
prophet_test_df = prophet_df.iloc[-9:]

# %%
prophet_model = Prophet()
prophet_model.fit(prophet_train_df) #fit training data to model

# %%
future = prophet_model.make_future_dataframe(periods=9, freq='M')
prophet_predictions = prophet_model.predict(future)

# %%
prophet_predictions.tail()

# %%
plot_plotly(prophet_model, prophet_predictions)

# %%
prophet_train_df['prophet_train_pred'] = prophet_predictions.iloc[:-9]['yhat']
prophet_test_df['prophet_test_pred'] = prophet_predictions.iloc[-9:]['yhat']

# %%
print('Train RMSE:',mean_squared_error(prophet_train_df['y'], prophet_train_df['prophet_train_pred'])**0.5)
print('Test RMSE:',mean_squared_error(prophet_test_df['y'], prophet_test_df['prophet_test_pred'])**0.5)
print('Train MAPE:',mean_absolute_percentage_error(prophet_train_df['y'], prophet_train_df['prophet_train_pred']))
print('Test MAPE:',mean_absolute_percentage_error(prophet_test_df['y'], prophet_test_df['prophet_test_pred']))

# %% [markdown]
# #### **MODELS COMPARISION**

# %%
comparision_df = pd.DataFrame(data=[['Single exp smoothing', 70722 , 0.2360],
                           ['double exp smoothing', 76647,  0.1790],
                          ['Triple exp smoothing', 108760, 0.3066],
                          ['ARIMA(5,0,5)', 99772, 0.3244],
                          ['SARIMAX', 78710, 0.1995],
                          ['prophet', 182805, 0.5567]], columns=['Model','RMSE','MAPE'])

comparision_df.set_index('Model', inplace=True)

# %%
comparision_df.sort_values(by='MAPE')

# %%
import pickle

with open('predict_income.pkl', 'wb') as f:
  pickle.dump(double_exp, f)

print("Done")

# %% [markdown]
# ## ATTENDANCE

# %%
attendance_series = attendance_series.sort_index()
index =  pd.to_datetime(attendance_series.index)
attendance_series.index = index

# %%
fig, ax = plt.subplots(figsize=(8, 5))
attendance_series.plot(kind='line', ax=ax, grid=True)
ax.set_title("Time series of Monthly Attendance")
plt.show()

# %% [markdown]
# **Time Series Decomposition:**   

# %%
decompose_add = seasonal_decompose(attendance_series.values, period=12)
decompose_add.plot()
plt.show()

# %% [markdown]
# **ADFuller Test:**

# %%
adf_test(attendance_series.values)

# %% [markdown]
# **Autocorrelation and Partial Autocorrelation Function:**

# %%
plot_acf(attendance_series, lags=12);
plot_pacf(attendance_series, lags=12);

# %% [markdown]
# 

# %% [markdown]
# ### Split dataset to Train and Test

# %%
attendance_series.index = pd.to_datetime(attendance_series.index, format='%m/%Y')
attendance_series = attendance_series.sort_index()

# %%
train_df = attendance_series.loc[:'2022-05-01']
test_df = attendance_series.loc['2022-06-01':]

# %%
test_df.shape

# %% [markdown]
# ### Time series forecasting

# %% [markdown]
# #### **Exponential Smoothing:**

# %% [markdown]
# ##### **Single Exponential Smoothing**

# %%
single_exp = SimpleExpSmoothing(train_df).fit()
single_exp_train_pred = single_exp.fittedvalues
single_exp_test_pred = single_exp.forecast(9)

# %%
train_df['total'].plot(style='--', color='gray', legend=True, label='train_df')
test_df['total'].plot(style='--', color='r', legend=True, label='test_df')
single_exp_test_pred.plot(color='b', legend=True, label='Prediction')

# %%
print('Train RMSE:',mean_squared_error(train_df, single_exp_train_pred)**0.5)
print('Test RMSE:',mean_squared_error(test_df, single_exp_test_pred)**0.5)
print('Train MAPE:',mean_absolute_percentage_error(train_df, single_exp_train_pred))
print('Test MAPE:',mean_absolute_percentage_error(test_df, single_exp_test_pred))

# %% [markdown]
# ##### **Double Exponential Smoothing**

# %%
double_exp = ExponentialSmoothing(train_df, trend=None, initialization_method='heuristic', seasonal='add', damped_trend=False).fit()
double_exp_train_pred = double_exp.fittedvalues
double_exp_test_pred = double_exp.forecast(9)

# %%
train_df['total'].plot(style='--', color='gray', legend=True, label='train_df')
test_df['total'].plot(style='--', color='r', legend=True, label='test_df')
double_exp_test_pred.plot(color='b', legend=True, label='prediction')
plt.show()

# %%
print('Train RMSE:',mean_squared_error(train_df, double_exp_train_pred)**0.5)
print('Test RMSE:',mean_squared_error(test_df, double_exp_test_pred)**0.5)
print('Train MAPE:',mean_absolute_percentage_error(train_df, double_exp_train_pred))
print('Test MAPE:',mean_absolute_percentage_error(test_df, double_exp_test_pred))

# %% [markdown]
# ##### **Triple Exponential Smoothing**

# %%
hw_model = ExponentialSmoothing(train_df['total'],
                          trend    ='add',
                          initialization_method='heuristic',
                          seasonal = "add",
                          damped_trend=True).fit()
hw_train_pred =  hw_model.fittedvalues
hw_test_pred =  hw_model.forecast(9)

# %%
train_df['total'].plot(style='--', color='gray', legend=True, label='train_df')
test_df['total'].plot(style='--', color='r', legend=True, label='test_df')
hw_test_pred.plot(color='b', legend=True, label='prediction')
plt.show()

# %%
print('Train RMSE:',mean_squared_error(train_df, hw_train_pred)**0.5)
print('Test RMSE:',mean_squared_error(test_df, hw_test_pred)**0.5)
print('Train MAPE:',mean_absolute_percentage_error(train_df, hw_train_pred))
print('Test MAPE:',mean_absolute_percentage_error(test_df, hw_test_pred))

# %% [markdown]
# #### **ARIMA**

# %% [markdown]
# 

# %%
ar = ARIMA(train_df, order=(4,0,4)).fit()
ar_train_pred = ar.fittedvalues
ar_test_pred = ar.forecast(9)

# %%
train_df['total'].plot(style='--', color='gray', legend=True, label='train_df')
test_df['total'].plot(style='--', color='r', legend=True, label='test_df')
ar_test_pred.plot(color='b', legend=True, label='prediction')
plt.show()

# %%
print('Train RMSE:',mean_squared_error(train_df, ar_train_pred)**0.5)
print('Test RMSE:',mean_squared_error(test_df, ar_test_pred)**0.5)
print('Train MAPE:',mean_absolute_percentage_error(train_df, ar_train_pred))
print('Test MAPE:',mean_absolute_percentage_error(test_df, ar_test_pred))

# %%
p = range(0, 4, 1)
d = 0
q = range(0, 4, 1)
P = range(0, 4, 1)
D = 0
Q = range(0, 4, 1)
s = parameters = product(p, q, P, Q)
parameters_list = list(parameters)
print(len(parameters_list))

# %%
result_df = optimize_SARIMA(parameters_list, 0, 0, 5, train_df)


# %%
result_df

# %%
train_df

# %%
sar =  SARIMAX(train_df, order=(0, 0, 1), seasonal_order=(3, 0, 2, 12)).fit(dis=-1)
sar_train_pred = sar.fittedvalues
sar_test_pred = sar.forecast(9)

# %%
train_df['total'].plot(style='--', color='gray', legend=True, label='train_df')
test_df['total'].plot(style='--', color='r', legend=True, label='test_df')
sar_test_pred.plot(color='b', legend=True, label='prediction')
plt.show()

# %%
sar.plot_diagnostics(figsize=(15,12));

# %% [markdown]
# #### **Prophet**

# %%
# converting the original dataframe into required format by prophet
prophet_df = attendance_series.copy()
prophet_df.reset_index(inplace=True)
prophet_df.columns=['ds','y']
prophet_df.head()

# %%
# train test split data
prophet_train_df = prophet_df.iloc[:-9]
prophet_test_df = prophet_df.iloc[-9:]

# %%
prophet_model = Prophet()
prophet_model.fit(prophet_train_df) #fit training data to model

# %%
future = prophet_model.make_future_dataframe(periods=9, freq='M')
prophet_predictions = prophet_model.predict(future)

# %%
prophet_predictions.tail()

# %%
plot_plotly(prophet_model, prophet_predictions)

# %%
prophet_train_df['prophet_train_pred'] = prophet_predictions.iloc[:-9]['yhat']
prophet_test_df['prophet_test_pred'] = prophet_predictions.iloc[-9:]['yhat']

# %%
print('Train RMSE:',mean_squared_error(prophet_train_df['y'], prophet_train_df['prophet_train_pred'])**0.5)
print('Test RMSE:',mean_squared_error(prophet_test_df['y'], prophet_test_df['prophet_test_pred'])**0.5)
print('Train MAPE:',mean_absolute_percentage_error(prophet_train_df['y'], prophet_train_df['prophet_train_pred']))
print('Test MAPE:',mean_absolute_percentage_error(prophet_test_df['y'], prophet_test_df['prophet_test_pred']))

# %%
import pickle

with open('predict_attendance.pkl', 'wb') as f:
  pickle.dump(ar, f)

print("Done")

# %%
comparision_df = pd.DataFrame(data=[['Single exp smoothing', 53.4436 , 0.1640],
                           ['double exp smoothing', 52.3209,  0.1608],
                          ['Triple exp smoothing', 51.4796, 0.1598],
                          ['ARIMA(4,0,4)', 61.0922, 0.1815],
                          ['prophet', 118.1606, 0.3639]], columns=['Model','RMSE','MAPE'])

comparision_df.set_index('Model', inplace=True)

# %%
comparision_df.sort_values(by='MAPE')


