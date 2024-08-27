import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from plotly import graph_objs as go
from datetime import timedelta

def xgboost_predictions(data):
    st.subheader('XGBoost Model Predictions')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
   
    # Add more features
    data['DayOfWeek'] = data.index.dayofweek
    data['Month'] = data.index.month
    data['Year'] = data.index.year
    data['HighLowDiff'] = data['High'] - data['Low']
    data['CloseOpenDiff'] = data['Close'] - data['Open']
    
    # Lag features: use previous days' Close prices as features
    data['Close_Lag1'] = data['Close'].shift(1)
    data['Close_Lag2'] = data['Close'].shift(2)
    
    data.dropna(inplace=True)

    features = ['Open', 'Volume', 'DayOfWeek', 'Month', 'Year', 'HighLowDiff', 'CloseOpenDiff', 'Close_Lag1', 'Close_Lag2']
    target = 'Close'
    
    # Split the data
    train_data = data.iloc[:-30]  # Use all but last 30 days for training
    test_data = data.iloc[-30:]   # Use last 30 days for testing

    scaler = StandardScaler()
    train_data[features] = scaler.fit_transform(train_data[features])
    test_data[features] = scaler.transform(test_data[features])
    
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
    model.fit(train_data[features], train_data[target])
    
    # Make predictions on test data
    test_predictions = model.predict(test_data[features])
    
    # Calculate MAPE and RMSE
    mape = mean_absolute_percentage_error(test_data[target], test_predictions)
    rmse = np.sqrt(mean_squared_error(test_data[target], test_predictions))
    
    st.write(f'Mean Absolute Percentage Error (MAPE): {mape:.2%}')
    st.write(f'Root Mean Square Error (RMSE): {rmse:.2f}')
    
    # Plot predictions vs actual
    def plot_xgb_predictions(actual, predictions):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual.index, y=actual, mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=actual.index, y=predictions, mode='lines', name='Predicted'))
        fig.layout.update(
            title_text='XGBoost Predictions vs Actual Values (Test Set)',
            xaxis_title='Date',
            yaxis_title='Stock Closing Price',
            xaxis_rangeslider_visible=True
        )
        st.plotly_chart(fig)
    
    plot_xgb_predictions(test_data[target], test_predictions)

    st.markdown("""
    **XGBoost Model:** XGBoost is an advanced gradient boosting machine learning algorithm that is quick and effective for predictive purpose. 
                In this code, XGBoost is leveraged to predict stock closing prices by training the model using historical data along with the 
                differences between high and low, day of the week, lagged closing prices and many other variables. The results obtained from the 
                model are compared to real world data, helping to determine the efficiency of the model via MAPE and RMSE. 
                This application shows how XGBoost can be applied to explore and predict time series data in this case of financial nature.
    """)