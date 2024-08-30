import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from plotly import graph_objs as go
from datetime import timedelta

# import seaborn as sns
# import matplotlib.pyplot as plt

# Streamline Data Processing
@st.cache_data
def preprocess_data(data):
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
    
    return data

def xgboost_predictions(data):
    st.subheader('XGBoost Model Predictions')
    
    # Preprocess data
    data = preprocess_data(data)
    
    features = ['Open', 'Volume', 'DayOfWeek', 'Month', 'Year', 'HighLowDiff', 'CloseOpenDiff', 'Close_Lag1', 'Close_Lag2']
    target = 'Close'
    
    # Split the data
    train_data = data.iloc[:-30]  # Use all but last 30 days for training
    test_data = data.iloc[-30:]   # Use last 30 days for testing

    scaler = StandardScaler()
    train_data[features] = scaler.fit_transform(train_data[features])
    test_data[features] = scaler.transform(test_data[features])
    
    model = xgb.XGBRegressor(tree_method='hist', n_estimators=100, learning_rate=0.05, max_depth=5, n_jobs=-1)
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


# checking for overlap (data leaks) or overfitting
# def xgboost_predictions(data):
#     st.subheader('XGBoost Model Predictions')
    
#     # Preprocess data
#     data = preprocess_data(data)
    
#     features = ['Open', 'Volume', 'DayOfWeek', 'Month', 'Year', 'HighLowDiff', 'CloseOpenDiff', 'Close_Lag1', 'Close_Lag2']
#     target = 'Close'
    
#     # Split the data
#     train_data = data.iloc[:-30]  # Use all but last 30 days for training
#     test_data = data.iloc[-30:]   # Use last 30 days for testing

#     # Check for overlap
#     train_indices = set(train_data.index)
#     test_indices = set(test_data.index)
    
#     overlap = train_indices.intersection(test_indices)
#     if overlap:
#         st.warning(f"Warning: Overlapping indices found: {overlap}")
#     else:
#         st.success("No overlap between training and testing indices.")

#     # Check data distribution
#     st.subheader("Data Distribution Comparison")
    
#     # Dynamically set the number of rows and columns based on the number of features
#     num_features = len(features)
#     num_cols = 4  # Number of columns you want for plotting
#     num_rows = (num_features + num_cols - 1) // num_cols  # Calculate rows needed
    
#     fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, num_rows * 5))
#     axes = axes.flatten()  # Flatten the axes array for easy indexing
    
#     for i, feature in enumerate(features):
#         ax = axes[i]
#         sns.histplot(train_data[feature], color='blue', label='Train', kde=True, ax=ax, stat='density', alpha=0.5)
#         sns.histplot(test_data[feature], color='orange', label='Test', kde=True, ax=ax, stat='density', alpha=0.5)
#         ax.set_title(f'Distribution of {feature}')
#         ax.legend()

#     # Hide any unused subplots
#     for j in range(num_features, len(axes)):
#         fig.delaxes(axes[j])
    
#     plt.tight_layout()
#     st.pyplot(fig)

#     scaler = StandardScaler()
#     train_data[features] = scaler.fit_transform(train_data[features])
#     test_data[features] = scaler.transform(test_data[features])
    
#     model = xgb.XGBRegressor(tree_method='hist', n_estimators=100, learning_rate=0.05, max_depth=5, n_jobs=-1)
#     model.fit(train_data[features], train_data[target])
    
#     # Make predictions on training data
#     train_predictions = model.predict(train_data[features])
#     test_predictions = model.predict(test_data[features])
    
#     # Calculate MAPE and RMSE for training data
#     train_mape = mean_absolute_percentage_error(train_data[target], train_predictions)
#     train_rmse = np.sqrt(mean_squared_error(train_data[target], train_predictions))
    
#     # Calculate MAPE and RMSE for testing data
#     test_mape = mean_absolute_percentage_error(test_data[target], test_predictions)
#     test_rmse = np.sqrt(mean_squared_error(test_data[target], test_predictions))
    
#     # Display metrics
#     st.write(f'Training Mean Absolute Percentage Error (MAPE): {train_mape:.2%}')
#     st.write(f'Training Root Mean Square Error (RMSE): {train_rmse:.2f}')
#     st.write(f'Testing Mean Absolute Percentage Error (MAPE): {test_mape:.2%}')
#     st.write(f'Testing Root Mean Square Error (RMSE): {test_rmse:.2f}')
    
#     # Plot predictions vs actual
#     def plot_xgb_predictions(actual, predictions, title):
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=actual.index, y=actual, mode='lines', name='Actual'))
#         fig.add_trace(go.Scatter(x=actual.index, y=predictions, mode='lines', name='Predicted'))
#         fig.layout.update(
#             title_text=title,
#             xaxis_title='Date',
#             yaxis_title='Stock Closing Price',
#             xaxis_rangeslider_visible=True
#         )
#         st.plotly_chart(fig)
    
#     plot_xgb_predictions(test_data[target], test_predictions, 'XGBoost Predictions vs Actual Values (Test Set)')
#     plot_xgb_predictions(train_data[target], train_predictions, 'XGBoost Predictions vs Actual Values (Training Set)')
