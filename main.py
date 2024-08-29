import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go

from prophet_model import prophet_predictions
from random_forest import random_forest_predictions
from xgboost_model import xgboost_predictions

START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction Web App")
stocks = ("AMD", "GOOG", "GME", "MSFT", "NVDA", "SPY", "TSLA")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

# Load data function
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start, end)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock, START, TODAY)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

model_choice = st.selectbox("Select prediction model", ["Prophet", "Random Forest", "XGBoost"])

if model_choice == "Prophet":
    n_days = st.slider("Days of prediction:", 1, 1000)
    prophet_predictions(data, n_days)

elif model_choice == "Random Forest":
    random_forest_predictions(data)

elif model_choice == "XGBoost":
    xgboost_predictions(data)