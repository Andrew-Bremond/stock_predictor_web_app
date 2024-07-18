import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")
stocks = ("SPY", "NVDA", "AMD", "TSLA", "AAPL", "GOOG", "MSFT", "GME")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_days = st.slider("Days of prediction:", 1, 1000)
period = n_days

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
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

# Predict with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
future['ds'] = future['ds'].astype('datetime64[ns]') # Ensure the future dataframe is correctly formatted
prediction = m.predict(future)

# Show and plot prediction
st.subheader('prediction data')
st.write(prediction.tail())
    
st.write(f'prediction plot for {n_days} days')
fig1 = plot_plotly(m, prediction)
st.plotly_chart(fig1)

st.write("prediction components")
fig2 = m.plot_components(prediction)
st.write(fig2)