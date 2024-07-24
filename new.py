import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction Web App")
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

# Prepare data for RandomForest
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data['Tomorrow'] = data['Close'].shift(-1)
data['Target'] = (data['Tomorrow'] > data['Close']).astype(int)
data = data.dropna()

horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = data.rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    data[ratio_column] = data["Close"] / rolling_averages["Close"]
    trend_column = f"Trend_{horizon}"
    data[trend_column] = data.shift(1).rolling(horizon).sum()["Target"]
    new_predictors += [ratio_column, trend_column]

data = data.dropna(subset=data.columns[data.columns != "Tomorrow"])

predictors = ["Close", "Volume", "Open", "High", "Low"] + new_predictors

# Train RandomForest model
train = data.iloc[:-100]
test = data.iloc[-100:]

model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

predictions = predict(train, test, predictors, model)

precision = precision_score(predictions["Target"], predictions["Predictions"])

st.subheader('Random Forest Predictions')
st.write(f'Precision: {precision}')
st.write(predictions.tail())

# Plot Random Forest predictions
def plot_rf_predictions(predictions):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=predictions.index, y=predictions['Target'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=predictions.index, y=predictions['Predictions'], mode='lines', name='Predicted'))
    fig.layout.update(title_text='Random Forest Predictions', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_rf_predictions(predictions)

# Predict with Prophet
df_train = data[['Close']].reset_index().rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
prediction = m.predict(future)

# Show and plot Prophet prediction
st.subheader('Prophet Prediction')
st.write(prediction.tail())

st.write(f'Prophet prediction plot for {n_days} days')
fig1 = plot_plotly(m, prediction)
st.plotly_chart(fig1)

st.write("Prediction components")
fig2 = m.plot_components(prediction)
st.write(fig2)
