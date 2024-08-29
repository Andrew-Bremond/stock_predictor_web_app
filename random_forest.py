import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from plotly import graph_objs as go

def random_forest_predictions(data):
    st.subheader('Random Forest Model Predictions')

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

    train = data.iloc[:-100]
    test = data.iloc[-100:]

    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1, n_jobs=-1 )

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

    st.write(f'Precision: {precision}')
    st.write(predictions.tail())

    def plot_rf_predictions(predictions):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions['Target'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions['Predictions'], mode='lines', name='Predicted'))
        fig.layout.update(title_text='Random Forest Predictions', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_rf_predictions(predictions)

    st.markdown("""
    **Random Forest Model:** Random Forest is a machine learning model that uses an ensemble of decision trees to make predictions. 
                It is particularly effective for classification tasks and can handle a large number of input features. 
                In this application, it predicts whether the stock price will go up or down the next day based on historical data and 
                various calculated features.
    """)
