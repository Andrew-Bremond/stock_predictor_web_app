import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly

def prophet_predictions(data, n_days):
    st.subheader('Prophet Model Predictions')
    df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=n_days)
    prediction = m.predict(future)

    st.write(prediction.tail())

    st.write(f'Prophet prediction plot for {n_days} days')
    fig1 = plot_plotly(m, prediction)
    st.plotly_chart(fig1)

    # st.write("Prediction components")
    # fig2 = m.plot_components(prediction)
    # st.write(fig2)

    st.markdown("""
    **Prophet Model:** Prophet is a time series forecasting model developed by Facebook. It is designed to handle time series 
                data with daily observations and can capture trends, seasonality, and holidays. It is particularly useful for 
                making predictions over different periods, such as weeks, months, or years.
    """)
