from flask import Flask, render_template
import requests
from datetime import datetime
from fbprophet import Prophet
import pandas as pd

app = Flask(__name__)

# Replace 'YOUR_ALPHA_VANTAGE_API_KEY' with your actual Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = 'KSZ1KD4VHOBEHSXC'


# Function to fetch the historical stock prices of Apple (AAPL) from the Alpha Vantage API
def get_aapl_stock_prices():
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey={ALPHA_VANTAGE_API_KEY}'
    response = requests.get(url)
    data = response.json()['Time Series (Daily)']
    timestamps = sorted(data.keys())  # timestamps sorted in ascending order
    formatted_dates = [datetime.strptime(ts, '%Y-%m-%d').strftime('%Y-%m-%d') for ts in timestamps]
    prices = [float(data[ts]['4. close']) for ts in timestamps]  # closing prices
    return formatted_dates, prices


# Function to prepare the dataset for Prophet model
def prepare_dataset():
    timestamps, prices = get_aapl_stock_prices()
    df = pd.DataFrame({'ds': timestamps, 'y': prices})
    return df


# Function to train the Prophet model
def train_prophet_model(df):
    model = Prophet()
    model.fit(df)
    return model


# Function to make predictions for future stock prices
def predict_future_price(model):
    future_dates = model.make_future_dataframe(periods=7)  # Predicting for the next 7 days
    forecast = model.predict(future_dates)
    future_price = forecast.iloc[-1]['yhat']  # Predicted price for the last date
    return future_price


@app.route('/')
def index():
    # Prepare dataset and train Prophet model
    df = prepare_dataset()
    model = train_prophet_model(df)

    # Predict future price
    future_price = predict_future_price(model)

    return render_template('index.html', future_price=future_price)


if __name__ == '__main__':
    app.run(debug=True)
