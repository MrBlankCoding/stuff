from flask import Flask, render_template
import requests
from datetime import datetime
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

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


# Function to prepare the dataset for ARIMA model
def prepare_dataset():
    timestamps, prices = get_aapl_stock_prices()
    df = pd.DataFrame({'ds': timestamps, 'y': prices})
    df['ds'] = pd.to_datetime(df['ds'])  # Convert 'ds' column to datetime
    df.set_index('ds', inplace=True)     # Set 'ds' column as the index

    # Resample to fill missing data and ensure a regular frequency
    df = df.resample('D').ffill()  # Resample to daily frequency and forward fill missing values

    return df


# Function to train the ARIMA model
def train_arima_model(df):
    model = ARIMA(df, order=(5, 1, 0))  # ARIMA(5,1,0) for example, you can tune these parameters
    return model.fit()


def predict_future_price(model, steps=7):
    try:
        forecast = model.forecast(steps=steps)
        return forecast.iloc[0]  # Predicted price for the last date
    except KeyError as e:
        print("Error occurred:", e)
        return None



@app.route('/')
def index():
    # Prepare dataset and train ARIMA model
    df = prepare_dataset()
    model = train_arima_model(df)

    # Predict future price
    future_price = predict_future_price(model)

    return render_template('index.html', future_price=future_price)


if __name__ == '__main__':
    app.run(debug=True, port=8080)
