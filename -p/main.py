from flask import Flask, render_template, request
import requests
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import os

app = Flask(__name__)

# Replace 'YOUR_ALPHA_VANTAGE_API_KEY' with your actual Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = 'KSZ1KD4VHOBEHSXC'
DATA_DIR = 'stock_data'

# Function to fetch the historical stock prices from the Alpha Vantage API and save to CSV
def fetch_and_save_stock_prices(symbol):
    data_file = os.path.join(DATA_DIR, f"{symbol.upper()}_stock_data.csv")
    if not os.path.exists(data_file):
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol.upper()}&apikey={ALPHA_VANTAGE_API_KEY}&outputsize=full'
        response = requests.get(url)
        data = response.json()['Time Series (Daily)']
        timestamps = sorted(data.keys(), reverse=True)  # timestamps sorted in descending order
        stock_data = [(ts, data[ts]['4. close']) for ts in timestamps]
        df = pd.DataFrame(stock_data, columns=['date', 'close'])
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(data_file, index=False)
    return data_file

# Function to load the stock prices from the CSV file
def load_stock_prices(data_file):
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df['close'].sort_index()

# Function to prepare the dataset for ARIMA model
def prepare_dataset(prices):
    df = pd.DataFrame({'y': prices})
    df.index = pd.to_datetime(df.index)  # Convert index to datetime
    df = df.asfreq('D')  # Set frequency to daily
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

@app.route('/', methods=['GET', 'POST'])
def index():
    stock_symbol = request.form.get('stock_symbol', 'AAPL')
    data_file = fetch_and_save_stock_prices(stock_symbol)
    stock_prices = load_stock_prices(data_file)

    # Prepare dataset and train ARIMA model
    df = prepare_dataset(stock_prices)
    model = train_arima_model(df)

    # Predict future price
    future_price = predict_future_price(model)

    # Convert stock prices to a list of tuples (date, price)
    stock_data = [(date.strftime('%Y-%m-%d'), price) for date, price in stock_prices.items()]

    return render_template('index.html', stock_data=stock_data, future_price=future_price, stock_symbol=stock_symbol.upper())

if __name__ == '__main__':
    app.run(debug=True, port=8080)