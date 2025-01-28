from flask import Flask, request, render_template
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Function to prepare data for LSTM
def prepare_data(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_stock():
    try:
        # Get inputs from the form
        stock_symbol = request.form.get('stock_symbol')
        days_to_predict = int(request.form.get('days_to_predict', 1))
        display_format = request.form.get('display_format', 'chart')  # Get format (chart/table)
        look_back = 60

        # Fetch historical stock data
        stock_data = yf.download(stock_symbol, period='2y')
        if stock_data.empty:
            return render_template('index.html', error='Invalid stock symbol or no data available')

        close_prices = stock_data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        X_data, y_data = prepare_data(scaled_data, look_back)
        X_data = X_data.reshape((X_data.shape[0], X_data.shape[1], 1))

        # Build and train the LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_data, y_data, batch_size=32, epochs=10, verbose=1)

        # Predict future prices
        recent_data = scaled_data[-look_back:]
        predicted_prices = []
        prediction_dates = []
        last_date = stock_data.index[-1]
        predictions_made = 0

        while predictions_made < days_to_predict:
            last_date += pd.Timedelta(days=1)
            if last_date.weekday() in [5, 6]:  # Skip weekends
                continue

            input_data = recent_data.reshape((1, look_back, 1))
            prediction = model.predict(input_data)
            predicted_prices.append(prediction[0, 0])
            prediction_dates.append(last_date)
            recent_data = np.append(recent_data[1:], prediction)
            predictions_made += 1

        predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1)).tolist()

        # Extract historical data
        historical_prices = scaler.inverse_transform(scaled_data[-look_back:]).flatten().tolist()
        historical_dates = stock_data.index[-look_back:].strftime('%Y-%m-%d').tolist()

        # If format is chart, generate the plot
        plot_url = None
        if display_format == 'chart':
            plt.figure(figsize=(14, 10))
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            historical_prices_plot = scaler.inverse_transform(scaled_data[-look_back:]).flatten()
            historical_dates_plot = stock_data.index[-look_back:]
            ax1.plot(historical_dates_plot, historical_prices_plot, label='Last 60 Days Prices', color='blue')
            ax1.set_title(f'{stock_symbol} - Last 60 Days Prices')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, linestyle='--', alpha=0.6)
            ax1.legend()

            ax2.plot(prediction_dates, predicted_prices, label='Predicted Prices', color='orange', marker='o', linestyle='--')
            ax2.set_title(f'{stock_symbol} - Predicted Prices ({days_to_predict} Days)')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Price')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, linestyle='--', alpha=0.6)
            ax2.legend()

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

        return render_template(
            'index.html',
            plot_url=f'data:image/png;base64,{plot_url}' if display_format in ['chart', 'both'] else None,
            historical_prices=historical_prices,  # Historical data
            historical_dates=historical_dates,
            predicted_prices=predicted_prices,  # Predicted data
            prediction_dates=[date.strftime('%Y-%m-%d') for date in prediction_dates],
            stock_symbol=stock_symbol,
            zip=zip,
            display_format=display_format
        )

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True,port=8080)
