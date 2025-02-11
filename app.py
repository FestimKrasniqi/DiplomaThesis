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
        stock_symbol = request.form.get('stock_symbol')
        days_to_predict = int(request.form.get('days_to_predict', 1))
        display_format = request.form.get('display_format', 'chart')
        look_back = 60  # Number of previous days to use for prediction

        # Fetch historical stock data
        stock_data = yf.download(stock_symbol, period='2y')
        if stock_data.empty:
            return render_template('index.html', error='Invalid stock symbol or no data available')

        # Use Open, High, Low, Close prices
        selected_features = ['Open', 'High', 'Low', 'Close']
        stock_prices = stock_data[selected_features].values

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(stock_prices)

        X_data, y_data = prepare_data(scaled_data, look_back)
        X_data = X_data.reshape((X_data.shape[0], look_back, 4))  # 4 features (OHLC)

        # Build and train the LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(look_back, 4)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(4)  # Predict Open, High, Low, Close
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_data, y_data, batch_size=32, epochs=10, verbose=1)

        # Predict future OHLC prices
        recent_data = scaled_data[-look_back:]
        predicted_prices = []
        prediction_dates = []
        last_date = stock_data.index[-1]
        predictions_made = 0

        while predictions_made < days_to_predict:
            last_date += pd.Timedelta(days=1)
            if last_date.weekday() in [5, 6]:  # Skip weekends
                continue

            input_data = recent_data.reshape((1, look_back, 4))
            prediction = model.predict(input_data)[0]
            predicted_prices.append(prediction)
            prediction_dates.append(last_date)
            recent_data = np.vstack([recent_data[1:], prediction])
            predictions_made += 1

        predicted_prices = scaler.inverse_transform(predicted_prices)

        # Plot only future predicted Close prices
        plot_url = None
        if display_format == 'chart':
            plt.figure(figsize=(12, 6))
            plt.plot(prediction_dates, predicted_prices[:, 3], label='Predicted Close Prices', color='orange', marker='o', linestyle='--')
            plt.title(f'{stock_symbol} - Future Predicted Close Prices')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

        return render_template(
            'index.html',
            plot_url=f'data:image/png;base64,{plot_url}' if display_format == 'chart' else None,
            predicted_prices=predicted_prices.tolist(),
            prediction_dates=[date.strftime('%Y-%m-%d') for date in prediction_dates],
            stock_symbol=stock_symbol,
            zip=zip,
            display_format=display_format
        )

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=8080)
