# 📈 Stock Price Prediction using LSTM

This is a Flask-based web application developed as part of a diploma thesis. It predicts future stock prices using a LSTM (Long Short-Term Memory) neural network. The app uses the `yfinance` library to fetch historical stock data and presents predictions on a user-friendly web interface built with Flask views.

---

## 🚀 Features
- Fetch historical stock data using stock ticker (e.g., AAPL, TSLA)
- Predict future **Open**, **High**, **Low**, and **Close** prices
- Display Close price trend chart
- Tabular view of future OHLC predictions
- Lightweight and easy to run locally

---

## 🛠️ Technologies Used
- **Python**
- **Flask** – Web server and UI rendering
- **LSTM (Keras/TensorFlow)** – Time-series prediction model
- **yfinance** – For fetching historical stock data
- **Pandas, NumPy** – Data processing
- **Matplotlib** – Charting/visualization

---

## 📁 Project Structure
# Step 1: Create Virtual Environment
python -m venv venv

# Step 2: Activate the Environment
venv\Scripts\activate

# Step 3: Install Required Packages
pip install -r requirements.txt

# Step 4: Run the Application
python app.py

👤 Author

- [Festim Krasniqi](https://github.com/FestimKrasniq)


