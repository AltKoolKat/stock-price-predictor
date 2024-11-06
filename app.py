import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# App title and description
st.title("📈 Stock Price Predictor App")
st.markdown("""
This app uses a pre-trained model to predict stock prices. 
It also displays the stock's historical moving averages.
""")

# User input for stock ticker
stock = st.text_input("Enter the Stock Ticker (e.g., AAPL, MSFT)", "AAPL").upper()
start = datetime(datetime.now().year - 20, 1, 1)
end = datetime.now()

# Attempt to load stock data
try:
    data = yf.download(stock, start=start, end=end)
    if data.empty:
        st.warning(f"No data found for the ticker symbol '{stock}'. Please try a different one.")
        st.stop()
    else:
        st.success(f"Successfully loaded data for {stock}.")
except Exception as e:
    st.error(f"Failed to download stock data: {e}")
    st.stop()

# Load pre-trained model
try:
    model = load_model("model.keras")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Display stock data
st.subheader("Stock Data")
st.write(data.tail())

# Calculate Moving Averages
data['MA_250'] = data['Close'].rolling(window=250).mean()
data['MA_200'] = data['Close'].rolling(window=200).mean()
data['MA_100'] = data['Close'].rolling(window=100).mean()

# Function to plot data
def plot_graph(figsize, close_data, moving_avg=None, title=""):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(close_data, label="Close Price", color="blue", linewidth=1.5)
    if moving_avg is not None:
        ax.plot(moving_avg, label="Moving Average", color="orange", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

# Show Moving Averages
st.subheader("Moving Averages")
plot_graph((15, 6), data['Close'], data['MA_250'], "Close Price with 250-day Moving Average")
plot_graph((15, 6), data['Close'], data['MA_100'], "Close Price with 100-day Moving Average")

# Scale data for prediction
splitting_len = int(len(data) * 0.9)
x_test = pd.DataFrame(data['Close'][splitting_len:])
if len(x_test) < 100:
    st.warning("Insufficient data for prediction. Please choose another stock ticker.")
    st.stop()

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test)

# Prepare data for prediction
x_data = []
y_data = []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Predict and inverse transform
predictions = model.predict(x_data)
inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Dataframe for plotting original vs predicted
plotting_data = pd.DataFrame({
    "Original": inv_y_test.reshape(-1),
    "Predicted": inv_predictions.reshape(-1)
}, index=data.index[splitting_len+100:])

# Plot original vs predicted values
st.subheader("Original vs Predicted Close Prices")
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(data['Close'], label="Actual Data", color="blue", linewidth=1.5)
ax.plot(plotting_data['Original'], label="Original Test Data", color="green", linestyle="--")
ax.plot(plotting_data['Predicted'], label="Predicted Data", color="red", linestyle="--")
ax.legend()
st.pyplot(fig)

# Predict tomorrow's price
last_100_days = scaled_data[-100:].reshape(1, 100, 1)
predicted_price = model.predict(last_100_days)
predicted_price = scaler.inverse_transform(predicted_price)
st.subheader("Prediction for Tomorrow's Price")
st.write(f"${predicted_price[0][0]:.2f}")
