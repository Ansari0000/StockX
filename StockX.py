import streamlit as st
from newsapi import NewsApiClient
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import json
from streamlit.components.v1 import html



def home():
    model = load_model(r'C:\Users\ansar\OneDrive\Desktop\stock\Stock Predictions Model.keras')
    
    
    st.title('Stock Market Predictor')
    st.markdown("Welcome to the StockX Dashboard! This dashboard provides analysis and insights into stock market data.")


    st.subheader('Current Stock Price')

    # Sidebar for user input
    ticker_symbol = st.sidebar.text_input("Enter Ticker Symbol (e.g., GOOG)", value='GOOG', max_chars=10)

    # Fetch stock data
    stock_data = yf.Ticker(ticker_symbol)

    # Fetch real-time stock price
    real_time_price = stock_data.history(period='1d')['Close'].iloc[-1]

    # Display real-time stock price
    st.write(f"Current Price of {ticker_symbol}: {real_time_price:.2f}/-")
    # Yahoo Finance URL button
    
    yahoo_finance_url = f"https://finance.yahoo.com/quote/{ticker_symbol}"
    button_html = f'''
<style>
    .yahoo-finance-button {{
        background-color: #343a40;
        color: #ffffff;
        text-align: center;
        padding: 10px 20px;
        display: inline-block;
        text-decoration: none;
        border-radius: 5px;
        transition: background-color 0.3s ease;
        border: 1px solid #343a40;
    }}
    .yahoo-finance-button:hover {{
        background-color: #1d2124;
        border-color: #1d2124;
    }}
</style>
<a href="{yahoo_finance_url}" target="_blank" class="yahoo-finance-button">Find Stock Symbol with Name on Yahoo Finance</a>
'''
    st.markdown(button_html, unsafe_allow_html=True)


    stock = st.sidebar.text_input('Enter Stock Symbol', 'GOOG')
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2012-01-01'))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('2023-12-31'))

    data = yf.download(stock, start=start_date, end=end_date)

    st.subheader(f'Stock Data for {stock}')
    st.write(data)

    data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)

    st.subheader('Price vs MA50')
    ma_50_days = data.Close.rolling(50).mean()
    fig1 = plt.figure(figsize=(8,6))
    plt.plot(ma_50_days, 'r')
    plt.plot(data.Close, 'g')
    plt.show()
    st.pyplot(fig1)

    st.subheader('Price vs MA50 vs MA100')
    ma_100_days = data.Close.rolling(100).mean()
    fig2 = plt.figure(figsize=(8,6))
    plt.plot(ma_50_days, 'r')
    plt.plot(ma_100_days, 'b')
    plt.plot(data.Close, 'g')
    plt.show()
    st.pyplot(fig2)

    st.subheader('Price vs MA100 vs MA200')
    ma_200_days = data.Close.rolling(200).mean()
    fig3 = plt.figure(figsize=(8,6))
    plt.plot(ma_100_days, 'r')
    plt.plot(ma_200_days, 'b')
    plt.plot(data.Close, 'g')
    plt.show()
    st.pyplot(fig3)

        # Volume Plot
    st.subheader('Volume')
    fig_volume = plt.figure(figsize=(8, 6))
    plt.plot(data.Volume, 'b')
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.title('Volume Traded Over Time')
    st.pyplot(fig_volume)

    # Relative Strength Index (RSI)
    def calculate_rsi(data, window=14):
        delta = data.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    data['RSI'] = calculate_rsi(data['Close'])
    st.subheader('Relative Strength Index (RSI)')
    fig_rsi = plt.figure(figsize=(8, 6))
    plt.plot(data.RSI, 'b')
    plt.xlabel('Time')
    plt.ylabel('RSI')
    plt.title('RSI Over Time')
    st.pyplot(fig_rsi)

    # Moving Average Convergence Divergence (MACD)
    exp1 = data.Close.ewm(span=12, adjust=False).mean()
    exp2 = data.Close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()

    st.subheader('MACD')
    fig_macd = plt.figure(figsize=(8, 6))
    plt.plot(macd, label='MACD', color='blue')
    plt.plot(signal, label='Signal Line', color='red')
    plt.xlabel('Time')
    plt.ylabel('MACD')
    plt.title('MACD and Signal Line Over Time')
    plt.legend()
    st.pyplot(fig_macd)

    # Bollinger Bands
    def calculate_bollinger_bands(data, window=20):
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        return rolling_mean, upper_band, lower_band

    data['RollingMean'], data['UpperBand'], data['LowerBand'] = calculate_bollinger_bands(data['Close'])
    st.subheader('Bollinger Bands')
    fig_bollinger = plt.figure(figsize=(8, 6))
    plt.plot(data.Close, label='Close Price', color='blue')
    plt.plot(data.RollingMean, label='Rolling Mean', color='black')
    plt.fill_between(data.index, data.UpperBand, data.LowerBand, color='gray', alpha=0.2)
    plt.legend()
    st.pyplot(fig_bollinger)

    # Correlation Matrix
    st.subheader('Correlation Matrix')
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')  # Add title to the plot
    st.pyplot(plt) 

    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i,0])

    x,y = np.array(x), np.array(y)

    predict = model.predict(x)

    scale = 1/scaler.scale_

    predict = predict * scale
    y = y * scale

    st.subheader('Original Price vs Predicted Price')
    fig4 = plt.figure(figsize=(8,6))
    plt.plot(predict, 'r', label='Original Price')
    plt.plot(y, 'g', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()
    st.pyplot(fig4)

# Pass the Matplotlib figure object plt

    # News Sentiment Analysis - You can use a third-party API or library to perform sentiment analysis on news articles related to the stock.

    # Fundamental Analysis Metrics - You can fetch fundamental data using yfinance or other APIs and display relevant metrics.

    # Prediction Confidence Intervals -

    # Add contact information in the sidebar
    st.sidebar.subheader('Contact Information')
    st.sidebar.write('- Email: stockx453@gmail.com')
    st.sidebar.write('- Phone: +91 9830023211')


    # Add copyright notice in the footer
    st.markdown(f"<center> © 2024 Stock X PVT-LTD . All rights reserved. | Contact: +91 9823567121 </center>",unsafe_allow_html=True)

# Initialize News API client
newsapi = NewsApiClient(api_key='d99479046c3f4fa59b7d590026dd0496')

def show_news():
    st.title("Stock News Viewer")
    st.markdown(
        """
        <style>
        .title {
            color: #ff6347;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .text-input {
            width: 300px;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
            margin-bottom: 20px;
        }
        .button {
            background-color: #ff6347;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
        }
        .button:hover {
            background-color: #ff7f50;
        }
        .subheader {
            color: #ff6347;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .article-title {
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        .description {
            font-size: 16px;
            margin-bottom: 5px;
        }
        .url {
            font-size: 14px;
            color: #007bff;
            margin-bottom: 5px;
        }
        .published-at {
            font-size: 14px;
            color: #777;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Allow users to enter a stock symbol
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):", max_chars=15, key='stock_symbol')

    # Fetch news articles related to the stock symbol
    if st.button("Get News", key='get_news'):
        if stock_symbol:
            # Fetch news articles using News API
            articles = newsapi.get_everything(q=stock_symbol, language='en', sort_by='publishedAt')

            # Display the articles
            if articles['totalResults'] > 0:
                st.subheader("Latest News Articles:")
                for article in articles['articles']:
                    st.markdown(
                        f"""
                        <div class="article-title">{article['title']}</div>
                        <div class="description">{article['description']}</div>
                        <a href="{article['url']}" class="url" target="_blank">{article['url']}</a>
                        <div class="published-at">Published at: {article['publishedAt']}</div>
                        <hr style="margin-top: 10px; margin-bottom: 10px;">
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.write("No news found for the given stock symbol.")
        else:
            
            st.write("Please enter a stock symbol.")

#CHATBOT
    def get_stock_price(ticker):
        return str(yf.Ticker(ticker).history(periods='1y').iloc[-1].Close)

    def calculate_SMA(ticker, window):
        data = yf.Ticker(ticker).history(period='1y').Close
        return str(data.rolling(window=window).mean().iloc[-1])

    def calculate_EMA(ticker, window):
        data = yf.Ticker(ticker).history(period='1y').Close
        return str(data.ewm(span=window, adjust=False).mean().iloc[-1])

    def calculate_RSI(ticker):
        data = yf.Ticker(ticker).history(period='1y').Close
        delta = data.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=14-1, adjust=False).mean()
        ema_down = down.ewm(com=14-1, adjust=False).mean()
        rs = ema_up / ema_down

    def calculate_MACD(ticker):
        data = yf.Ticker(ticker).history(period='1y').Close
        short_EMA = data.ewm(span=12 , adjust=False).mean()
        long_EMA = data.ewm(span=26 , adjust=False).mean()
        MACD = short_EMA - long_EMA
        singal = MACD.ewm(span=9, adjust=False).mean()
        MACD_histogram = MACD - singal
        return f'{MACD[-1]}, {singal[-1]}, {MACD_histogram[-1]}'

    def plot_stock_price(ticker):
        data = yf.Ticker(ticker).history(period='1y')
        plt.figure(figsize=(10,5))
        plt.plot(data.index, data.Close)
        plt.title(f'{ticker} stock price over last year')
        plt.xlabel('date')
        plt.ylabel('stock price ($)')
        plt.grid(True)
        plt.savefig('stock.png')
        plt.close()

    functions = [
        {
            'name': 'get_stock_price',
            'description': 'gets the latest stock price given the ticker symbol of company.',
            'parameter': {
                'type': 'object',
                'properties': {
                    'ticker': {
                        'type': 'string',
                        'description': 'the stock ticker symbol for a company (e.g., AAPL for Apple).'
                    }
                },
                'required': ['ticker']
            }
    },
        {
    "name": "calculate_SMA",
            "description": "calculate the simple moving average for a given stock ticker and a window",
            "parameter": {
                "type": "object",
                "properties": {
    "ticker": {
                        "type": "string",
                        "description": "the stock ticker symbol for a company (e.g., AAPL for Apple)"
                    },
                    "window": {
                        "type": "integer",
                        "description": "the timeframe to consider when calculating the SMA"
                    }
                },
                "required": ["ticker", "window"]
            }
        },
        # Define other functions similarly
    ]

    available_function = {
        'get_stock_price': get_stock_price,
        'calculate_SMA': calculate_SMA,
        'calculate_EMA': calculate_EMA,
        'calculate_MACD': calculate_MACD,
        'plot_stock_price': plot_stock_price
    }


            
def integrate_virtual_stock_assistant():
    st.title("Virtual Stock Assistant")
    st.subheader('"I am your stocking buddy, here to help you navigate the markets!"')

    # Initialize OpenAI client
    client = OpenAI()

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Function to display chat messages
    def display_messages(messages):
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Display chat messages from history on app rerun
    display_messages(st.session_state.messages)

    # Accept user input
    if prompt := st.text_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

    # Generate response from OpenAI
    if st.session_state.messages:
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="gpt-4-0125-preview",  # Updated model name
                messages=st.session_state.messages
            )
        # Display assistant response in chat message container
        assistant_response = response.choices[0].message.content.strip()
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    # Add clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages.clear()  # Clear the chat history
        st.empty()  # Clear the chat display area
# Main function to control the app flow
def main():
    st.set_page_config(page_title="StockX Dashboard", page_icon=":chart_with_upwards_trend:")
    st.sidebar.markdown("<h1 style='font-size: 24px; font-weight: bold;'>Dashboard</h1>", unsafe_allow_html=True)
    page = st.sidebar.radio("Go to", ["📈 Home", "📰 News","🤖 Assistant"])

    if page == "📈 Home":
        home()

    elif page == "📰 News":
        show_news()

    elif page == "🤖 Assistant":
        integrate_virtual_stock_assistant()

if __name__ == "__main__":
    main()