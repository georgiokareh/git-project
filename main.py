import requests
import os 
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
def fetch_stock_data(symbol):
    key = os.environ.get('API_KEY')
    r = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={key}")
    data = r.json()
    return data['Time Series (Daily)']
def init_db(db):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS stock_data
                      (symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER)''')
    conn.commit()
    conn.close()
def save_stock_data(db, symbol,data):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    for date, values in data.items():
        cursor.execute('''INSERT INTO stock_data (symbol, date, open, high, low, close, volume)
                          VALUES (?, ?, ?, ?, ?, ?, ?)''',
                       (symbol, date, float(values['1. open']), float(values['2. high']),
                        float(values['3. low']), float(values['4. close']), int(values['5. volume'])))
    conn.commit()
    conn.close()
def load_to_dataframe(db):
    conn = sqlite3.connect(db)
    df = pd.read_sql_query("SELECT * FROM stock_data", conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', ascending=True, inplace=True)
    df['change'] = df['close']-df['open']
    return df
def plot_stock_data(df, symbols):
    plt.figure(figsize=(10, 5))
    for symbol in symbols:
        symbol_df = df[df['symbol'] == symbol]
        plt.plot(symbol_df['date'], symbol_df['change'], label=symbol)
    plt.title('Daily Stock Price Change')
    plt.xlabel('Date')
    plt.ylabel('Price Change')
    plt.legend()
    plt.show()
def plot_volume_data(df, symbols):
    plt.figure(figsize=(10, 5))
    for symbol in symbols:
        symbol_df = df[df['symbol'] == symbol]
        plt.plot(symbol_df['date'], symbol_df['volume'], label=symbol)
    plt.title('Daily Stock Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.show()
def plot_highs(df, symbols):
    plt.figure(figsize=(10, 5))
    for symbol in symbols:
        symbol_df = df[df['symbol'] == symbol]
        plt.plot(symbol_df['date'], symbol_df['high'], label=symbol)
    plt.title('Daily Stock High Prices')
    plt.xlabel('Date')
    plt.ylabel('High Price')
    plt.legend()
    plt.show()
        

def main():
    init_db('stocks.db')
    symbols = ['NVDA', 'GOOGL', 'PLTR']
    for symbol in symbols:
        data = fetch_stock_data(symbol)
        save_stock_data('stocks.db', symbol, data)
    df = load_to_dataframe('stocks.db')
    plot_highs(df, symbols)

if __name__ == "__main__":
    main()


