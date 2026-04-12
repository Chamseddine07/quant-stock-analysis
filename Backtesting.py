import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = yf.download("AAPL", start = "2015-01-01", end = "2026-04-01")
df["Log Return"] = np.log(df["Close"]/df["Close"].shift(1))

train = df.iloc[:int(len(df)*0.7)]
test = df.iloc[int(len(df)*0.7):]

def run_strategy(data, short_window, long_window, transaction_cost = 0.001):
    data = data.copy()
    short_MA = data["Close"].rolling(short_window).mean()
    long_MA = data["Close"].rolling(long_window).mean()

    data["Signal"] = np.where(short_MA > long_MA, 1, 0)
    trades = data["Signal"].diff().abs()
    data["Strategy Return"] = data["Signal"].shift(1) * data["Log Return"] - transaction_cost*trades

    data["Cumulative Market"] = data["Log Return"].cumsum().apply(np.exp) 
    data["Cumulative Strategy"] = data["Strategy Return"].cumsum().apply(np.exp)

    annual_return = data["Strategy Return"].mean() * 252
    annual_vol = data["Strategy Return"].std() * np.sqrt(252)
    sharpe = (annual_return - 0.05) / annual_vol 

    return data, sharpe

train_results, train_sharpe = run_strategy(train, 20, 50)
test_results, test_sharpe = run_strategy(test, 20, 50)

print(f"Train Sharpe Ratio: {train_sharpe:.2f}")
print(f"Test Sharpe Ratio: {test_sharpe:.2f}")

plt.figure(figsize=(12,5))
plt.plot(train_results["Cumulative Market"], color="steelblue", alpha = 1, label="Train Market")
plt.plot(train_results["Cumulative Strategy"], color="green", alpha = 1, label="Train Strategy")
plt.plot(test_results["Cumulative Market"], color="orange", alpha = 1, label="Test Market")
plt.plot(test_results["Cumulative Strategy"], color="red", alpha = 1, label="Test Strategy")
plt.title("Backtesting - Train vs Test")
plt.legend()
plt.show()


best_sharpe = 0
best_short = 0
best_long = 0
"""
for short in range(5, 50):
    for long in range (20, 200):
        if long <= short:
            continue

        _, sharpe = run_strategy(train, short, long)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_long = long
            best_short = short

print(best_sharpe, best_long, best_short)
"""

train_results, train_sharpe = run_strategy(train, 9, 20)
test_results, test_sharpe = run_strategy(test, 9, 20)
print(train_sharpe, test_sharpe)


plt.figure(figsize=(12,5))
plt.plot(train_results["Cumulative Market"], color="steelblue", alpha = 1, label="Train Market")
plt.plot(train_results["Cumulative Strategy"], color="green", alpha = 1, label="Train Strategy")
plt.plot(test_results["Cumulative Market"], color="orange", alpha = 1, label="Test Market")
plt.plot(test_results["Cumulative Strategy"], color="red", alpha = 1, label="Test Strategy")
plt.title("19/22 Strategy")
plt.legend()
plt.show()
