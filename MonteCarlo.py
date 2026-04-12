import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = yf.download("AAPL", start = "2020-01-01", end = "2024-01-01")
df["Log Return"]= np.log(df["Close"]/df["Close"].shift(1))
df["Volatility"] = df["Log Return"].rolling(30).std() * np.sqrt(252)

plt.figure(figsize=(12, 5))
plt.plot(df["Close"], label="Close Price")
plt.title("Apple Stock Price (2020 - 2024)")
plt.xlabel("X axis")
plt.ylabel("Y axis")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(df["Log Return"], color="steelblue", alpha=0.7, label="Daily Log Return")
ax1.legend()

ax2.plot(df["Volatility"], color="crimson", alpha=0.7, label="30 day Rolling Volatility")
ax2.legend()

df["MA20"] =  df["Close"].rolling(20).mean()
df["MA50"] =  df["Close"].rolling(50).mean()
df["Signal"] = np.where(df["MA20"] > df["MA50"], 1, 0)
df["Strategy Return"] = df["Signal"].shift(1) * df["Log Return"]
df["Cumulative Market"] = df["Log Return"].cumsum().apply(np.exp)
df["Cumulative Strategy"] = df["Strategy Return"].cumsum().apply(np.exp)

plt.figure(figsize=(12, 5))
plt.plot(df["Cumulative Market"], color="steelblue", alpha=0.7, label="Buy & Hold")
plt.plot(df["Cumulative Strategy"], color="green", alpha=0.7, label="MA Crossover Strategy")
plt.legend()
plt.show()

mu = df["Log Return"].mean() * 252
sigma = df["Log Return"].std() * np.sqrt(252)
S0 = float(df["Close"].iloc[-1].iloc[0])
T = 1
dt = 1/252
N = 252
n_sims = 500
simulations = np.zeros((N, n_sims))

for i in range(n_sims):
    prices = [S0]
    for t in range(251):
        prices.append((prices[-1] * np.exp((mu-np.square(sigma)/2)*dt + (sigma*np.sqrt(dt) * np.random.standard_normal()))))
    simulations[:, i] = prices

plt.plot(simulations, alpha = 0.05, color = "steelblue")
plt.plot(np.percentile(simulations, 5, axis = 1), color = "red", alpha = 1, label = "5th Percentile")
plt.plot(np.percentile(simulations, 95, axis = 1), color = "green", alpha = 1, label = "95th Percentile")
plt.plot(np.median(simulations, axis = 1), color = "orange", alpha = 1, label = "Median")
plt.title("Monte Carlo Simulation")
plt.legend()
plt.show()

final_prices = simulations[-1, :]
print(f"5th Percentile:  ${np.percentile(final_prices, 5):.2f}")
print(f"95th Percentile: ${np.percentile(final_prices, 95):.2f}")
print(f"Median:          ${np.median(final_prices):.2f}")