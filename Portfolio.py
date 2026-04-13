import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

tickers = ["TSLA", "NVDA", "GC=F", "^GSPC"]
df = yf.download(tickers, start="2015-01-01", end="2026-04-12")
df = df["Close"]
returns = np.log(df/df.shift(1))
weights = np.array([0.25, 0.25, 0.25, 0.25])
portfolio_returns = returns.dot(weights)

portfolio_train = portfolio_returns[:int(len(df)*0.7)]
portfolio_test = portfolio_returns[int(len(df)*0.7):]

mu = portfolio_train.mean() * 252
sigma = portfolio_train.std() * np.sqrt(252) 

S0 = 10000
T = 1
dt = 1/252
N = 252
n_sims = 500
simulation = np.zeros((N, n_sims))

for i in range(n_sims):
    portfolio = [S0]
    for t in range(N-1):
        portfolio.append(portfolio[-1] * np.exp((mu - (np.square(sigma)/2)) * dt + sigma*np.sqrt(dt)*np.random.standard_normal()))
    simulation[:, i] = portfolio

plt.figure(figsize = (12,5))
plt.plot(simulation, color="steelblue", alpha = 0.05)
plt.plot(np.percentile(simulation, 5, axis = 1), color = "red", alpha = 1, label = "5th percentile")
plt.plot(np.percentile(simulation, 95, axis = 1), color = "green", alpha = 1, label = "95th percentile")
plt.plot(np.median(simulation, axis = 1), color ="orange", alpha = 1, label = "Median")
plt.title("Portfolio Simulation")
plt.legend()
plt.show()

final_values = simulation[-1, :]

print(f"5th percentile: ${np.percentile(final_values, 5):.2f}")
print(f"95th percentile: ${np.percentile(final_values, 95):.2f}")
print(f"Median: ${np.median(final_values):.2f}")

tsla_value = returns["TSLA"].cumsum().apply(np.exp) * 2500
nvda_value = returns["NVDA"].cumsum().apply(np.exp) * 2500
gold_value = returns["GC=F"].cumsum().apply(np.exp) * 2500
sp500_value = returns["^GSPC"].cumsum().apply(np.exp) * 2500

plt.figure(figsize = (12,5))
plt.plot(tsla_value, color = "red", alpha = 1, label = "Tesla")
plt.plot(nvda_value, color = "green", alpha = 1, label = "NVDIA")
plt.plot(gold_value, color = "yellow", alpha = 1, label = "Gold")
plt.plot(sp500_value, color = "blue", alpha = 1, label = "S&P500")
plt.title("Best Return")
plt.legend()
plt.show()

print(f"TESLA: ${tsla_value.iloc[-1]:.2f}")
print(f"NVDIA: ${nvda_value.iloc[-1]:.2f}")
print(f"GOLD: ${gold_value.iloc[-1]:.2f}")
print(f"S&P500: ${sp500_value.iloc[-1]:.2f}")
