import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import time

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

# load the model
best_model = torch.load('./lstm_model_1472.pt')
best_model = best_model.module_
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model = best_model.to(device)
best_model.eval()

# load completely new test data
df = pd.read_csv('./trading_alg_tuning_data/full_btc_usdt_data_feature_engineered.csv')
df = df.dropna()

# remove constant columns
std_dev = df.std()
non_constant_columns = std_dev[std_dev != 0].index.tolist()
df = df[non_constant_columns]

# scale the data
X = df.drop('Close', axis=1).values
y = df['Close'].values.reshape(-1, 1)

# open the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X = scaler.transform(X)

def calculate_past_average(predictions, window):
    if len(predictions) < window:
        return np.mean(predictions)
    else:
        return np.mean(predictions[-window:])
    
# env set up
initial_budget = 10000

def calculate_sharpe_ratio(returns, risk_free_rate):
    excess_returns = returns - risk_free_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std()
    return sharpe_ratio

def calculate_max_drawdown(portfolio_values):
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    max_drawdown = drawdown.min()
    return max_drawdown

current_best_profit = -100

# Function to run a single simulation with given parameters
def run_simulation(params):

    alpha_atr, alpha_rsi, base_sell_threshold, base_buy_threshold, sell_percentage, buy_percentage, window_size, min_profit_threshold = params['alpha_atr'], params['alpha_rsi'], params['sell_threshold'], params['buy_threshold'], params['sell_percentage'], params['buy_percentage'], params['window_size'], params['min_profit_threshold']


    current_budget = initial_budget
    assets_held = 0
    past_predictions = []
    portfolio_values = []

    trading_fee_percentage = 0.0180 / 100  # Trading fee of 0.0180%

    # the simulated trading
    for i in range(X.shape[0]):
        # Reshape the data
        input_data = torch.tensor(X[i].reshape(1, 1, -1)).float().to(device)  # Reshape to (1, 1, num_features)

        atr_value = df['ATR_15'].iloc[i]
        rsi_value = df['RSI_15'].iloc[i]

        sell_threshold = base_sell_threshold + alpha_atr * atr_value + alpha_rsi * max(0, rsi_value - 70)
        buy_threshold = base_buy_threshold - alpha_atr * atr_value - alpha_rsi * max(0, 70 - rsi_value)

        # predictions
        prediction = best_model(input_data).item()

        past_predictions.append(prediction)
        past_average = calculate_past_average(past_predictions, window=window_size)
        
        trend_direction = 'up' if prediction > past_average else 'down'
        # positive -> probably going up
        # negative -> probably going down
        confidence = (prediction - past_average) / past_average

        asset_price = y[i][0]

        # Calculate expected profit
        expected_price_increase = prediction - asset_price
        expected_profit_percent = (expected_price_increase / asset_price) * 100

        if trend_direction == 'up' and confidence >= buy_threshold and expected_profit_percent >= min_profit_threshold:
            investment = min(current_budget * abs(confidence) * buy_percentage, current_budget)
            fee = investment * trading_fee_percentage
            rounded_investment = round(investment - fee, 6)
            assets_bought = (rounded_investment / asset_price)
            assets_held += assets_bought
            current_budget -= rounded_investment
        elif trend_direction == 'down' and confidence <= sell_threshold and assets_held > 0:
            assets_to_sell = assets_held * sell_percentage * abs(confidence)
            sale_revenue = assets_to_sell * asset_price
            fee = sale_revenue * trading_fee_percentage
            assets_held -= assets_to_sell
            current_budget += round(sale_revenue - fee, 6)

        final_asset_value = assets_held * asset_price
        current_portfolio_value = current_budget + final_asset_value

        if i % 1000 == 0:
            print(f"atr_value: {atr_value}, rsi_value: {rsi_value}")
            print(f"Prediction: {trend_direction} Confidence: {confidence}")
            print(f"Second {i}, Budget: {current_budget}, Assets Held: {assets_held}, Asset Price: {asset_price}")
            print(f"Portfolio Value: {current_portfolio_value}")
            portfolio_values.append(current_portfolio_value)
    


    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]  # Calculate returns
    sharpe_ratio = calculate_sharpe_ratio(returns, 0.000006811)  # Calculate sharpe ratio, risk free rate per 1000 seconds is interpreted from a 3 month US treasury bill yield of 5.37%
    max_drawdown = calculate_max_drawdown(portfolio_values) # Calculate max drawdown

    print("LSTM strategy")
    print(f"Initial Budget: {initial_budget}, Final Portfolio Value: {current_portfolio_value}")
    print(f"Total Profit: {current_portfolio_value - initial_budget}")
    print("Sharpe Ratio:", sharpe_ratio)
    print("Maximum Drawdown:", max_drawdown)

    return current_portfolio_value - initial_budget, sharpe_ratio, max_drawdown, portfolio_values  # Return profit


    
# [-inf, sell_threshold]  -> sell
# (sell_threshold, buy_threshold) -> hold
# [buy_threshold, inf] -> buy

# optimal parameters number 1, sorted by profit found a local max, but way too conservative
base_sell_threshold = 0.000387    # Confidence level to trigger a sell
base_buy_threshold = 0.000595     # Confidence level to trigger a buy
alpha_atr = 0.420739              # Weight of ATR's effect in thresholds
alpha_rsi = 0.284716              # Weight of RSI's effect in thresholds
sell_percentage = 0.178912        # Percentage of assets to sell
buy_percentage = 0.419224         # Percentage of budget to buy with
window_size = 49                  # Number of past predictions to average over


# optimal parameters number 2, sorted by sharpe ratio instead!
base_sell_threshold = 0.000318
base_buy_threshold = 0.000541
alpha_atr = 26.117467
alpha_rsi = 29.549422
sell_percentage = 0.314003
buy_percentage = 0.671026
window_size = 17
min_profit_threshold = 0.768993

base_sell_threshold_range = [0.0002, 0.0006]
base_buy_threshold_range = [0.0001, 0.0006]
alpha_atr_range = [2, 32]
alpha_rsi_range = [9, 47]
sell_percentage_range = [0.2, 0.81]
buy_percentage_range = [0.4, 0.82]
window_size_range = [5, 25]
min_profit_threshold_range = [0.2, 0.95]

num_iterations = 200

results = []
for i in range(num_iterations):
    # Randomly select parameters
    alpha_atr = round(random.uniform(*alpha_atr_range), 6)
    alpha_rsi = round(random.uniform(*alpha_rsi_range), 6)
    base_sell_threshold = round(random.uniform(*base_sell_threshold_range), 6)
    base_buy_threshold = round(random.uniform(*base_buy_threshold_range), 6)
    sell_percentage = round(random.uniform(*sell_percentage_range), 6)
    buy_percentage = round(random.uniform(*buy_percentage_range), 6)
    min_profit_threshold = round(random.uniform(*min_profit_threshold_range), 6)
    window_size = random.randint(*window_size_range)

    params = {
        'alpha_atr': alpha_atr,
        'alpha_rsi': alpha_rsi,
        'sell_threshold': base_sell_threshold,
        'buy_threshold': base_buy_threshold,
        'sell_percentage': sell_percentage,
        'buy_percentage': buy_percentage,
        'window_size': window_size,
        'min_profit_threshold': min_profit_threshold
    }

    print(f"Running simulation {i + 1} with parameters:")

    # Run the simulation

    start = time.time()
    profit, sharpe, drawdown, resulting_portfolio_values = run_simulation(params)
    end = time.time()

    print(f"Simulation {i + 1} complete")
    print(f"Time taken: {end - start} seconds")

    # Store the results
    results.append({
        'alpha_atr': alpha_atr,
        'alpha_rsi': alpha_rsi,
        'sell_threshold': base_sell_threshold,
        'buy_threshold': base_buy_threshold,
        'sell_percentage': sell_percentage,
        'buy_percentage': buy_percentage,
        'window_size': window_size,
        'min_profit_threshold': min_profit_threshold,
        'profit': profit,
        'sharpe_ratio': sharpe,
        'max_drawdown': drawdown,
        'resulting_portfolio_values': resulting_portfolio_values
    })

# Sort the results by profit
results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)

# Print the top results
for result in results[:20]:  # Print top 10 results
    resulting_portfolio_value_array = result['resulting_portfolio_values']
    plt.plot(resulting_portfolio_value_array)

    del result['resulting_portfolio_values']

    print(result)


# compare with buy and hold
portfolio_values = []
current_budget = initial_budget
assets_held = 0
trading_fee_percentage = 0.0180 / 100  # Trading fee of 0.0180%
for i in range(X.shape[0]):
    asset_price = y[i][0]
    final_asset_value = assets_held * asset_price
    current_portfolio_value = current_budget + final_asset_value
    if i == 0:
        assets_bought = (current_budget / asset_price)
        assets_held += assets_bought
        current_budget -= current_budget
    elif (i%1000 == 0):
        portfolio_values.append(current_portfolio_value)
        assets_held = assets_held

portfolio_values = np.array(portfolio_values)
returns = np.diff(portfolio_values) / portfolio_values[:-1]  # Calculate returns
sharpe_ratio = calculate_sharpe_ratio(returns, 0.000006811)  # Calculate sharpe ratio, risk free rate per 1000 seconds is interpreted from a 3 month US treasury bill yield of 5.37%
max_drawdown = calculate_max_drawdown(portfolio_values) # Calculate max drawdown

print("Buy and hold strategy")
print(f"Initial Budget: {initial_budget}, Final Portfolio Value: {current_portfolio_value}")
print(f"Total Profit: {current_portfolio_value - initial_budget}")
print("Sharpe Ratio:", sharpe_ratio)
print("Maximum Drawdown:", max_drawdown)

# plot as dotted line
plt.plot(portfolio_values, '--')
plt.title('Portfolio Value Over Time')
plt.xlabel('Time (1000 seconds)')
plt.ylabel('Portfolio Value ($)')
plt.show()