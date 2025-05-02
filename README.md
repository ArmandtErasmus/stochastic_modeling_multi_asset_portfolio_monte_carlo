# Stochastic Modeling of the Future Value of a Single and Multi-Asset Portfolio Using a Geometric Brownian Motion Algorithm

---

![dashboard](https://github.com/ArmandtErasmus/stochastic_modeling_multi_asset_portfolio_monte_carlo/blob/main/stochastic.png)

# Try it out! Visit the Dashboard Below:
[Demo](https://stochasticmodelingmultiassetportfoliomontecarlo.streamlit.app/)
[Embed](https://stochasticmodelingmultiassetportfoliomontecarlo.streamlit.app/?embed_options=show_toolbar,light_theme,show_colored_line,show_padding,show_footer)

---

### Table of Contents
1. [Introduction](#1.-Introduction)
2. [Features](#2.-Features)
3. [Installation](#3.-Installation)
4. [Usage](#4.-Usage)
5. [Contributing](#6.Contributing)
6. [License](#7.License)

---

### 1. Introduction

This Streamlit web application enables users to simulate the future price of a single stock or an entire customizable portfolio of multiple stocks, using Geometric Brownian Motion (GBM). It offers:

- Monte Carlo simulations of future stock or portfolio prices.
- Interactive parameter selection (number of simulations, forecast horizon).
- Return and volatility statistics.
- Theoretical and simulated expected value comparisons.
- Graphical visualizations of historical data and simulation results.

This tool is ideal for educational, research, or personal finance experimentation purposes.

---

### 2. Features

## Application Structure
The app is divided into two modes:

- Single Stock
- Portfolio

Users can toggle between these via a radio button in the sidebar.

## Inputs (Sidebar Controls)

| Parameter                      | Description                                                             |
| ------------------------------ | ----------------------------------------------------------------------- |
| **Mode**                       | "Single Stock" or "Portfolio"                                           |
| **Ticker Symbol(s)**           | Stock symbols (e.g., `AAPL`, `TSLA`, `MSFT`)                            |
| **Weights** *(Portfolio only)* | Comma-separated values (e.g., `0.5, 0.5`) that must sum to 1            |
| **Number of Simulations**      | Number of Monte Carlo paths to generate                                 |
| **Number of Future Days**      | Time horizon over which simulation is done (e.g., 252 = 1 trading year) |

## Mathematical Model
This app uses the Geometric Brownian Motion (GBM) model for stock price evolution:

### GBM Equation
For a single asset:

    Sₜ = Sₜ₋₁ × exp[(μ - ½σ²)Δt + σε√Δt]

### Theoretical Expected Price

    𝐄[Sₜ] = S₀ × exp[(μ − ½σ²) × T]

For a portfolio, the weighted path of each asset is simulated independently and aggregated.
Portfolio drift and volatility are computed as:

    Drift (μₚ)     = ∑ wᵢ × μᵢ  
    Volatility (σₚ) = √(∑ wᵢ² × σᵢ²)

## Workflow and Simulation Details

1. Historical Data Fetching:
   - Uses yfinance to pull 1 year of daily historical data.
   - Only the closing price is used to compute log returns.
2. Log Returns:
   Computed as:

       rₜ = ln(Sₜ / Sₜ₋₁)

4. Monte Carlo Simulation:
   - For each simulation path:
     1. Start with the initial price.
     2. Iteratively compute next price using GBM.
     3. Repeat for num_of_days.
   - Repeat num_simulations times.
   - Aggregate and plot all paths.

## Interface Components
### Single Stock Mode
Left Panel:

- GBM Model Explanation with LaTeX math
- Return and volatility display
- Theoretical and simulated expected price
- Ticker historical dataframe

Right Panel:

- Monte Carlo simulation plot

### Portfolio Mode
Left Panel:

- GBM model for multi-asset portfolios
- Return statistics (drift, volatility, expected price)

Right Panel:

- Plot of Monte Carlo portfolio paths
- Density estimation (KDE) of terminal portfolio value

## Outputs & Interpretation
### Graphs:
- Monte Carlo Price Paths: Shows price evolution of simulated paths.
- KDE Plot: Probability density of final prices.

### Numerical Statistics:
- Initial price.
- Estimated μ and σ.
- Final expected price (simulation average).
- Theoretical expected price using GBM formula.
---
 
### 3. Installation

To run the dashboard, follow these steps:

#### Prerequisites
Ensure you have the following installed on your system:

- Python 3/above
- streamlit
- yfinance
- numpy
- matplotlib
- scipy

OR install them by copying and pasting the command below into your terminal:
```
pip install streamlit yfinance matplotlib numpy scipy
```
#### Step-by-Step Installation

1. Clone the repository:
   ```
   git clone https://github.com/armandterasmus/stochastic_modeling_multi_asset_portfolio_monte_carlo
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

---

### 4. Usage

After installation, you can run the tool using:
    ```
    streamlit run app.py
    ```
This will open the app in a new browser window.

---

### 5. Contributing
Contributions to the dashboard are welcome! If you encounter bugs or would like to add new features, please feel free to submit issues or pull requests.

#### Ideas for Future Features:
- Correlated GBM paths using Cholesky decomposition.
- Sharpe ratio and Value-at-Risk (VaR) statistics.
- User-defined time intervals (weekly/monthly returns).
- Integration with other APIs (e.g., Alpha Vantage, Quandl).
- Saving and exporting simulated data to CSV.

#### Steps to contribute:
1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Make changes and commit (git commit -m 'Add some feature').
4. Push changes (git push origin feature-branch).
5. Submit a pull request.

---

### 6. License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/ArmandtErasmus/xpl0it3r/blob/main/LICENSE) file for more details.

