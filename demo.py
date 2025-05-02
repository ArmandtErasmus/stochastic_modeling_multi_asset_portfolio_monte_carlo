import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(layout="wide")

# Define popular tickers
popular_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "BRK-B", "UNH"]

# Function to load historical data for popular tickers
@st.cache_data
def load_popular_data():
    data = {}
    for ticker in popular_tickers:
        try:
            hist = yf.Ticker(ticker).history(period='1y')
            if not hist.empty:
                data[ticker] = hist
        except Exception as e:
            st.warning(f"Error loading data for {ticker}: {e}")
    return data

# Load data
popular_data = load_popular_data()

# Sidebar for mode selection
mode = st.sidebar.radio("Mode", ["Single Stock", "Portfolio"])
st.sidebar.title("Simulation Inputs")

if mode == "Single Stock":
    # Single stock selection
    ticker = st.sidebar.selectbox("Select a Ticker", popular_tickers)
    num_simulations = int(st.sidebar.text_input("Number of Simulations", value="100"))
    num_of_days = int(st.sidebar.text_input("Number of Future Days", value="252"))

    price_history = popular_data.get(ticker)
    if price_history is None or price_history.empty:
        st.error(f"No data found for ticker '{ticker}'. Please try another.")
    else:
        prices = price_history['Close']
        returns = np.log(prices / prices.shift(1)).dropna()
        mu = returns.mean()
        sigma = returns.std()
        dt = 1
        initial_price = prices.iloc[-1]

        # Monte Carlo simulation
        price_paths = []
        for _ in range(num_simulations):
            future_prices = [initial_price]
            for _ in range(1, num_of_days):
                epsilon = np.random.normal()
                future_price = future_prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * epsilon * np.sqrt(dt))
                future_prices.append(future_price)
            price_paths.append(future_prices)

        expected_price = np.mean([path[-1] for path in price_paths])
        expected_price_theoretical = initial_price * np.exp((mu - 0.5 * sigma**2) * num_of_days)

        # Layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Geometric Brownian Motion Model")
            st.markdown(
                r"""
                This app uses a **Geometric Brownian Motion (GBM)** model to simulate stock price paths.
                
                The model assumes returns are normally distributed with:
                - Mean: $\mu$
                - Standard deviation: $\sigma$
                
                The simulated price at time step $t$ is:
                $$
                S_t = S_{t-1} \cdot e^{\left( \mu - \frac{1}{2}\sigma^2 \right)\Delta t + \sigma \epsilon \sqrt{\Delta t}}
                $$
                where $\epsilon \sim \mathcal{N}(0,1)$.
                """
            )

        with col2:
            st.subheader("Monte Carlo Simulation Plot")
            fig, ax = plt.subplots(figsize=(10, 5))
            for path in price_paths:
                ax.plot(path)
            ax.set_title(f'{num_simulations} Simulations for {ticker} ({num_of_days} Days)')
            ax.set_xlabel('Time Step (Days)')
            ax.set_ylabel('Predicted Price ($)')
            st.pyplot(fig)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Return Statistics")
            st.write(f"Mean Return (μ): {mu:.5f}")
            st.write(f"Volatility (σ): {sigma:.5f}")
            st.write(f"Initial Price: ${initial_price:.2f}")
            st.write(f"Expected Price (in {num_of_days} days): ${expected_price:.2f}")
            st.write(f"Theoretical Expected Price (in {num_of_days} days): ${expected_price_theoretical:.2f}")

        with col4:
            st.subheader(f"{ticker} Price History (1 Year)")
            st.line_chart(prices)

else:
    # Portfolio mode
    selected_tickers = st.sidebar.multiselect("Select Ticker Symbols", popular_tickers, default=["AAPL", "MSFT"])
    weights_input = st.sidebar.text_input("Individual Stock Weights (comma-separated and must sum to 1)", value="0.5, 0.5")
    try:
        weights = list(map(float, weights_input.split(',')))
    except ValueError:
        st.error("Please enter valid weights.")
        st.stop()

    if len(weights) != len(selected_tickers):
        st.error("Number of weights must match number of selected tickers.")
        st.stop()

    if not np.isclose(sum(weights), 1.0):
        st.error("Weights must sum to 1.")
        st.stop()

    num_simulations = int(st.sidebar.text_input("Number of Simulations", value="100"))
    num_of_days = int(st.sidebar.text_input("Number of Future Days", value="252"))

    price_paths = []
    individual_mus = []
    individual_sigmas = []
    initial_prices = []

    for ticker, weight in zip(selected_tickers, weights):
        price_data = popular_data.get(ticker)
        if price_data is None or price_data.empty:
            st.error(f"No data found for ticker '{ticker}'")
            st.stop()

        prices = price_data['Close']
        returns = np.log(prices / prices.shift(1)).dropna()
        mu = returns.mean()
        sigma = returns.std()
        initial_price = prices.iloc[-1]

        individual_mus.append(mu)
        individual_sigmas.append(sigma)
        initial_prices.append(initial_price)

        simulated_paths = []
        for _ in range(num_simulations):
            path = [initial_price]
            for _ in range(1, num_of_days):
                epsilon = np.random.normal()
                next_price = path[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * epsilon)
                path.append(next_price)
            simulated_paths.append(path)

        weighted_paths = np.array(simulated_paths) * weight
        price_paths.append(weighted_paths)

    portfolio_paths = np.sum(price_paths, axis=0)

    expected_price = np.mean([path[-1] for path in portfolio_paths])
    mu_portfolio = np.dot(individual_mus, weights)
    sigma_portfolio = np.sqrt(np.dot(np.square(individual_sigmas), np.square(weights)))
    initial_portfolio_price = np.dot(initial_prices, weights)

    expected_price_theoretical = initial_portfolio_price * np.exp((mu_portfolio - 0.5 * sigma_portfolio**2) * num_of_days)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Geometric Brownian Motion Model for Portfolio")
        st.markdown(
            r"""
            This app uses a **Geometric Brownian Motion (GBM)** model to simulate price paths for each stock in a portfolio.
            
            For each stock, the model assumes returns are normally distributed with:
            - Mean: $\mu$
            - Standard deviation: $\sigma$
            
            The simulated price at time step $t$ for each stock is given by:
            $$
            S_t = S_{t-1} \cdot e^{\left( \mu - \frac{1}{2}\sigma^2 \right)\Delta t + \sigma \epsilon \sqrt{\Delta t}}
            $$
            where $\epsilon \sim \mathcal{N}(0,1)$.

            Each simulated stock path is then weighted by its user-defined portfolio allocation. The final **portfolio path** is the sum of these weighted paths across all stocks.

            For the theoretical expected return of the portfolio, we compute:
            - **Portfolio drift**: weighted average of individual stock drifts, $\mu_p = \sum w_i \mu_i$
            - **Portfolio volatility**: weighted root-sum-square of individual volatilities, $\sigma_p = \sqrt{\sum w_i^2 \sigma_i^2}$
            - **Expected future price**: 
            $$
            \mathbb{E}[P_T] = P_0 \cdot e^{\left( \mu_p - \frac{1}{2} \sigma_p^2 \right) T}
            $$
            where $P_0$ is the weighted initial portfolio price, and $T$ is the number of time steps (days).
            """
        )

    with col2:
        st.subheader("Monte Carlo Simulation Plot")
        fig, ax = plt.subplots(figsize=(10, 5))
        for path in portfolio_paths:
            ax.plot(path)
        ax.set_title(f'{num_simulations} Simulations for Portfolio ({num_of_days} Days)')
        ax.set_xlabel('Time Step (Days)')
        ax.set_ylabel('Predicted Portfolio Value ($)')
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Portfolio Return Statistics")
        st.write(f"Mean Return (μ): {mu_portfolio:.5f}")
        st.write(f"Volatility (σ): {sigma_portfolio:.5f}")
        st.write(f"Initial Portfolio Value: ${initial_portfolio_price:.2f}")
        st.write(f"Expected Portfolio Value (in {num_of_days} days): ${expected_price:.2f}")
        st.write(f"Theoretical Expected Portfolio Value (in {num_of_days} days): ${expected_price_theoretical:.2f}")

    with col4:
        st.subheader(f"Distribution of Simulated Portfolio Prices After {num_of_days} Days")
        final_prices = portfolio_paths[:, -1]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(final_prices, bins=30, density=True, color='g')
        ax.axvline(np.mean(final_prices), color='r', linestyle='dashed', linewidth=2)
        ax.set_title("Histogram of Final Portfolio Values")
        ax.set_xlabel("Portfolio Value ($)")
        ax.set_ylabel("Density")
        st.pyplot(fig)
