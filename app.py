import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
mode = st.sidebar.radio("Mode", ["Single Stock", "Portfolio"])
st.sidebar.title("Simulation Inputs")

if mode == "Single Stock":
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL", help="e.g., AAPL, TSLA, MSFT")
    num_simulations = int(st.sidebar.text_input("Number of Simulations", value="100"))
    num_of_days = int(st.sidebar.text_input("Number of Future Days", value="252"))
    price_history = yf.Ticker(ticker).history(period='1y')
    price_close_history = yf.Ticker(ticker).history(period='1y')[['Close']]

    if price_close_history.empty:
        st.error(f"No data found for ticker '{ticker}'. Please try another.")
    else:
        prices = price_close_history['Close']
        prices_length = len(prices)
        #returns = [(prices.iloc[i+1] - prices.iloc[i]) / prices.iloc[i] for i in range(prices_length - 1)]
        returns = np.log(prices / prices.shift(1)).dropna()
        #mu = np.mean(returns)
        #sigma = np.std(returns)
        mu = returns.mean()
        sigma = returns.std()
        dt = 1
        initial_price = prices.iloc[0]

        price_paths = []
        for _ in range(num_simulations):
            future_prices = [initial_price]
            for i in range(1, num_of_days):
                epsilon = np.random.normal()
                future_price = future_prices[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * epsilon * np.sqrt(dt))
                future_prices.append(future_price)
            price_paths.append(future_prices)

        expected_price = np.mean([path[-1] for path in price_paths])
        expected_price_theoretical = initial_price * np.exp((mu - 0.5 * sigma**2) * num_of_days)


        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Stochastic Modeling of Future Stock Prices Using a Geometric Brownian Motion Algorithm")
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

                In this model, $\mu$ is the drift, calculated as the mean of the historical logarithmic returns, $\sigma$ is the volatility, computed as the standard deviation of those returns, and the stochastic term is modeled by a Wiener process $W_{t}=\sigma\sqrt{\Delta t}$.
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
            st.write(f"Expected Theoretical Price (in {num_of_days} days): ${expected_price_theoretical:.2f}")

        with col4:
            st.subheader(f"{ticker} Price History (1 Year)")
            st.dataframe(price_history)
    
else:
    tickers_input = st.sidebar.text_input("Ticker Symbols (comma-separated)", value="AAPL, MSFT", help="e.g., AAPL, TSLA, MSFT")
    weights_input = st.sidebar.text_input("Individual Stock Weights (comma-separated and must sum to 1)", value="0.5, 0.5", help='e.g., 0.5, 0.5')
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    weights = list(map(float, weights_input.split(',')))
    if not np.isclose(sum(weights), 1.0):
        st.error("Weights must sum to 1.")
        st.stop()
    num_simulations = int(st.sidebar.text_input("Number of Simulations", value="100"))
    num_of_days = int(st.sidebar.text_input("Number of Future Days", value="252"))
    #price_close_history = yf.Ticker(tickers).history(period='1y')[['Close']]

    price_paths = []
    individual_mus = []
    individual_sigmas = []
    initial_prices = []

    for ticker, weight in zip(tickers, weights):
        price_data = yf.Ticker(ticker).history(period='1y')['Close']
        if price_data.empty:
            st.error(f"No data found for ticker '{ticker}'")
            st.stop()
        
        log_returns = np.log(price_data / price_data.shift(1)).dropna()
        mu = log_returns.mean()
        sigma = log_returns.std()
        initial_price = price_data.iloc[0]

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
        st.subheader("Stochastic Modeling of the Future Value of a Multi-Asset Portfolio Using a Geometric Brownian Motion Algorithm")
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

            In this model, $\mu$ is the drift, calculated as the mean of the historical logarithmic returns, $\sigma$ is the volatility, computed as the standard deviation of those returns, and the stochastic term is modeled by a Wiener process $W_{t}=\sigma\sqrt{\Delta t}$.

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
        ax.set_title(f'{num_simulations} Simulations for {tickers} ({num_of_days} Days)')
        ax.set_xlabel('Time Step (Days)')
        ax.set_ylabel('Predicted Price ($)')
        st.pyplot(fig)
        st.subheader("Return Statistics")
        st.write(f"Mean Return (μ): {mu:.5f}")
        st.write(f"Volatility (σ): {sigma:.5f}")
        st.write(f"Initial Price: ${initial_price:.2f}")
        st.write(f"Expected Price (in {num_of_days} days): ${expected_price:.2f}")
        st.write(f"Expected Theoretical Price (in {num_of_days} days): ${expected_price_theoretical:.2f}")
        

    col3, col4 = st.columns(2)

    with col3:
        st.subheader(f"Distribution of Simulated Portfolio Prices After {num_of_days} Days")
        
        final_prices = portfolio_paths[:, -1]

        fig, ax = plt.subplots(figsize=(6, 2.5))
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(final_prices)
        x_vals = np.linspace(min(final_prices), max(final_prices), 500)
        ax.plot(x_vals, kde(x_vals), color='black', label='KDE')
        
        ax.axvline(np.mean(final_prices), color='grey', linestyle='--', label='Expected Price')
        
        ax.set_xlabel("Portfolio Value ($)")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)
        

    with col4:
        pass
        
    
