import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

st.set_page_config(layout="wide")

DATA_DIR = "stock_data"
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

mode = st.sidebar.radio("Mode", ["Single Stock", "Portfolio"])
st.sidebar.title("Simulation Inputs")

def load_price_data(file_name):
    df = pd.read_csv(os.path.join(DATA_DIR, file_name))
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df[['Close']]

if mode == "Single Stock":
    selected_file = st.sidebar.selectbox("Select Stock CSV File", csv_files)
    num_simulations = int(st.sidebar.text_input("Number of Simulations", value="100"))
    num_of_days = int(st.sidebar.text_input("Number of Future Days", value="252"))

    price_data = load_price_data(selected_file)

    if price_data.empty:
        st.error("No data in selected CSV file.")
    else:
        prices = price_data['Close']
        returns = np.log(prices / prices.shift(1)).dropna()
        mu = returns.mean()
        sigma = returns.std()
        dt = 1
        initial_price = prices.iloc[0]

        price_paths = []
        for _ in range(num_simulations):
            path = [initial_price]
            for _ in range(1, num_of_days):
                epsilon = np.random.normal()
                next_price = path[-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * epsilon * np.sqrt(dt))
                path.append(next_price)
            price_paths.append(path)

        expected_price = np.mean([path[-1] for path in price_paths])
        expected_price_theoretical = initial_price * np.exp((mu - 0.5 * sigma**2) * num_of_days)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Geometric Brownian Motion Simulation (Single Stock)")
            st.markdown(r"""
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
                """)
        with col2:
            st.subheader("Monte Carlo Simulation Plot")
            fig, ax = plt.subplots(figsize=(10, 5))
            for path in price_paths:
                ax.plot(path)
            ax.set_title(f'{num_simulations} Simulations for {selected_file} ({num_of_days} Days)')
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
            st.write(f"Theoretical Price: ${expected_price_theoretical:.2f}")
        with col4:
            st.subheader("1-Year Price History")
            st.line_chart(prices)

else:
    selected_files = st.sidebar.multiselect("Select Stock CSV Files", csv_files, default=csv_files[:2])
    weights_input = st.sidebar.text_input("Weights (comma-separated, must sum to 1)", value="0.5, 0.5")

    weights = list(map(float, weights_input.split(',')))
    if len(weights) != len(selected_files) or not np.isclose(sum(weights), 1.0):
        st.error("Weights must match number of stocks and sum to 1.")
        st.stop()

    num_simulations = int(st.sidebar.text_input("Number of Simulations", value="100"))
    num_of_days = int(st.sidebar.text_input("Number of Future Days", value="252"))

    price_paths = []
    mus, sigmas, initials = [], [], []

    for file, w in zip(selected_files, weights):
        price_data = load_price_data(file)['Close']
        if price_data.empty:
            st.error(f"No data in {file}")
            st.stop()

        returns = np.log(price_data / price_data.shift(1)).dropna()
        mu = returns.mean()
        sigma = returns.std()
        initial_price = price_data.iloc[0]

        mus.append(mu)
        sigmas.append(sigma)
        initials.append(initial_price)

        simulated = []
        for _ in range(num_simulations):
            path = [initial_price]
            for _ in range(1, num_of_days):
                epsilon = np.random.normal()
                next_price = path[-1] * np.exp((mu - 0.5 * sigma ** 2) + sigma * epsilon)
                path.append(next_price)
            simulated.append(path)

        weighted_paths = np.array(simulated) * w
        price_paths.append(weighted_paths)

    portfolio_paths = np.sum(price_paths, axis=0)
    final_prices = portfolio_paths[:, -1]
    expected_price = np.mean(final_prices)

    mu_port = np.dot(mus, weights)
    sigma_port = np.sqrt(np.dot(np.square(sigmas), np.square(weights)))
    initial_port_price = np.dot(initials, weights)
    expected_theoretical = initial_port_price * np.exp((mu_port - 0.5 * sigma_port**2) * num_of_days)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Geometric Brownian Motion Simulation (Portfolio)")
        st.markdown(r"""
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
            """)
    with col2:
        st.subheader("Monte Carlo Simulation Plot")
        fig, ax = plt.subplots(figsize=(10, 5))
        for path in portfolio_paths:
            ax.plot(path)
        ax.set_title(f'{num_simulations} Simulations for Portfolio')
        ax.set_xlabel('Time Step (Days)')
        ax.set_ylabel('Portfolio Value ($)')
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Return Statistics")
        st.write(f"Portfolio μ: {mu_port:.5f}")
        st.write(f"Portfolio σ: {sigma_port:.5f}")
        st.write(f"Initial Portfolio Price: ${initial_port_price:.2f}")
        st.write(f"Expected Price (in {num_of_days} days): ${expected_price:.2f}")
        st.write(f"Theoretical Price: ${expected_theoretical:.2f}")
    with col4:
        st.subheader(f"Distribution of Portfolio Final Prices")
        fig, ax = plt.subplots(figsize=(6, 2.5))
        kde = gaussian_kde(final_prices)
        x_vals = np.linspace(min(final_prices), max(final_prices), 500)
        ax.plot(x_vals, kde(x_vals), color='black')
        ax.axvline(np.mean(final_prices), color='grey', linestyle='--', label='Expected Price')
        ax.set_xlabel("Portfolio Value ($)")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)
