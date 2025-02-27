import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Strategy parameters
params = {
    'lookback_days': 5,
    'vol_ma_days': 7,
    'hold_days': 3,
    'rsi_overbought': 65
}

def calculate_rsi(close_prices, window=14):
    """Calculate RSI using Wilder's smoothing (matches TradingView/pandas-ta)"""
    delta = close_prices.diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    # Calculate initial averages
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    # Wilder's smoothing: subsequent averages
    for i in range(window, len(gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (window-1) + gain.iloc[i]) / window
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (window-1) + loss.iloc[i]) / window
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(close_prices, window=20):
    """Custom EMA implementation"""
    sma = close_prices.rolling(window=window).mean()[:window]
    if len(sma) == 0:
        return pd.Series([np.nan]*len(close_prices), index=close_prices.index)
    
    ema = [sma.iloc[-1]]
    multiplier = 2 / (window + 1)
    
    for price in close_prices.iloc[window:]:
        ema_val = (price - ema[-1]) * multiplier + ema[-1]
        ema.append(ema_val)
    
    return pd.Series([np.nan]*window + ema[1:], index=close_prices.index)

def implement_strategy(data):
    # Calculate indicators
    data['rolling_high'] = data['High'].shift(1).rolling(params['lookback_days']).max()
    data['vol_ma'] = data['Volume'].rolling(params['vol_ma_days']).mean().shift(1)
    data['rsi'] = calculate_rsi(data['Close'], 14).shift(1)
    # data['ema_50'] = calculate_ema(data['Close'], 20).shift(1)  # Trend filter

    # Generate signals
    data['buy_signal'] = (
        (data['Close'] > data['rolling_high']) &
        (data['Volume'] > data['vol_ma'] * 1) &
        (data['rsi'] < params['rsi_overbought'])
        # (data['Close'] > data['ema_50'])
    ).astype(int)
    return data

def run_backtest(df):
    results = []
    active_positions = []
    
    for date, row in df.iterrows():
        # Close positions first
        for pos in active_positions.copy():
            days_held = (date - pos['entry_date']).days
            current_price = row['Close']
            
            # Dynamic exit conditions
            if pos['type'] == 'Long':
                return_pct = (current_price - pos['entry_price']) / pos['entry_price']
                stop_loss = pos['entry_price'] * 0.96
                take_profit = pos['entry_price'] * 1.25
            else:
                return_pct = (pos['entry_price'] - current_price) / pos['entry_price']
                stop_loss = pos['entry_price'] * 1.03
                take_profit = pos['entry_price'] * 0.85
                
            # Exit reasons
            if days_held >= params['hold_days']:
                exit_reason = 'Time Exit'
            elif current_price <= stop_loss and pos['type'] == 'Long':
                exit_reason = 'Stop Loss'
            elif current_price >= stop_loss and pos['type'] == 'Short':
                exit_reason = 'Stop Loss'
            elif current_price >= take_profit and pos['type'] == 'Long':
                exit_reason = 'Take Profit'
            elif current_price <= take_profit and pos['type'] == 'Short':
                exit_reason = 'Take Profit'
            else:
                continue
                
            results.append({
                'Entry Date': pos['entry_date'],
                'Exit Date': date,
                'Entry Price': pos['entry_price'],
                'Exit Price': current_price,
                'Return': return_pct,
                'Days Held': days_held,
            })
            active_positions.remove(pos)
        
        # Enter new positions
        if row['buy_signal'] and len(active_positions) < 2:  # Max 2 positions
            active_positions.append({
                'type': 'Long',
                'entry_date': date,
                'entry_price': row['Close']
            })
    return pd.DataFrame(results)

def fetch_data(ticker, start_date, end_date):
    # Get extra data for EMA calculation
    adjusted_start = start_date - timedelta(days=30)
    attempts = 0
    while attempts < 3:
        try:
            data = yf.download(ticker, start=adjusted_start, end=end_date)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
                # Select and reorder columns
                data = data[['Close', 'High', 'Low', 'Open', 'Volume']]
                # Convert index to proper datetime format
                data.index = pd.to_datetime(data.index)
                return data
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            attempts += 1
    return pd.DataFrame()

def main():
    st.title("Stock Breakout Strategy Backtester")
    
    # Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Stock Ticker", "BTC-USD")
    with col2:
        start_date = st.date_input("Start Date", datetime(2023, 1, 1))
    with col3:
        end_date = st.date_input("End Date", datetime(2023, 12, 31))
    
    if st.button("Run Backtest"):
        with st.spinner("Fetching data and running backtest..."):
            # Fetch data with buffer
            data = fetch_data(ticker, start_date, end_date)
            if data.empty:
                st.error("Failed to fetch data. Please try again.")
                return
            
            # Implement strategy
            strategy_data = implement_strategy(data.copy()).dropna()
            
            # Run backtest
            results = run_backtest(strategy_data.loc[start_date:end_date])
            
            # Create main price plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=strategy_data.index, y=strategy_data['Close'], 
                                   name='Price', line=dict(color='#1f77b4')))
            
            # Add buy signals
            buy_dates = strategy_data[strategy_data['buy_signal'] == 1].index
            fig.add_trace(go.Scatter(
                x=buy_dates,
                y=strategy_data.loc[buy_dates, 'Close'],
                mode='markers',
                name='Buy Signal',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ))
            
            # Add exit points if any trades occurred
            if not results.empty:
                exit_dates = results['Exit Date']
                fig.add_trace(go.Scatter(
                    x=exit_dates,
                    y=results['Exit Price'],
                    mode='markers',
                    name='Exit Point',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ))
            
            fig.update_layout(title=f"{ticker} Price with Signals", 
                            xaxis_title='Date', yaxis_title='Price')
            st.plotly_chart(fig)
            
            if results.empty:
                st.warning("No trades generated during this period")
                return
            
            # Calculate metrics
            total_trades = len(results)
            win_rate = (results['Return'] > 0).mean() * 100
            avg_return = results['Return'].mean() * 100
            avg_days_held = results['Days Held'].mean()

            # Profit Factor
            winners = results[results['Return'] > 0]
            losers = results[results['Return'] < 0]
            total_gains = winners['Return'].sum()
            total_losses = abs(losers['Return'].sum())
            profit_factor = total_gains / total_losses if total_losses != 0 else float('inf')
            
            
            # Display metrics
            st.subheader("Performance Metrics")
            cols = st.columns(5)  # Changed from 4 to 5 columns
            cols[0].metric("Total Trades", total_trades)
            cols[1].metric("Win Rate", f"{win_rate:.1f}%")
            cols[2].metric("Avg Return", f"{avg_return:.1f}%")
            cols[3].metric("Profit Factor", f"{profit_factor:.2f}")
            cols[4].metric("Avg Days Held", f"{avg_days_held:.1f}")

            
            # Show raw trades
            st.subheader("Trade Details")
            st.dataframe(results.sort_values('Entry Date', ascending=False))

if __name__ == "__main__":
    main()
