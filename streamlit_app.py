# /home/ubuntu/ifr2_backtester/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Import functions from the original app.py (or copy/adapt them here)
# We will copy/adapt the necessary functions to keep this self-contained for now
import sys
sys.path.append("/opt/.manus/.sandbox-runtime")
from data_api import ApiClient

# Initialize API client (should be done once)
client = ApiClient()

# --- Copy/Adapt Functions from app.py --- 

def get_stock_data(ticker, start_date_str, end_date_str):
    """
    Fetches historical stock data using the YahooFinance API.
    (Copied and adapted from app.py - Consider refactoring later)
    """
    try:
        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1)
        period1 = str(int(time.mktime(start_dt.timetuple())))
        period2 = str(int(time.mktime(end_dt.timetuple())))

        st.write(f"Buscando dados para {ticker} de {start_date_str} até {end_date_str}...")
        region = "BR" if ticker.endswith(".SA") else "US"

        api_response = client.call_api(
            "YahooFinance/get_stock_chart",
            query={
                "symbol": ticker,
                "region": region,
                "interval": "1d",
                "period1": period1,
                "period2": period2,
                "includeAdjustedClose": "false"
            }
        )

        if not api_response or api_response.get("chart", {}).get("error"):
            error_msg = api_response.get("chart", {}).get("error", "Unknown API error")
            if isinstance(error_msg, dict) and "description" in error_msg:
                 st.error(f"Erro ao buscar dados para {ticker}: {error_msg[	'description	']}")
            else:
                 st.error(f"Erro ao buscar dados para {ticker}: {error_msg}")
            return None

        result = api_response.get("chart", {}).get("result", [])
        if not result or result[0] is None:
            st.warning(f"Nenhum dado encontrado para {ticker} no período especificado.")
            return None

        data = result[0]
        timestamps = data.get("timestamp", [])
        indicators = data.get("indicators", {}).get("quote", [{}])[0]

        if not timestamps or not indicators.get("open") or not indicators.get("close") or \
           not indicators.get("high") or not indicators.get("low") or not indicators.get("volume"):
             st.error(f"Dados incompletos recebidos para {ticker}.")
             return None

        required_keys = ["open", "high", "low", "close", "volume"]
        data_dict = {"Date": pd.to_datetime(timestamps, unit="s").date}
        min_len = len(timestamps)
        valid_data = True
        for key in required_keys:
            indicator_list = indicators.get(key, [])
            if key != "volume" and (None in indicator_list or len(indicator_list) != min_len):
                st.error(f"Dados inválidos ou comprimento incorreto para 	\'{key}\t' em {ticker}.")
                valid_data = False
                break
            elif key == "volume" and len(indicator_list) != min_len:
                 st.error(f"Comprimento incorreto para 	\'{key}\t' em {ticker}.")
                 valid_data = False
                 break
            data_dict[key.capitalize()] = [0 if v is None else v for v in indicator_list] if key == "volume" else indicator_list

        if not valid_data:
            st.error(f"Pulando {ticker} devido a inconsistências nos dados.")
            return None

        df = pd.DataFrame(data_dict)
        df.set_index("Date", inplace=True)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

        if df.empty:
            st.warning(f"DataFrame vazio após processamento para {ticker}.")
            return None

        df.sort_index(inplace=True)
        st.success(f"Dados para {ticker} carregados com sucesso ({len(df)} pontos).")
        return df

    except Exception as e:
        st.error(f"Erro inesperado ao buscar dados para {ticker}: {e}")
        return None

def calculate_rsi(data, n=2):
    """
    Calculates the Relative Strength Index (RSI) using Simple Moving Average (SMA).
    (Copied and adapted from app.py)
    """
    if "Close" not in data.columns:
        st.error("DataFrame precisa conter a coluna 'Close'.")
        return data # Return original data
    if len(data) < n + 1:
        st.warning(f"Pontos de dados insuficientes ({len(data)}) para calcular RSI({n}).")
        data[f"RSI_{n}"] = np.nan
        return data

    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=n, min_periods=n).mean()
    avg_loss = loss.rolling(window=n, min_periods=n).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100
    rsi[(avg_gain == 0) & (avg_loss == 0)] = np.nan
    data[f"RSI_{n}"] = rsi
    # st.write(f"RSI_{n} calculado.") # Optional: less verbose
    return data

def run_ifr2_backtest(ticker, data, oversold_level=10, target_days=3, time_stop_days=7, shares_per_trade=100):
    """
    Runs the IFR2 backtest simulation for a single ticker.
    (Copied and adapted from app.py)
    """
    if f"RSI_2" not in data.columns:
        st.error("DataFrame precisa conter a coluna 'RSI_2'.")
        return [], data
    if data.empty:
        st.warning("Dados de entrada vazios. Não é possível executar o backtest.")
        return [], data

    trades = []
    in_position = False
    entry_price = 0.0
    entry_date = None
    target_price = 0.0
    exit_reason = None
    days_in_trade = 0

    data['Signal'] = 0
    data['Position'] = 0
    data['RSI_2_Prev'] = data['RSI_2'].shift(1)

    # Use st.progress for visual feedback if the loop takes time
    progress_bar = st.progress(0)
    total_steps = len(data)

    # st.write(f"Iniciando backtest para {ticker}...") # Optional
    for i in range(target_days + 1, len(data)):
        # Update progress bar
        progress_bar.progress(i / total_steps)
        
        current_date = data.index[i]
        current_open = data['Open'].iloc[i]
        current_high = data['High'].iloc[i]
        current_low = data['Low'].iloc[i]
        current_close = data['Close'].iloc[i]
        prev_rsi = data['RSI_2_Prev'].iloc[i]

        # --- Exit Logic ---
        if in_position:
            days_in_trade += 1
            data.loc[data.index[i], 'Position'] = 1

            exit_price = None # Initialize exit_price for the day
            if current_high >= target_price:
                exit_price = target_price
                exit_date = current_date
                exit_reason = "Alvo"
            elif days_in_trade >= time_stop_days:
                if i + 1 < len(data):
                    exit_price = data['Open'].iloc[i+1]
                    exit_date = data.index[i+1]
                    exit_reason = "Tempo"
                else:
                    exit_price = current_close
                    exit_date = current_date
                    exit_reason = "Tempo (Fim Dados)"

            if exit_reason:
                result_pct = ((exit_price - entry_price) / entry_price) * 100
                result_fin = (exit_price - entry_price) * shares_per_trade
                trades.append({
                    "Ticker": ticker,
                    "Entry Date": entry_date,
                    "Entry Price": entry_price,
                    "Exit Date": exit_date,
                    "Exit Price": exit_price,
                    "Result (%)": result_pct,
                    "Result Fin (R$)": result_fin,
                    "Exit Reason": exit_reason,
                    "Days Held": days_in_trade
                })
                in_position = False
                exit_reason = None
                days_in_trade = 0
                # Continue to potentially enter on the same day if a signal exists?
                # Stormer's original logic might suggest waiting a day after exit.
                # For now, we allow immediate re-entry check. 
                # If exit was time stop at next open, skip next day's entry check?
                if exit_reason == "Tempo" and exit_date == data.index[i+1]:
                     # If we exited at tomorrow's open, skip entry check for tomorrow
                     # This requires adjusting the loop or adding a flag. Simpler: don't skip for now.
                     pass 
                else: # If exited today (target hit or end of data)
                    continue # Skip entry logic for *today*

        # --- Entry Logic ---
        if not in_position and not pd.isna(prev_rsi):
            if prev_rsi < oversold_level:
                target_calc_start_idx = i - 1 - target_days
                target_calc_end_idx = i # Use i for exclusive slicing to get up to i-1

                if target_calc_start_idx >= 0:
                    # Ensure the slice is valid and has data
                    high_slice = data['High'].iloc[target_calc_start_idx:target_calc_end_idx]
                    if not high_slice.empty:
                        target_price = high_slice.max()
                        entry_price = current_open
                        entry_date = current_date
                        in_position = True
                        days_in_trade = 0
                        data.loc[data.index[i], 'Signal'] = 1
                        data.loc[data.index[i], 'Position'] = 1
                    # else: st.warning(f"Slice vazia para cálculo do alvo em {current_date}") # Debug
                # else: st.warning(f"Índice inicial inválido para alvo em {current_date}") # Debug

    progress_bar.progress(1.0) # Complete the progress bar
    # st.write(f"Backtest para {ticker} concluído.") # Optional
    if not trades:
        st.info(f"Nenhuma operação executada para {ticker} com os parâmetros fornecidos.")

    return trades, data

# --- Streamlit UI --- 

st.title("Backtest IFR2 do Stormer")

st.sidebar.header("Configurações do Backtest")

# Inputs na sidebar
tickers_input = st.sidebar.text_input("Ativo(s) (ex: PETR4.SA, VALE3.SA)", "PETR4.SA")
start_date_input = st.sidebar.date_input("Data de Início", datetime.now() - timedelta(days=365*5))
end_date_input = st.sidebar.date_input("Data de Fim", datetime.now())

st.sidebar.subheader("Parâmetros da Estratégia")
param_oversold = st.sidebar.number_input("Nível Sobrevenda IFR(2)", min_value=1, max_value=50, value=10, step=1)
param_target_days = st.sidebar.number_input("Dias para Alvo (Máx. X Dias)", min_value=1, max_value=10, value=3, step=1)
param_time_stop = st.sidebar.number_input("Stop no Tempo (Dias)", min_value=1, max_value=20, value=7, step=1)
param_shares = st.sidebar.number_input("Lote (Ações por Trade)", min_value=1, value=100, step=1)

# Botão para iniciar
run_button = st.sidebar.button("Iniciar Backtest")

# --- Lógica Principal --- 
if run_button:
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]
    start_date_str = start_date_input.strftime("%Y-%m-%d")
    end_date_str = end_date_input.strftime("%Y-%m-%d")

    if not tickers:
        st.warning("Por favor, insira pelo menos um ticker de ativo.")
    elif start_date_input >= end_date_input:
        st.warning("A data de início deve ser anterior à data de fim.")
    else:
        st.header("Resultados do Backtest")
        all_trades = []
        
        for ticker in tickers:
            st.subheader(f"Processando: {ticker}")
            # 1. Obter Dados
            stock_data = get_stock_data(ticker, start_date_str, end_date_str)

            if stock_data is not None and not stock_data.empty:
                # 2. Calcular IFR
                stock_data = calculate_rsi(stock_data, n=2)
                
                # 3. Executar Backtest
                trades_list, data_with_signals = run_ifr2_backtest(
                    ticker=ticker,
                    data=stock_data.copy(),
                    oversold_level=param_oversold,
                    target_days=param_target_days,
                    time_stop_days=param_time_stop,
                    shares_per_trade=param_shares
                )
                
                if trades_list:
                    all_trades.extend(trades_list)
                # else: st.info(f"Nenhuma operação para {ticker}.") # Already handled inside function
            else:
                st.error(f"Não foi possível obter ou processar dados para {ticker}. Pulando...")

        # 4. Exibir Resultados Consolidados
        if all_trades:
            st.subheader("Todas as Operações")
            trades_df = pd.DataFrame(all_trades)
            # Format dates for display
            trades_df["Entry Date"] = pd.to_datetime(trades_df["Entry Date"]).dt.strftime('%Y-%m-%d')
            trades_df["Exit Date"] = pd.to_datetime(trades_df["Exit Date"]).dt.strftime('%Y-%m-%d')
            st.dataframe(trades_df.style.format({
                "Entry Price": "{:.2f}", 
                "Exit Price": "{:.2f}", 
                "Result (%)": "{:.2f}%", 
                "Result Fin (R$)": "R$ {:.2f}"
            }))

            st.subheader("Métricas de Desempenho Consolidadas")
            total_trades = len(trades_df)
            winners = trades_df[trades_df["Result Fin (R$)"] > 0]
            losers = trades_df[trades_df["Result Fin (R$)"] <= 0]
            num_winners = len(winners)
            num_losers = len(losers)
            win_rate = (num_winners / total_trades) * 100 if total_trades > 0 else 0
            total_profit_loss = trades_df["Result Fin (R$)"].sum()
            avg_profit_loss_trade = trades_df["Result Fin (R$)"].mean() if total_trades > 0 else 0
            avg_win = winners["Result Fin (R$)"].mean() if num_winners > 0 else 0
            avg_loss = losers["Result Fin (R$)"].mean() if num_losers > 0 else 0
            payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

            col1, col2, col3 = st.columns(3)
            col1.metric("Total de Trades", total_trades)
            col2.metric("Trades Vencedores", num_winners)
            col3.metric("Trades Perdedores", num_losers)
            
            col1b, col2b, col3b = st.columns(3)
            col1b.metric("Taxa de Acerto", f"{win_rate:.2f}%")
            col2b.metric("Payoff Ratio", f"{payoff_ratio:.2f}")
            col3b.metric("Lucro/Prejuízo Total", f"R$ {total_profit_loss:.2f}")
            
            st.write(f"Lucro/Prejuízo Médio por Trade: R$ {avg_profit_loss_trade:.2f}")
            st.write(f"Ganho Médio (Trades Vencedores): R$ {avg_win:.2f}")
            st.write(f"Perda Média (Trades Perdedores): R$ {avg_loss:.2f}")
            
            # Opcional: Gráfico de Curva de Capital (Simplificado)
            # Requer calcular o capital acumulado
            try:
                trades_df_sorted = trades_df.sort_values(by="Exit Date").copy()
                trades_df_sorted['Cumulative PnL'] = trades_df_sorted['Result Fin (R$)' ].cumsum()
                # Adicionar ponto inicial (capital 0 antes do primeiro trade)
                start_point = pd.DataFrame([{'Exit Date': start_date_str, 'Cumulative PnL': 0}])
                # Convert Exit Date to datetime for plotting if needed
                trades_df_sorted['Exit Date'] = pd.to_datetime(trades_df_sorted['Exit Date'])
                start_point['Exit Date'] = pd.to_datetime(start_point['Exit Date'])
                
                # Combine start point with trades
                capital_curve_data = pd.concat([start_point, trades_df_sorted[['Exit Date', 'Cumulative PnL']]], ignore_index=True)
                
                st.subheader("Curva de Capital (Simplificada)")
                st.line_chart(capital_curve_data.set_index('Exit Date')['Cumulative PnL'])
            except Exception as e:
                st.warning(f"Não foi possível gerar o gráfico da curva de capital: {e}")

        else:
            st.info("Nenhuma operação foi executada para os ativos e parâmetros fornecidos no período.")
else:
    st.info("Ajuste os parâmetros na barra lateral e clique em 'Iniciar Backtest'.")


