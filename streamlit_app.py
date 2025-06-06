# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import traceback

# --- Função para Obter Dados de Ações (CORREÇÃO FINAL) ---
@st.cache_data(ttl=3600) # Cache de 1 hora
def get_stock_data(ticker, start_date_str, end_date_str):
    """
    Busca dados históricos de ações, normalizando os nomes das colunas para evitar KeyErrors.
    """
    try:
        st.write(f"Buscando dados para {ticker} de {start_date_str} até {end_date_str}...")
        
        end_date_dt = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1)
        end_date_yf_str = end_date_dt.strftime("%Y-%m-%d")

        data = yf.download(ticker, start=start_date_str, end=end_date_yf_str, progress=False)

        if data.empty:
            st.warning(f"Nenhum dado encontrado para o ticker '{ticker}'. Verifique o código do ativo e o período.")
            return None

        # --- INÍCIO DA CORREÇÃO DEFINITIVA ---
        # Normaliza os nomes das colunas de forma robusta, lidando com strings e tuplas.
        new_columns = []
        for col in data.columns:
            # Se o nome for uma tupla (multi-index), usa o primeiro item.
            name = col[0] if isinstance(col, tuple) else col
            new_columns.append(str(name).title())
        data.columns = new_columns
        # --- FIM DA CORREÇÃO DEFINITIVA ---

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            st.error(f"Erro para o ticker '{ticker}'. Os dados recebidos não contêm as colunas necessárias: {missing_cols}.")
            st.info(f"Colunas recebidas (após normalização): {data.columns.tolist()}")
            return None
            
        df = data[required_cols].copy()
        df.index = df.index.date
        df.index.name = "Date"
        df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
        df["Volume"] = df["Volume"].fillna(0).astype(np.int64)

        if df.empty:
            st.warning(f"DataFrame vazio após processamento para {ticker}.")
            return None

        df.sort_index(inplace=True)
        st.success(f"Dados para {ticker} carregados com sucesso ({len(df)} pregões).")
        return df

    except Exception as e:
        st.error(f"Erro inesperado ao buscar dados para {ticker}: {e}")
        st.code(traceback.format_exc())
        return None

# --- Função de Cálculo do RSI --- 
def calculate_rsi(data, n=2):
    """Calcula o Índice de Força Relativa (RSI) usando Média Móvel Simples (SMA)."""
    if "Close" not in data.columns or len(data) < n + 1:
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
    return data

# --- Lógica do Backtest --- 
def run_ifr2_backtest(ticker, data, oversold_level, target_days, time_stop_days, shares_per_trade):
    """Executa a simulação de backtest do IFR2 para um único ativo."""
    if f"RSI_2" not in data.columns or data.empty:
        return [], data

    trades = []
    in_position = False
    entry_price, entry_date, target_price = 0.0, None, 0.0
    days_in_trade = 0

    data['RSI_2_Prev'] = data['RSI_2'].shift(1)

    for i in range(target_days + 1, len(data)):
        current_date = data.index[i]
        current_open = data['Open'].iloc[i]
        current_high = data['High'].iloc[i]
        current_close = data['Close'].iloc[i]
        prev_rsi = data['RSI_2_Prev'].iloc[i]

        if in_position:
            days_in_trade += 1
            exit_price, exit_reason = None, None

            if current_high >= target_price:
                exit_price = target_price
                exit_reason = "Alvo"
            elif days_in_trade >= time_stop_days:
                if i + 1 < len(data):
                    exit_price = data['Open'].iloc[i+1]
                    exit_reason = "Tempo"
                else:
                    exit_price = current_close
                    exit_reason = "Tempo (Fim Dados)"

            if exit_reason:
                result_fin = (exit_price - entry_price) * shares_per_trade
                trades.append({
                    "Ticker": ticker, "Entry Date": entry_date, "Entry Price": entry_price,
                    "Exit Date": current_date if exit_reason == "Alvo" else data.index[i+1] if exit_reason == "Tempo" else current_date,
                    "Exit Price": exit_price, "Result Fin (R$)": result_fin,
                    "Exit Reason": exit_reason, "Days Held": days_in_trade
                })
                in_position = False

        if not in_position and not pd.isna(prev_rsi) and prev_rsi < oversold_level:
            target_calc_start_idx = max(0, i - 1 - target_days)
            target_calc_end_idx = i
            high_slice = data['High'].iloc[target_calc_start_idx:target_calc_end_idx]
            if not high_slice.empty:
                target_price = high_slice.max()
                entry_price = current_open
                entry_date = current_date
                in_position = True
                days_in_trade = 0
    return trades, data

# --- Interface do Usuário com Streamlit --- 
st.set_page_config(layout="wide")
st.title("Backtest IFR2 do Stormer")

st.sidebar.header("Configurações do Backtest")
tickers_input = st.sidebar.text_input("Ativo(s) (ex: PETR4.SA, VALE3.SA)", "PETR4.SA")
start_date_input = st.sidebar.date_input("Data de Início", datetime.now() - timedelta(days=365*5))
end_date_input = st.sidebar.date_input("Data de Fim", datetime.now())

st.sidebar.subheader("Parâmetros da Estratégia")
param_oversold = st.sidebar.slider("Nível Sobrevenda IFR(2)", 1, 50, 20)
param_target_days = st.sidebar.number_input("Dias para Alvo (Máx. X Dias)", 1, 10, 3)
param_time_stop = st.sidebar.number_input("Stop no Tempo (Dias)", 1, 20, 7)
param_shares = st.sidebar.number_input("Lote (Ações por Trade)", 1, 10000, 100)

if st.sidebar.button("Iniciar Backtest"):
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]
    if not tickers:
        st.warning("Por favor, insira pelo menos um ticker de ativo.")
    elif start_date_input >= end_date_input:
        st.warning("A data de início deve ser anterior à data de fim.")
    else:
        all_trades = []
        progress_bar = st.progress(0, "Iniciando...")
        for i, ticker in enumerate(tickers):
            progress_bar.progress((i) / len(tickers), f"Processando: {ticker}")
            stock_data = get_stock_data(ticker, start_date_input.strftime("%Y-%m-%d"), end_date_input.strftime("%Y-%m-%d"))
            if stock_data is not None:
                stock_data_rsi = calculate_rsi(stock_data, n=2)
                trades_list, _ = run_ifr2_backtest(
                    ticker, stock_data_rsi.copy(), param_oversold, param_target_days, param_time_stop, param_shares
                )
                if trades_list:
                    all_trades.extend(trades_list)
        progress_bar.progress(1.0, "Backtest concluído!")
        
        st.header("Resultados Consolidados")
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_df["Result (%)"] = ((trades_df["Exit Price"] - trades_df["Entry Price"]) / trades_df["Entry Price"]) * 100
            
            st.subheader("Todas as Operações")
            st.dataframe(trades_df.style.format({
                "Entry Price": "R$ {:.2f}", "Exit Price": "R$ {:.2f}", 
                "Result (%)": "{:.2f}%", "Result Fin (R$)": "R$ {:,.2f}"
            }))

            st.subheader("Métricas de Desempenho")
            total_trades = len(trades_df)
            winners = trades_df[trades_df["Result Fin (R$)"] > 0]
            win_rate = (len(winners) / total_trades) * 100 if total_trades > 0 else 0
            total_pnl = trades_df["Result Fin (R$)"].sum()
            avg_win = winners["Result Fin (R$)"].mean() if len(winners) > 0 else 0
            avg_loss = trades_df[trades_df["Result Fin (R$)"] <= 0]['Result Fin (R$)'].mean() if len(winners) < total_trades else 0
            payoff = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

            col1, col2, col3 = st.columns(3)
            col1.metric("Total de Trades", total_trades)
            col2.metric("Taxa de Acerto", f"{win_rate:.2f}%")
            col3.metric("Payoff Ratio", f"{payoff:.2f}")

            st.metric("Lucro/Prejuízo Total", f"R$ {total_pnl:,.2f}")

            # Curva de Capital
            trades_df = trades_df.sort_values(by="Exit Date")
            trades_df['Cumulative PnL'] = trades_df['Result Fin (R$)'].cumsum()
            st.subheader("Curva de Capital")
            st.line_chart(trades_df.set_index('Exit Date')['Cumulative PnL'])
        else:
            st.info("Nenhuma operação foi executada para os ativos e parâmetros fornecidos.")
else:
    st.info("Ajuste os parâmetros na barra lateral e clique em 'Iniciar Backtest' para começar.")
