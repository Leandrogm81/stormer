# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import traceback

# --- Função para Obter Dados de Ações ---
@st.cache_data(ttl=3600)
def get_stock_data(ticker, start_date_str, end_date_str):
    """
    Busca dados históricos de ações, normalizando os nomes das colunas para evitar KeyErrors.
    """
    try:
        st.write(f"Buscando dados para {ticker} de {start_date_str} até {end_date_str}...")
        
        end_date_dt = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1)
        end_date_yf_str = end_date_dt.strftime("%Y-%m-%d")

        data = yf.download(ticker, start=start_date_str, end=end_date_yf_str, progress=False, auto_adjust=True)

        if data.empty:
            st.warning(f"Nenhum dado encontrado para o ticker '{ticker}'.")
            return None

        # Normaliza os nomes das colunas de forma robusta
        new_columns = []
        for col in data.columns:
            name = col[0] if isinstance(col, tuple) else col
            new_columns.append(str(name).title())
        data.columns = new_columns

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in data.columns for col in required_cols):
            st.error(f"Erro para '{ticker}'. Dados recebidos não contêm as colunas necessárias.")
            st.info(f"Colunas recebidas: {data.columns.tolist()}")
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
    rsi[rsi.isna()] = 50 # Evita NaNs iniciais
    
    data[f"RSI_{n}"] = rsi
    return data

# --- Lógica do Backtest --- 
def run_ifr2_backtest(ticker, data, oversold_level, target_days, time_stop_days, shares_per_trade):
    """Executa a simulação de backtest do IFR2 para um único ativo."""
    if f"RSI_2" not in data.columns or data.empty:
        return []

    trades = []
    in_position = False
    entry_price, entry_date, target_price = 0.0, None, 0.0
    days_in_trade = 0

    data['RSI_2_Prev'] = data['RSI_2'].shift(1)

    for i in range(1, len(data)): # Inicia em 1 por causa do shift
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
                exit_price = current_open
                exit_reason = "Tempo"

            if exit_reason:
                result_fin = (exit_price - entry_price) * shares_per_trade
                trades.append({
                    "Ticker": ticker, "Entry Date": entry_date, "Entry Price": entry_price,
                    "Exit Date": current_date, "Exit Price": exit_price, 
                    "Result Fin (R$)": result_fin, "Exit Reason": exit_reason, 
                    "Days Held": days_in_trade
                })
                in_position = False

        if not in_position and not pd.isna(prev_rsi) and prev_rsi < oversold_level:
            target_calc_start_idx = max(0, i - target_days)
            target_calc_end_idx = i
            high_slice = data['High'].iloc[target_calc_start_idx:target_calc_end_idx]
            if not high_slice.empty:
                target_price = high_slice.max()
                entry_price = current_open
                entry_date = current_date
                in_position = True
                days_in_trade = 0
    return trades

# --- Interface do Usuário com Streamlit --- 
st.set_page_config(layout="wide")
st.title("Backtest IFR2 do Stormer")

st.sidebar.header("Configurações do Backtest")
tickers_input = st.sidebar.text_input("Ativo(s) (ex: PETR4.SA, VALE3.SA)", "PETR4.SA, MGLU3.SA")
start_date_input = st.sidebar.date_input("Data de Início", datetime.now() - timedelta(days=365*5))
end_date_input = st.sidebar.date_input("Data de Fim", datetime.now())

st.sidebar.subheader("Parâmetros da Estratégia")
param_oversold = st.sidebar.slider("Nível Sobrevenda IFR(2)", 1, 50, 20, help="O RSI(2) deve estar abaixo deste nível para gerar um sinal de compra.")
param_target_days = st.sidebar.number_input("Dias para Alvo (Máx. X Dias)", 1, 10, 3, help="O alvo será a máxima dos últimos X dias antes da entrada.")
param_time_stop = st.sidebar.number_input("Stop no Tempo (Dias)", 1, 20, 7, help="Se o alvo não for atingido, a operação é encerrada após X dias.")
param_shares = st.sidebar.number_input("Lote (Ações por Trade)", 1, 10000, 100)

if st.sidebar.button("Iniciar Backtest"):
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]
    if not tickers:
        st.warning("Por favor, insira pelo menos um ticker de ativo.")
    elif start_date_input >= end_date_input:
        st.warning("A data de início deve ser anterior à data de fim.")
    else:
        all_trades = []
        with st.spinner('Executando backtest... Por favor, aguarde.'):
            for ticker in tickers:
                stock_data = get_stock_data(ticker, start_date_input.strftime("%Y-%m-%d"), end_date_input.strftime("%Y-%m-%d"))
                if stock_data is not None:
                    stock_data_rsi = calculate_rsi(stock_data, n=2)
                    trades_list = run_ifr2_backtest(
                        ticker, stock_data_rsi.copy(), param_oversold, param_target_days, param_time_stop, param_shares
                    )
                    if trades_list:
                        all_trades.extend(trades_list)
        
        st.header("Resultados Consolidados")
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_df['Entry Date'] = pd.to_datetime(trades_df['Entry Date'])
            trades_df['Exit Date'] = pd.to_datetime(trades_df['Exit Date'])
            trades_df = trades_df.sort_values(by="Exit Date")
            trades_df["Result (%)"] = ((trades_df["Exit Price"] - trades_df["Entry Price"]) / trades_df["Entry Price"]) * 100
            
            # --- NOVAS MÉTRICAS ---
            total_trades = len(trades_df)
            winners = trades_df[trades_df["Result Fin (R$)"] > 0]
            losers = trades_df[trades_df["Result Fin (R$)"] <= 0]
            
            win_rate = (len(winners) / total_trades) * 100 if total_trades > 0 else 0
            total_pnl = trades_df["Result Fin (R$)"].sum()
            avg_win = winners["Result Fin (R$)"].mean() if len(winners) > 0 else 0
            avg_loss = losers["Result Fin (R$)"].mean() if len(losers) > 0 else 0
            payoff = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            avg_days = trades_df["Days Held"].mean()
            max_profit = trades_df["Result Fin (R$)"].max()
            max_loss = trades_df["Result Fin (R$)"].min()

            # Cálculo do Drawdown
            initial_capital = 50000 # Capital inicial hipotético para o cálculo
            trades_df['Cumulative PnL'] = trades_df['Result Fin (R$)'].cumsum()
            trades_df['Capital'] = initial_capital + trades_df['Cumulative PnL']
            trades_df['Peak'] = trades_df['Capital'].cummax()
            trades_df['Drawdown Pct'] = ((trades_df['Capital'] - trades_df['Peak']) / trades_df['Peak']) * 100
            max_drawdown = trades_df['Drawdown Pct'].min() if not trades_df['Drawdown Pct'].empty else 0

            # --- NOVO CÁLCULO ---
            return_pct = (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0

            # --- NOVA ESTRUTURA DE EXIBIÇÃO ---
            st.subheader("Métricas de Desempenho")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Lucro/Prejuízo Total", f"R$ {total_pnl:,.2f}")
            col2.metric("Retorno sobre Capital", f"{return_pct:.2f}%")
            col3.metric("Total de Trades", f"{total_trades}")
            col4.metric("Taxa de Acerto", f"{win_rate:.2f}%")
            
            col5, col6, col7 = st.columns(3)
            col5.metric("Payoff Ratio", f"{payoff:.2f}")
            col6.metric("Média de Dias/Trade", f"{avg_days:.1f}")
            col7.metric("Drawdown Máximo", f"{max_drawdown:.2f}%")
            
            st.write("---")
            
            col8, col9 = st.columns(2)
            with col8:
                st.subheader("Performance dos Ganhos")
                st.metric("Média de Ganho", f"R$ {avg_win:,.2f}", delta_color="normal")
                st.metric("Maior Lucro", f"R$ {max_profit:,.2f}", delta_color="normal")
            with col9:
                st.subheader("Performance das Perdas")
                st.metric("Média de Perda", f"R$ {avg_loss:,.2f}", delta_color="inverse")
                st.metric("Maior Prejuízo", f"R$ {max_loss:,.2f}", delta_color="inverse")
            
            st.write("---")
            
            st.subheader("Evolução do Capital")
            st.line_chart(trades_df.set_index('Exit Date')['Capital'])
            
            st.subheader("Curva de Drawdown (%)")
            st.area_chart(trades_df.set_index('Exit Date')['Drawdown Pct'])

            st.subheader("Tabela de Operações")
            st.dataframe(trades_df[['Ticker', 'Entry Date', 'Entry Price', 'Exit Date', 'Exit Price', 'Result (%)', 'Result Fin (R$)', 'Exit Reason', 'Days Held']].style.format({
                "Entry Price": "R$ {:.2f}", "Exit Price": "R$ {:.2f}", 
                "Result (%)": "{:.2f}%", "Result Fin (R$)": "R$ {:,.2f}"
            }))
        else:
            st.info("Nenhuma operação foi executada para os ativos e parâmetros fornecidos.")
else:
    st.info("Ajuste os parâmetros na barra lateral e clique em 'Iniciar Backtest' para começar.")
