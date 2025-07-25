import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import datetime
import json
import os

# Configurazione della pagina
st.set_page_config(
    page_title="Analisi Fondamentale Azioni",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  # Sidebar nascosta per default
)

# Funzioni per recuperare dati da Yahoo Finance (per le informazioni base del titolo)

@st.cache_data
def fetch_stock_info_yahoo(symbol):
    """Recupera informazioni di base sul titolo da Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Recupera anche i dati di prezzo più recenti
        hist = stock.history(period='1d')
        
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            prev_close = info.get('previousClose', hist['Close'].iloc[-1])
            
            # Calcola la variazione percentuale
            if prev_close and prev_close != 0:
                percent_change = ((current_price - prev_close) / prev_close) * 100
            else:
                percent_change = 0
                
            # Aggiorna le informazioni con i dati più recenti
            info['currentPrice'] = current_price
            info['regularMarketChangePercent'] = percent_change
        
        return info
    except Exception as e:
        st.error(f"Errore nel recupero delle informazioni sul titolo da Yahoo Finance: {str(e)}")
        return {}

# Funzioni per recuperare dati da FinancialDatasets AI

@st.cache_data
def fetch_financial_metrics(symbol, api_key):
    """Recupera metriche finanziarie aggiuntive da FinancialDatasets AI"""
    url = f"https://api.financialdatasets.ai/financial-metrics/snapshot?ticker={symbol}"
    headers = {
        "X-API-KEY": api_key
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json().get('snapshot', {})
        return data
    except Exception as e:
        st.warning(f"Impossibile recuperare metriche aggiuntive da FinancialDatasets AI: {str(e)}")
        return {}

@st.cache_data
def fetch_price_history_yahoo(symbol):
    """Recupera lo storico dei prezzi da Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        # Recupera i dati settimanali dell'ultimo anno
        hist = stock.history(period='1y', interval='1wk')
        return hist
    except Exception as e:
        st.error(f"Errore nel recupero dello storico dei prezzi da Yahoo Finance: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def fetch_income_statements(symbol, api_key, period='annual', limit=60):
    """Recupera i dati del conto economico per 15 anni"""
    url = f"https://api.financialdatasets.ai/financials/income-statements?ticker={symbol}&period={period}&limit={limit}"
    
    headers = {
        "X-API-KEY": api_key
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        statements = response.json().get('income_statements', [])
        
        # Converti in DataFrame
        df = pd.DataFrame(statements)
        
        # Utilizza calendar_date o report_period come indice
        if 'report_period' in df.columns:
            df['Date'] = pd.to_datetime(df['report_period'])
            df.set_index('Date', inplace=True)
        elif 'calendar_date' in df.columns:
            df['Date'] = pd.to_datetime(df['calendar_date'])
            df.set_index('Date', inplace=True)
            
        return df
    except Exception as e:
        st.error(f"Errore nel recupero dei dati del conto economico: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def fetch_balance_sheets(symbol, api_key, period='annual', limit=60):
    """Recupera i dati dello stato patrimoniale per 15 anni"""
    url = f"https://api.financialdatasets.ai/financials/balance-sheets?ticker={symbol}&period={period}&limit={limit}"
    
    headers = {
        "X-API-KEY": api_key
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        statements = response.json().get('balance_sheets', [])
        
        # Converti in DataFrame
        df = pd.DataFrame(statements)
        
        # Utilizza calendar_date o report_period come indice
        if 'report_period' in df.columns:
            df['Date'] = pd.to_datetime(df['report_period'])
            df.set_index('Date', inplace=True)
        elif 'calendar_date' in df.columns:
            df['Date'] = pd.to_datetime(df['calendar_date'])
            df.set_index('Date', inplace=True)
            
        return df
    except Exception as e:
        st.error(f"Errore nel recupero dei dati dello stato patrimoniale: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def fetch_cashflow_statements(symbol, api_key, period='annual', limit=60):
    """Recupera i dati del rendiconto finanziario per 15 anni"""
    url = f"https://api.financialdatasets.ai/financials/cash-flow-statements?ticker={symbol}&period={period}&limit={limit}"
    
    headers = {
        "X-API-KEY": api_key
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        statements = response.json().get('cash_flow_statements', [])
        
        # Converti in DataFrame
        df = pd.DataFrame(statements)
        
        # Utilizza calendar_date o report_period come indice
        if 'report_period' in df.columns:
            df['Date'] = pd.to_datetime(df['report_period'])
            df.set_index('Date', inplace=True)
        elif 'calendar_date' in df.columns:
            df['Date'] = pd.to_datetime(df['calendar_date'])
            df.set_index('Date', inplace=True)
            
        return df
    except Exception as e:
        st.error(f"Errore nel recupero dei dati del rendiconto finanziario: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def analyze_stock_with_perplexity(symbol, company_name, api_key):
    """Analizza news e sentiment di mercato con Perplexity Sonar API"""
    url = "https://api.perplexity.ai/chat/completions"
    
    prompt = f"""Analizza gli ultimi risultati finanziari per {company_name} (simbolo: {symbol}).
    
Fornisci un analisi finale senza mostrare alcun ragionamento o processo di ricerca. Analisi deve:
1. Descrivere nel dettaglio gli ultimi risultati finanziari pubblicati nell'ultimo trimestre
2. Menzionare eventi significativi o annunci dell'azienda
3. Indicare il sentiment del mercato e previsioni degli analisti

Rispondi direttamente con l'analisi finale, senza introduzioni né commenti sul tuo processo di pensiero."""
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "sonar",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 512,
        "search_context": {
            "search_context_size": "high"
        }
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Solleva eccezione per errori HTTP
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Errore nell'analisi: {str(e)}"

def get_balance_sheet_field(df, field_options):
    """Trova il primo campo disponibile da una lista di opzioni"""
    for field in field_options:
        if field in df.columns:
            return field
    return None

def calculate_book_value_per_share(df_balance, shares_outstanding):
    """Calcola Book Value per Azione"""
    # Possibili nomi per Total Equity
    equity_field_options = [
        'shareholders_equity',
        'total_equity',
        'total_equity_gross_minority_interest',
        'stockholders_equity',
        'total_stockholder_equity'
    ]
    
    equity_field = get_balance_sheet_field(df_balance, equity_field_options)
    
    if equity_field and shares_outstanding:
        # Calcola book value per share per ogni periodo
        df_balance['Book Value per Share'] = df_balance[equity_field] / shares_outstanding
        return 'Book Value per Share', equity_field
    return None, None

def calculate_cash_flow_per_share(df_cashflow, shares_outstanding):
    """Calcola Cash Flow per Azione utilizzando Free Cash Flow o Operating Cash Flow"""
    # Prima prova con Free Cash Flow
    fcf_field_options = [
        'free_cash_flow',
        'freeCashFlow'
    ]
    
    fcf_field = get_balance_sheet_field(df_cashflow, fcf_field_options)
    
    if fcf_field and shares_outstanding:
        # Calcola free cash flow per share per ogni periodo
        df_cashflow['Cash Flow per Share'] = df_cashflow[fcf_field] / shares_outstanding
        return 'Cash Flow per Share', fcf_field
    
    # Se Free Cash Flow non è disponibile, usa Operating Cash Flow
    ocf_field_options = [
        'operating_cash_flow',
        'cash_from_operations',
        'cash_flow_from_operations',
        'net_cash_from_operating_activities',
        'operatingCashFlow',
        'totalCashFromOperatingActivities'
    ]
    
    ocf_field = get_balance_sheet_field(df_cashflow, ocf_field_options)
    
    if ocf_field and shares_outstanding:
        # Calcola operating cash flow per share per ogni periodo
        df_cashflow['Cash Flow per Share'] = df_cashflow[ocf_field] / shares_outstanding
        return 'Cash Flow per Share', ocf_field
    
    # Se neanche Operating Cash Flow è disponibile, prova a calcolarlo manualmente
    # Free Cash Flow = Operating Cash Flow - Capital Expenditures
    ocf_fields = ['operating_cash_flow', 'cash_from_operations', 'operatingCashFlow']
    capex_fields = ['capital_expenditure', 'capitalExpenditure', 'capitalExpenditures']
    
    ocf_field = get_balance_sheet_field(df_cashflow, ocf_fields)
    capex_field = get_balance_sheet_field(df_cashflow, capex_fields)
    
    if ocf_field and capex_field and shares_outstanding:
        # Calcola Free Cash Flow manualmente
        df_cashflow['Free Cash Flow Calcolato'] = df_cashflow[ocf_field] - abs(df_cashflow[capex_field])
        df_cashflow['Cash Flow per Share'] = df_cashflow['Free Cash Flow Calcolato'] / shares_outstanding
        return 'Cash Flow per Share', 'Free Cash Flow Calcolato'
        
    return None, None

def calculate_dcf_value(info, annual_financials, annual_cashflow, annual_balance_sheet):
    """Calcolo valore intrinseco con metodo DCF (Discounted Cash Flow)"""
    try:
        # Parametri per DCF
        discount_rate = st.slider("Tasso di Sconto (%)", min_value=5.0, max_value=20.0, value=10.0, step=0.5) / 100
        growth_rate_initial = st.slider("Tasso di Crescita Iniziale (%)", min_value=1.0, max_value=30.0, value=15.0, step=0.5) / 100
        growth_rate_terminal = st.slider("Tasso di Crescita Terminale (%)", min_value=1.0, max_value=5.0, value=2.5, step=0.1) / 100
        forecast_period = st.slider("Periodo di Previsione (anni)", min_value=5, max_value=20, value=10)
        
        # Ottieni Free Cash Flow più recente
        fcf_field_options = [
            'free_cash_flow',
            'freeCashFlow'
        ]
        
        # Trova il campo FCF disponibile
        fcf_field = get_balance_sheet_field(annual_cashflow, fcf_field_options)
        
        # Se Free Cash Flow non è disponibile, prova con Operating Cash Flow
        if fcf_field is None:
            ocf_field_options = [
                'operating_cash_flow',
                'cash_from_operations',
                'cash_flow_from_operations',
                'net_cash_from_operating_activities',
                'operatingCashFlow',
                'totalCashFromOperatingActivities'
            ]
            
            fcf_field = get_balance_sheet_field(annual_cashflow, ocf_field_options)
            
            if fcf_field:
                st.info("Free Cash Flow non disponibile. Utilizzando Operating Cash Flow per il calcolo DCF.")
        
        # Se neanche Operating Cash Flow è disponibile, prova a calcolare FCF manualmente
        if fcf_field is None:
            ocf_fields = ['operating_cash_flow', 'cash_from_operations', 'operatingCashFlow']
            capex_fields = ['capital_expenditure', 'capitalExpenditure', 'capitalExpenditures']
            
            ocf_field = get_balance_sheet_field(annual_cashflow, ocf_fields)
            capex_field = get_balance_sheet_field(annual_cashflow, capex_fields)
            
            if ocf_field and capex_field:
                # Calcola Free Cash Flow manualmente
                annual_cashflow['free_cash_flow_calculated'] = annual_cashflow[ocf_field] - abs(annual_cashflow[capex_field])
                fcf_field = 'free_cash_flow_calculated'
                st.info("Free Cash Flow calcolato come: Operating Cash Flow - Capital Expenditures")
        
        if fcf_field is None:
            st.warning("Dati di Free Cash Flow e Operating Cash Flow non disponibili.")
            return None
            
        # Prendi l'ultimo valore disponibile di FCF
        fcf = annual_cashflow[fcf_field].iloc[0]
        
        # Se FCF è negativo, usa la media degli ultimi 3 anni (se disponibile)
        if fcf <= 0 and len(annual_cashflow) >= 3:
            fcf = annual_cashflow[fcf_field].iloc[:3].mean()
            if fcf <= 0:
                st.warning("Free Cash Flow negativo o zero, impossibile calcolare DCF.")
                return None
        elif fcf <= 0:
            st.warning("Free Cash Flow negativo o zero, impossibile calcolare DCF.")
            return None
        
        # Numero di azioni in circolazione - usa diverse fonti
        shares_outstanding = (information.get('sharesOutstanding') or 
                            information.get('impliedSharesOutstanding') or
                            information.get('floatShares'))
        
        # Se non trovato in Yahoo Finance, cerca nei dati FinancialDatasets
        if not shares_outstanding and additional_metrics:
            shares_outstanding = (additional_metrics.get('outstanding_shares') or 
                                additional_metrics.get('weighted_average_shares_diluted') or
                                additional_metrics.get('shares_outstanding'))
        
        # Se ancora non trovato, cerca nei dati finanziari
        if not shares_outstanding and not annual_income.empty:
            weighted_avg_field_options = [
                'weighted_average_shares_diluted',
                'weighted_average_shares',
                'diluted_average_shares'
            ]
            
            for field in weighted_avg_field_options:
                if field in annual_income.columns:
                    shares_outstanding = annual_income[field].iloc[0]
                    if shares_outstanding and shares_outstanding > 0:
                        break
        
        if not shares_outstanding:
            st.warning("Numero di azioni in circolazione non disponibile.")
            return None
        
        # Calcola i flussi di cassa proiettati
        projected_cash_flows = []
        
        for year in range(1, forecast_period + 1):
            # Diminuisci gradualmente il tasso di crescita verso il tasso terminale
            if forecast_period > 1:
                weight = (forecast_period - year) / (forecast_period - 1)
                growth_rate = weight * growth_rate_initial + (1 - weight) * growth_rate_terminal
            else:
                growth_rate = growth_rate_terminal
                
            # Calcola il flusso di cassa per questo anno
            projected_cf = fcf * (1 + growth_rate) ** year
            
            # Calcola il valore presente di questo flusso di cassa
            present_value = projected_cf / (1 + discount_rate) ** year
            
            projected_cash_flows.append(present_value)
        
        # Calcola il valore terminale
        terminal_value = (fcf * (1 + growth_rate_terminal) ** forecast_period * (1 + growth_rate_terminal)) / (discount_rate - growth_rate_terminal)
        present_terminal_value = terminal_value / (1 + discount_rate) ** forecast_period
        
        # Calcola il valore totale dell'azienda
        enterprise_value = sum(projected_cash_flows) + present_terminal_value
        
        # Ottieni il debito totale e la liquidità
        debt_field_options = [
            'total_debt',
            'non_current_debt',
            'long_term_debt',
            'total_debt_and_capital_lease_obligation'
        ]
        
        cash_field_options = [
            'cash_and_equivalents',
            'cash_and_cash_equivalents',
            'cash_cash_equivalents_and_short_term_investments',
            'cash_and_short_term_investments'
        ]
        
        # Get data from the most recent balance sheet
        balance_sheet = annual_balance_sheet.iloc[0] if not annual_balance_sheet.empty else None
        
        if balance_sheet is not None:
            debt_field = get_balance_sheet_field(annual_balance_sheet, debt_field_options)
            cash_field = get_balance_sheet_field(annual_balance_sheet, cash_field_options)
            
            total_debt = balance_sheet[debt_field] if debt_field else 0
            total_cash = balance_sheet[cash_field] if cash_field else 0
            
            # Calcola equity value
            equity_value = enterprise_value - total_debt + total_cash
            
            # Calcola valore intrinseco per azione
            intrinsic_value_per_share = equity_value / shares_outstanding
            
            # Prezzo attuale
            current_price = info.get('current_price', info.get('price', 0))
            
            # Mostra il valore intrinseco calcolato
            st.metric(
                label="Valore Intrinseco per Azione (DCF)",
                value=f"${intrinsic_value_per_share:.2f}",
                delta=f"{(intrinsic_value_per_share / current_price - 1) * 100:.1f}% vs prezzo attuale" if current_price > 0 else "N/A"
            )
            
            # Mostra informazioni di base sui parametri utilizzati
            st.info(f"""
            **Parametri utilizzati:**
            - Free Cash Flow: ${fcf/1000000:.2f}M
            - Tasso di crescita iniziale: {growth_rate_initial*100:.1f}%
            - Tasso di crescita terminale: {growth_rate_terminal*100:.1f}%
            - Tasso di sconto: {discount_rate*100:.1f}%
            - Periodo di previsione: {forecast_period} anni
            """)
            
            return intrinsic_value_per_share
        else:
            st.warning("Dati del bilancio non disponibili.")
            return None
        
    except Exception as e:
        st.error(f"Errore nel calcolo DCF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())  # Mostra lo stack trace completo per il debug
        return None

def calculate_graham_value(info, annual_financials, earnings_growth=None):
    """Calcolo valore intrinseco con Formula di Graham"""
    try:
        # Prova a ottenere EPS da diverse fonti
        eps = (information.get('trailingEPS') or 
               information.get('ttmEPS') or
               additional_metrics.get('trailing_eps') or
               additional_metrics.get('earnings_per_share'))
        
        # Se EPS non è disponibile, prova a cercarlo nei dati finanziari
        if not eps or eps <= 0:
            if not annual_financials.empty and 'earnings_per_share_diluted' in annual_financials.columns:
                # Prendi l'EPS più recente
                eps = annual_financials['earnings_per_share_diluted'].iloc[0]
            elif not annual_financials.empty and 'earnings_per_share' in annual_financials.columns:
                eps = annual_financials['earnings_per_share'].iloc[0]
        
        # Se ancora non disponibile, consenti all'utente di inserirlo manualmente
        if not eps or eps <= 0:
            st.warning("EPS non disponibile dai dati. Inserisci un valore manualmente.")
            eps = st.number_input("Inserisci EPS manualmente:", value=1.0, step=0.1)
        
        # Stima del tasso di crescita (se non fornito)
        if earnings_growth is None:
            # Calcola il tasso di crescita se abbiamo abbastanza dati
            if not annual_financials.empty and len(annual_financials) > 1:
                eps_column = 'earnings_per_share_diluted' if 'earnings_per_share_diluted' in annual_financials.columns else 'earnings_per_share'
                
                if eps_column in annual_financials.columns:
                    # Ordina per data decrescente
                    sorted_data = annual_financials.sort_index(ascending=False)
                    
                    # Prendi primo e ultimo valore disponibile
                    latest_eps = sorted_data[eps_column].iloc[0]
                    oldest_eps = sorted_data[eps_column].iloc[-1]
                    
                    # Calcola il numero di anni
                    years = (sorted_data.index[0] - sorted_data.index[-1]).days / 365.25
                    
                    if years > 0 and latest_eps > 0 and oldest_eps > 0:
                        # Calcola CAGR
                        earnings_growth = ((latest_eps / oldest_eps) ** (1/years) - 1) * 100
            
            # Se non riusciamo a calcolare il tasso, usa un valore predefinito
            if earnings_growth is None:
                earnings_growth = 10.0
                st.warning("Tasso di crescita degli utili non disponibile. Utilizzando 10.0% come valore predefinito.")
        
        # Assicurati che il valore sia nell'intervallo consentito
        growth_rate_default = min(max(float(earnings_growth), 0.0), 30.0)
        
        # Parametri per la formula di Graham
        growth_rate = st.slider("Tasso di Crescita Annuale (%)", 
                               min_value=0.0, 
                               max_value=30.0, 
                               value=float(growth_rate_default),
                               step=0.5)
        
        bond_yield = st.slider("Rendimento Bond AAA (%)", 
                              min_value=1.0, 
                              max_value=10.0, 
                              value=4.5, 
                              step=0.1)
        
        # Formula originale di Graham: V = EPS × (8.5 + 2g) × 4.4 / Y
        # dove:
        # V = valore intrinseco
        # EPS = utile per azione
        # g = tasso di crescita degli utili (previsto per i prossimi 7-10 anni)
        # Y = rendimento corrente dei bond AAA
        
        intrinsic_value = eps * (8.5 + 2 * (growth_rate / 100)) * (4.4 / bond_yield)
        
        # Prezzo attuale
        current_price = information.get('currentPrice', information.get('regularMarketPrice', 0))
        
        # Mostra il valore intrinseco calcolato
        st.metric(
            label="Valore Intrinseco per Azione (Graham)",
            value=f"${intrinsic_value:.2f}",
            delta=f"{(intrinsic_value / current_price - 1) * 100:.1f}% vs prezzo attuale" if current_price > 0 else "N/A"
        )
        
        # Mostra la formula utilizzata
        st.info(f"""
        **Formula di Graham utilizzata:** V = EPS × (8.5 + 2g) × 4.4 / Y
        
        **Parametri:**
        - EPS: ${eps:.2f}
        - Tasso di crescita (g): {growth_rate:.1f}%
        - Rendimento Bond AAA (Y): {bond_yield:.1f}%
        
        **Calcolo:** ${eps:.2f} × ({8.5 + (2 * growth_rate / 100):.2f}) × ({4.4 / bond_yield:.2f}) = ${intrinsic_value:.2f}
        """)
        
        return intrinsic_value
        
    except Exception as e:
        st.error(f"Errore nel calcolo con la Formula di Graham: {str(e)}")
        import traceback
        st.error(traceback.format_exc())  # Mostra lo stack trace completo per il debug
        return None

# Funzione per formattare i valori in modo sicuro
def safe_format(value, format_str):
    try:
        if value is None:
            return "N/A"
        return format_str.format(value)
    except:
        return "N/A"

# Funzione per determinare se un valore deve essere colorato di verde
def should_be_green(value, condition_type):
    """Determina se un valore soddisfa le condizioni per essere colorato di verde"""
    if value is None or value == "N/A":
        return False
    
    try:
        # Estrai il valore numerico dalla stringa
        if isinstance(value, str):
            # Rimuovi % e altri caratteri
            numeric_value = float(value.replace('%', '').replace(',', '').replace('$', ''))
        else:
            numeric_value = float(value)
        
        # Definisci le condizioni per il colore verde
        if condition_type == "peg" and 0 <= numeric_value <= 2:
            return True
        elif condition_type == "pe" and 0 <= numeric_value <= 15:
            return True
        elif condition_type == "pb" and 0 <= numeric_value <= 1.5:
            return True
        elif condition_type == "roic" and numeric_value > 10:
            return True
        elif condition_type == "debt_equity" and numeric_value < 1:
            return True
        
        return False
            
    except (ValueError, TypeError):
        return False

# === CONFIGURAZIONE API (INSERISCI LE TUE CHIAVI QUI) ===
# IMPORTANTE: Sostituisci queste chiavi con le tue API keys reali
financial_datasets_api_key = os.getenv("MY_DATASET_API_KEY")  
perplexity_api_key = os.getenv("MY_SONAR_API_KEY")  

# Applicazione principale ---------------------------------------------------

st.title('Analisi Fondamentale Azioni')
st.caption('Analisi Fondamentale secondo i principi del Value Investing interpretati da DIRAMCO')

# === SEZIONE INPUT TICKER IN ALTO ===
st.header("Seleziona Titolo del Mercato USA")
col_ticker, col_button = st.columns([3, 1])

with col_ticker:
    symbol = st.text_input(
        'Inserisci il Ticker del Titolo Azionario', 
        'AAPL', 
        help="Es. AAPL, MSFT, GOOGL",
        placeholder="Inserisci il simbolo del titolo..."
    )

with col_button:
    st.write("")  # Spazio per allineare il pulsante
    st.write("")  # Spazio per allineare il pulsante
    analyze_button = st.button("Analizza", type="primary", use_container_width=True)

# Procedi solo se il simbolo è inserito
if not symbol:
    st.info("Inserisci il ticker di un titolo per iniziare l'analisi")
    st.stop()

# Recupera le informazioni di base sul titolo da Yahoo Finance
information = fetch_stock_info_yahoo(symbol)

if not information:
    st.error(f"❌ Nessuna informazione trovata per il ticker {symbol}. Verifica che il ticker sia corretto.")
    st.stop()

# Recupera metriche aggiuntive da FinancialDatasets AI se disponibile
additional_metrics = {}
if financial_datasets_api_key and financial_datasets_api_key != "YOUR_FINANCIAL_DATASETS_API_KEY_HERE":
    additional_metrics = fetch_financial_metrics(symbol, financial_datasets_api_key)

# === SEZIONE INFORMAZIONI TITOLO E GRAFICO ===
st.header('Panoramica Titolo')

# Layout a due colonne: info + grafico
col_info, col_chart = st.columns([1, 2])

with col_info:
    # Nomi aziendali e prezzo corrente
    company_name = information.get("longName", information.get("shortName", symbol))
    current_price = information.get("currentPrice", 0)
    percent_change = information.get("regularMarketChangePercent", 0)
    color = "🟢" if percent_change >= 0 else "🔴"
    change_sign = "+" if percent_change >= 0 else ""

    st.subheader(f'{company_name}')
    st.metric(
        label="Prezzo Attuale",
        value=f"${current_price:.2f}",
        delta=f"{change_sign}{percent_change:.2f}%"
    )

    market_cap = information.get("marketCap", 0)
    if market_cap:
        st.metric(
            label="Capitalizzazione",
            value=f"${market_cap/1000000000:.1f}B"
        )

    sector = information.get("sector", "N/A")
    st.info(f'**Settore:** {sector}')

    # Indicatori rapidi
    st.subheader("Indicatori Chiave")
    
    col_ind1, col_ind2 = st.columns(2)
    
    with col_ind1:
        pe_ratio = information.get("trailingPE", 0)
        st.metric("P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
        
        dividend_yield = information.get("dividendYield", 0)
        dividend_pct = dividend_yield * 100 if dividend_yield else 0
        st.metric("Rendimento Dividendo", f"{dividend_pct/100:.2f}%" if dividend_pct else "N/A")
    
    with col_ind2:
        pb_ratio = information.get("priceToBook", 0)
        st.metric("P/B Ratio", f"{pb_ratio:.2f}" if pb_ratio else "N/A")
        
        beta = information.get("beta", 0)
        st.metric("Beta", f"{beta:.2f}" if beta else "N/A")

with col_chart:
    # GRAFICO A CANDELE
    price_history = fetch_price_history_yahoo(symbol)
    
    if not price_history.empty:
        st.subheader('Grafico 1 Anno')
        
        price_history_reset = price_history.rename_axis('Data').reset_index()
        
        # Prepara i dati per il grafico a candele
        candle_stick_chart = go.Figure(data=[go.Candlestick(
            x=price_history_reset['Data'], 
            open=price_history_reset['Open'], 
            low=price_history_reset['Low'],
            high=price_history_reset['High'],
            close=price_history_reset['Close'],
            name=symbol
        )])
        
        candle_stick_chart.update_layout(
            height=500,
            xaxis_title="Data",
            yaxis_title="Prezzo ($)",
            xaxis_rangeslider_visible=False,
            showlegend=False
        )
        
        st.plotly_chart(candle_stick_chart, use_container_width=True)
        
        st.caption('Visualizza Analisi Tecnica Avanzata tramite IA [qui](https://diramco.com/analisi-tecnica-ai/)')
    else:
        st.warning("⚠️ Dati dei prezzi storici non disponibili.")

# === SEZIONE INDICATORI FINANZIARI DETTAGLIATI ===
st.header('Indicatori Finanziari Dettagliati')

# Funzione per determinare se un valore deve essere colorato di verde
def should_be_green(value, condition_type):
    """Determina se un valore soddisfa le condizioni per essere colorato di verde"""
    if value is None or value == "N/A":
        return False
    
    try:
        # Estrai il valore numerico dalla stringa
        if isinstance(value, str):
            # Rimuovi % e altri caratteri
            numeric_value = float(value.replace('%', '').replace(',', '').replace('$', ''))
            # Se la stringa originale conteneva %, il valore è già in percentuale
            is_percentage_string = '%' in value
        else:
            numeric_value = float(value)
            is_percentage_string = False
        
        # Per ROIC, ROE, ROA: gestisci sia formato decimale che percentuale
        if condition_type in ["roic", "roe", "roa"]:
            if is_percentage_string:
                # Se è una stringa con %, il valore è già convertito (es. "15.00%" -> 15.0)
                threshold_roic = 10.0
                threshold_roe = 15.0 
                threshold_roa = 10.0
            else:
                # Se è un decimale (es. 0.15), converti la soglia
                threshold_roic = 0.10  # 10%
                threshold_roe = 0.15   # 15%
                threshold_roa = 0.10   # 10%
            
            if condition_type == "roic":
                return numeric_value > threshold_roic
            elif condition_type == "roe":
                return numeric_value > threshold_roe  
            elif condition_type == "roa":
                return numeric_value > threshold_roa
        
        # Altre condizioni rimangono uguali
        conditions = {
            "peg": 0 <= numeric_value <= 2,
            "pe": 0 <= numeric_value <= 15,
            "pb": 0 <= numeric_value <= 1.5,
            "ps": 0 <= numeric_value <= 2,
            "debt_equity": numeric_value < 1,
            "current_ratio": numeric_value >= 1.5,
            "quick_ratio": numeric_value >= 1.0,
            "beta": 0.5 <= numeric_value <= 1.5
        }
        
        return conditions.get(condition_type, False)
            
    except (ValueError, TypeError):
        return False

# Preparazione dati per la visualizzazione con formattazione colorata
def format_indicator_value(value, condition_type):
    """Formatta un valore e restituisce sia il valore che se deve essere verde"""
    if condition_type in ["roic", "roe", "roa"]:
        # Questi devono essere mostrati come percentuale
        if value is not None and value != 0:
            if value < 1:
                # Valore in formato decimale (es. 0.15 = 15%)
                formatted_value = f"{value * 100:.2f}%"
                is_green = should_be_green(value, condition_type)  # Passa il valore decimale originale
            else:
                # Valore già in percentuale (es. 15.0 = 15%)
                formatted_value = f"{value:.2f}%"
                is_green = should_be_green(formatted_value, condition_type)  # Passa come stringa con %
        else:
            formatted_value = "N/A"
            is_green = False
    elif condition_type in ["operating_margin", "profit_margin", "gross_margin"]:
        # Margini sono già in formato decimale
        if value is not None and value != 0:
            formatted_value = f"{value * 100:.2f}%"
            is_green = value > 0.10  # Verde se margine > 10%
        else:
            formatted_value = "N/A"
            is_green = False
    else:
        formatted_value = safe_format(value, "{:.2f}")
        is_green = should_be_green(formatted_value, condition_type)
    
    return formatted_value, is_green

# Calcola PEG Ratio
peg_ratio = None
pe_ratio = information.get("trailingPE", 0)
eps_growth = information.get("earningsGrowth", 0)

if pe_ratio and eps_growth and eps_growth > 0:
    peg_ratio = pe_ratio / (eps_growth * 100)

# Calcola ROIC - prova prima da FinancialDatasets, poi usa ROA come fallback
roic = None
if additional_metrics:
    # Prova diversi nomi possibili per ROIC in FinancialDatasets
    roic_field_options = [
        'roic',
        'return_on_invested_capital', 
        'returnOnInvestedCapital',
        'roic_percent',
        'roic_ratio'
    ]
    
    for field in roic_field_options:
        if field in additional_metrics and additional_metrics[field] is not None:
            roic = additional_metrics[field]
            # Se il valore è già in percentuale (>1), convertilo in decimale
            if roic > 1:
                roic = roic / 100
            break

# Se ROIC non è disponibile da FinancialDatasets, usa ROA come proxy
if roic is None:
    roa = information.get("returnOnAssets", 0)
    if roa:
        roic = roa
        # Se ROA è in percentuale (>1), convertilo in decimale
        if roic > 1:
            roic = roic / 100

# Calcola altri indicatori
price_to_sales = information.get("priceToSalesTrailing12Months", 0)
quick_ratio = information.get("quickRatio", 0)
current_ratio = information.get("currentRatio", 0)
operating_margin = information.get("operatingMargins", 0)
profit_margin = information.get("profitMargins", 0)
gross_margins = information.get("grossMargins", 0)

# Creazione dei dati per le colonne
pe_value, pe_is_green = format_indicator_value(information.get("trailingPE", 0), "pe")
pb_value, pb_is_green = format_indicator_value(information.get("priceToBook", 0), "pb")
peg_value, peg_is_green = format_indicator_value(peg_ratio, "peg") if peg_ratio else ("N/A", False)
ps_value, ps_is_green = format_indicator_value(price_to_sales, "ps")
debt_equity_ratio = information.get("debtToEquity", 0)/100 if information.get("debtToEquity") else 0
debt_eq_value, debt_eq_is_green = format_indicator_value(debt_equity_ratio, "debt_equity")
roic_value, roic_is_green = format_indicator_value(roic, "roic")
roe_value, roe_is_green = format_indicator_value(information.get("returnOnEquity", 0), "roe")
roa_value, roa_is_green = format_indicator_value(information.get("returnOnAssets", 0), "roa")

# Layout a due colonne per le tabelle
col_table1, col_table2 = st.columns(2)

with col_table1:
    st.subheader("Indicatori di Prezzo")
    
    # Crea la tabella con colorazione
    indicators_price = [
        ("P/E Ratio", pe_value, pe_is_green, "Prezzo/Utili - Quanto si paga per 1$ di utili"),
        ("P/B Ratio", pb_value, pb_is_green, "Prezzo/Valore Contabile - Rapporto con patrimonio netto"),
        ("P/S Ratio", ps_value, ps_is_green, "Prezzo/Ricavi - Valutazione basata sui ricavi"),
        ("PEG Ratio", peg_value, peg_is_green, "P/E aggiustato per crescita - Ideale tra 0.5-2.0")
    ]
    
    # Visualizza ogni riga con colorazione appropriata
    for indicator, value, is_green, tooltip in indicators_price:
        col_label, col_value = st.columns([2.5, 1])
        with col_label:
            st.write(f"**{indicator}**")
            st.caption(tooltip)
        with col_value:
            if is_green:
                st.success(f"✅ {value}")
            elif value != "N/A" and value != "0.00":
                st.write(f"{value}")
            else:
                st.write("❌ N/A")

with col_table2:
    st.subheader("Indicatori di Performance")
    
    # Crea la tabella con colorazione
    indicators_performance = [
        ("ROE", roe_value, roe_is_green, "Return on Equity - Redditività del capitale proprio"),
        ("ROA", roa_value, roa_is_green, "Return on Assets - Efficienza nell'uso degli asset"),
        ("ROIC", roic_value, roic_is_green, "Return on Invested Capital - Redditività capitale investito"),
        ("Debt/Equity", debt_eq_value, debt_eq_is_green, "Rapporto Debito/Equity - Leva finanziaria")
    ]
    
    # Visualizza ogni riga con colorazione appropriata
    for indicator, value, is_green, tooltip in indicators_performance:
        col_label, col_value = st.columns([2.5, 1])
        with col_label:
            st.write(f"**{indicator}**")
            st.caption(tooltip)
        with col_value:
            if is_green:
                st.success(f"✅ {value}")
            elif value != "N/A" and value != "0.00":
                st.write(f"{value}")
            else:
                st.write("❌ N/A")

# === SEZIONE INDICATORI DI MARGINE E LIQUIDITÀ ===
st.subheader("Indicatori di Redditività e Liquidità")

col_margin1, col_margin2 = st.columns(2)

with col_margin1:
    st.write("**Margini di Redditività**")
    
    # Margini di redditività
    margin_indicators = [
        ("Margine Lordo", gross_margins, "Ricavi - Costi diretti / Ricavi"),
        ("Margine Operativo", operating_margin, "Utile operativo / Ricavi"),
        ("Margine Netto", profit_margin, "Utile netto / Ricavi")
    ]
    
    for indicator, value, tooltip in margin_indicators:
        col_label_m, col_value_m = st.columns([2, 1])
        with col_label_m:
            st.write(f"**{indicator}**")
            st.caption(tooltip)
        with col_value_m:
            if value and value > 0.10:  # Verde se margine > 10%
                st.success(f"✅ {value*100:.2f}%")
            elif value and value > 0:
                st.write(f"{value*100:.2f}%")
            else:
                st.write("❌ N/A")

with col_margin2:
    st.write("**Indicatori di Liquidità**")
    
    # Indicatori di liquidità
    liquidity_indicators = [
        ("Current Ratio", current_ratio, "Attività correnti / Passività correnti"),
        ("Quick Ratio", quick_ratio, "(Liquidità + Crediti) / Passività correnti"),
        ("Beta", information.get("beta", 0), "Volatilità rispetto al mercato")
    ]
    
    for indicator, value, tooltip in liquidity_indicators:
        col_label_l, col_value_l = st.columns([2, 1])
        with col_label_l:
            st.write(f"**{indicator}**")
            st.caption(tooltip)
        with col_value_l:
            if indicator == "Current Ratio" and value and value >= 1.5:
                st.success(f"✅ {value:.2f}")
            elif indicator == "Quick Ratio" and value and value >= 1.0:
                st.success(f"✅ {value:.2f}")
            elif indicator == "Beta" and value and 0.5 <= value <= 1.5:
                st.success(f"✅ {value:.2f}")
            elif value and value > 0:
                st.write(f"{value:.2f}")
            else:
                st.write("❌ N/A")

# === LEGENDA INTERPRETAZIONE ===
with st.expander("Come interpretare questi indicatori", expanded=False):
    st.markdown("""
    ### Indicatori di Valutazione
    - **P/E Ratio < 15**: Potenzialmente sottovalutato ✅
    - **P/B Ratio < 1.5**: Prezzo ragionevole rispetto al patrimonio ✅  
    - **P/S Ratio < 2**: Valutazione conservativa sui ricavi ✅
    - **PEG Ratio 0.5-2.0**: Crescita giustifica la valutazione ✅
    
    ### Indicatori di Performance  
    - **ROE > 15%**: Ottima redditività del capitale ✅
    - **ROA > 10%**: Efficiente utilizzo degli asset ✅
    - **ROIC > 10%**: Buon ritorno sul capitale investito ✅
    - **Debt/Equity < 1**: Leva finanziaria controllata ✅
    
    ### Margini e Liquidità
    - **Margini > 10%**: Buona redditività operativa ✅
    - **Current Ratio > 1.5**: Buona liquidità a breve termine ✅
    - **Quick Ratio > 1.0**: Liquidità immediata sufficiente ✅
    - **Beta 0.5-1.5**: Volatilità in linea con il mercato ✅
    """)
# === SEZIONE ANALISI NEWS ===
if perplexity_api_key and perplexity_api_key != "YOUR_PERPLEXITY_API_KEY_HERE":
    st.header('Analisi delle News e degli ultimi Risultati Finanziari tramite IA')

    with st.expander("Clicca per vedere gli ultimi risultati", expanded=False):
        with st.spinner("Analizzando le notizie di mercato con IA..."):
            ai_analysis = analyze_stock_with_perplexity(symbol, company_name, perplexity_api_key)
            st.markdown(ai_analysis)
            
        st.info("Analisi generata tramite IA basata sulle informazioni di mercato più recenti.")
else:
    st.info("Configura la chiave API Perplexity nel codice per abilitare l'analisi delle notizie con IA.")

# === SEZIONE DATI FINANZIARI ===
st.header('Dati Finanziari Storici')

if not financial_datasets_api_key or financial_datasets_api_key == "YOUR_FINANCIAL_DATASETS_API_KEY_HERE":
    st.warning("Configura la chiave API FinancialDatasets nel codice per visualizzare i dati finanziari dettagliati.")
else:
    # Controlli per il periodo
    col_period, col_spacer = st.columns([1, 3])
    with col_period:
        selection = st.segmented_control(
            label='Periodo di Analisi', 
            options=['Trimestrale', 'Annuale'], 
            default='Annuale'
        )

    # Carica i dati finanziari
    if selection == 'Trimestrale':
        period = 'quarterly'
    else:
        period = 'annual'

    with st.spinner("📈 Caricamento dati finanziari..."):
        income_statements = fetch_income_statements(symbol, financial_datasets_api_key, period=period, limit=60)
        balance_sheets = fetch_balance_sheets(symbol, financial_datasets_api_key, period=period, limit=60)
        cashflow_statements = fetch_cashflow_statements(symbol, financial_datasets_api_key, period=period, limit=60)

    if income_statements.empty or balance_sheets.empty or cashflow_statements.empty:
        st.warning("⚠️ Alcuni dati finanziari non sono disponibili. I grafici potrebbero essere incompleti.")

    # === GRAFICI CONTO ECONOMICO ===
    if not income_statements.empty:
        st.subheader("Analisi Conto Economico")
        
        # Prepara income statements per la visualizzazione
        income_reset = income_statements.reset_index()
        if 'Date' in income_reset.columns:
            if period == 'annual':
                income_reset['Anno'] = income_reset['Date'].dt.strftime('%Y')
            else:
                income_reset['Quarter'] = income_reset['Date'].dt.strftime('%Y-Q%q')
        
        income_reset = income_reset.sort_values(by='Date', ascending=False).head(60)
        income_reset = income_reset.sort_values(by='Date')
        
        # Layout a coppie per i grafici
        import plotly.express as px
        
        # Prima coppia: Ricavi e EPS
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            if 'revenue' in income_reset.columns:
                fig_revenue = px.bar(
                    income_reset, 
                    x='Anno' if period == 'annual' else 'Quarter', 
                    y='revenue',
                    title="Ricavi Totali",
                    labels={"revenue": "Ricavi ($)", 'Anno' if period == 'annual' else 'Quarter': "Periodo"},
                    color_discrete_sequence=['#1f77b4']
                )
                fig_revenue.update_layout(height=350)
                st.plotly_chart(fig_revenue, use_container_width=True)
        
        with col_chart2:
            eps_field = 'earnings_per_share_diluted' if 'earnings_per_share_diluted' in income_reset.columns else 'earnings_per_share'
            if eps_field in income_reset.columns:
                fig_eps = px.bar(
                    income_reset, 
                    x='Anno' if period == 'annual' else 'Quarter', 
                    y=eps_field,
                    title="Utile per Azione (EPS)",
                    labels={eps_field: "EPS ($)", 'Anno' if period == 'annual' else 'Quarter': "Periodo"},
                    color_discrete_sequence=['#ff7f0e']
                )
                fig_eps.update_layout(height=350)
                st.plotly_chart(fig_eps, use_container_width=True)
        
        # Seconda coppia: Azioni in circolazione e Dividendi
        col_chart3, col_chart4 = st.columns(2)
        
        with col_chart3:
            shares_field_options = ['weighted_average_shares_diluted', 'weighted_average_shares', 'diluted_average_shares']
            shares_field = get_balance_sheet_field(income_statements, shares_field_options)
            if shares_field:
                fig_shares = px.bar(
                    income_reset, 
                    x='Anno' if period == 'annual' else 'Quarter', 
                    y=shares_field,
                    title="Azioni in Circolazione",
                    labels={shares_field: "Numero di Azioni", 'Anno' if period == 'annual' else 'Quarter': "Periodo"},
                    color_discrete_sequence=['#2ca02c']
                )
                fig_shares.update_layout(height=350)
                st.plotly_chart(fig_shares, use_container_width=True)
        
        with col_chart4:
            dividend_field_options = ['dividends_per_common_share', 'dividends_per_share', 'dividend_per_share']
            dividend_field = get_balance_sheet_field(income_statements, dividend_field_options)
            if dividend_field:
                dividend_data = income_reset[income_reset[dividend_field] > 0] if not income_reset.empty else income_reset
                if not dividend_data.empty:
                    fig_dividends = px.bar(
                        dividend_data, 
                        x='Anno' if period == 'annual' else 'Quarter', 
                        y=dividend_field,
                        title="Dividendi per Azione",
                        labels={dividend_field: "Dividendi per Azione ($)", 'Anno' if period == 'annual' else 'Quarter': "Periodo"},
                        color_discrete_sequence=['#d62728']
                    )
                    fig_dividends.update_layout(height=350)
                    st.plotly_chart(fig_dividends, use_container_width=True)
                else:
                    st.info("ℹ️ Nessun dividendo pagato nel periodo selezionato")
        
        # Terza coppia: Payout Ratio e Margine Lordo
        col_chart5, col_chart6 = st.columns(2)
        
        with col_chart5:
            # Calcola Payout Ratio se disponibile
            if eps_field in income_reset.columns and dividend_field and dividend_field in income_reset.columns:
                income_reset['Payout Ratio'] = (income_reset[dividend_field] / income_reset[eps_field]) * 100
                payout_data = income_reset[(income_reset['Payout Ratio'] >= 0) & (income_reset['Payout Ratio'] <= 200)]
                
                if not payout_data.empty:
                    fig_payout = px.bar(
                        payout_data, 
                        x='Anno' if period == 'annual' else 'Quarter', 
                        y='Payout Ratio',
                        title="Payout Ratio (%)",
                        labels={"Payout Ratio": "Payout Ratio (%)", 'Anno' if period == 'annual' else 'Quarter': "Periodo"},
                        color_discrete_sequence=['#9467bd']
                    )
                    fig_payout.update_layout(height=350)
                    st.plotly_chart(fig_payout, use_container_width=True)
        
        with col_chart6:
            # Calcola Margine Lordo
            if 'revenue' in income_reset.columns and 'cost_of_revenue' in income_reset.columns:
                income_reset['Margine Lordo'] = ((income_reset['revenue'] - income_reset['cost_of_revenue']) / income_reset['revenue']) * 100
                margin_data = income_reset[(income_reset['Margine Lordo'] >= 0) & (income_reset['Margine Lordo'] <= 100)]
                
                if not margin_data.empty:
                    fig_gross_margin = px.bar(
                        margin_data, 
                        x='Anno' if period == 'annual' else 'Quarter', 
                        y='Margine Lordo',
                        title="Margine Lordo (%)",
                        labels={"Margine Lordo": "Margine Lordo (%)", 'Anno' if period == 'annual' else 'Quarter': "Periodo"},
                        color_discrete_sequence=['#8c564b']
                    )
                    fig_gross_margin.update_layout(height=350)
                    st.plotly_chart(fig_gross_margin, use_container_width=True)
            elif 'gross_profit' in income_reset.columns and 'revenue' in income_reset.columns:
                income_reset['Margine Lordo'] = (income_reset['gross_profit'] / income_reset['revenue']) * 100
                margin_data = income_reset[(income_reset['Margine Lordo'] >= 0) & (income_reset['Margine Lordo'] <= 100)]
                
                if not margin_data.empty:
                    fig_gross_margin = px.bar(
                        margin_data, 
                        x='Anno' if period == 'annual' else 'Quarter', 
                        y='Margine Lordo',
                        title="Margine Lordo (%)",
                        labels={"Margine Lordo": "Margine Lordo (%)", 'Anno' if period == 'annual' else 'Quarter': "Periodo"},
                        color_discrete_sequence=['#8c564b']
                    )
                    fig_gross_margin.update_layout(height=350)
                    st.plotly_chart(fig_gross_margin, use_container_width=True)

    # === GRAFICI BILANCIO ===
    if not balance_sheets.empty:
        st.subheader("Analisi Stato Patrimoniale")
        
        balance_reset = balance_sheets.reset_index()
        if 'Date' in balance_reset.columns:
            if period == 'annual':
                balance_reset['Anno'] = balance_reset['Date'].dt.strftime('%Y')
            else:
                balance_reset['Quarter'] = balance_reset['Date'].dt.strftime('%Y-Q%q')
        
        balance_reset = balance_reset.sort_values(by='Date', ascending=False).head(60)
        balance_reset = balance_reset.sort_values(by='Date')
        
        # Ottieni il numero di azioni per i calcoli
        shares_outstanding = (information.get("sharesOutstanding") or 
                             information.get("impliedSharesOutstanding") or
                             information.get("floatShares") or
                             additional_metrics.get("outstanding_shares"))
        
        col_balance1, col_balance2 = st.columns(2)
        
        with col_balance1:
            # Book Value per Share
            if shares_outstanding:
                bvps_field, equity_field = calculate_book_value_per_share(balance_sheets, shares_outstanding)
                if bvps_field:
                    balance_reset = balance_sheets.reset_index()
                    if 'Date' in balance_reset.columns:
                        if period == 'annual':
                            balance_reset['Anno'] = balance_reset['Date'].dt.strftime('%Y')
                        else:
                            balance_reset['Quarter'] = balance_reset['Date'].dt.strftime('%Y-Q%q')
                    
                    balance_reset = balance_reset.sort_values(by='Date')
                    
                    fig_bvps = px.bar(
                        balance_reset, 
                        x='Anno' if period == 'annual' else 'Quarter', 
                        y='Book Value per Share',
                        title="Book Value per Share",
                        labels={"Book Value per Share": "Book Value per Share ($)", 'Anno' if period == 'annual' else 'Quarter': "Periodo"},
                        color_discrete_sequence=['#17becf']
                    )
                    fig_bvps.update_layout(height=350)
                    st.plotly_chart(fig_bvps, use_container_width=True)
        
        with col_balance2:
            # Rapporto Debito/Equity
            debt_fields = ['total_debt', 'totalDebt', 'non_current_debt', 'long_term_debt']
            equity_fields = ['shareholders_equity', 'total_equity', 'stockholders_equity']
            
            debt_field = get_balance_sheet_field(balance_sheets, debt_fields)
            equity_field = get_balance_sheet_field(balance_sheets, equity_fields)
            
            if debt_field and equity_field:
                debt_equity_data = []
                for date in balance_sheets.index:
                    total_debt = balance_sheets.loc[date, debt_field] if balance_sheets.loc[date, debt_field] > 0 else 0
                    shareholders_equity = balance_sheets.loc[date, equity_field]
                    
                    if shareholders_equity > 0:
                        debt_to_equity = (total_debt / shareholders_equity) * 100
                        debt_equity_data.append({
                            'Date': date,
                            'Debt to Equity': debt_to_equity,
                            'Anno' if period == 'annual' else 'Quarter': date.strftime('%Y') if period == 'annual' else date.strftime('%Y-Q%q')
                        })
                
                if debt_equity_data:
                    debt_equity_df = pd.DataFrame(debt_equity_data)
                    debt_equity_df = debt_equity_df[(debt_equity_df['Debt to Equity'] >= 0) & (debt_equity_df['Debt to Equity'] <= 500)]
                    
                    if not debt_equity_df.empty:
                        fig_debt_equity = px.bar(
                            debt_equity_df, 
                            x='Anno' if period == 'annual' else 'Quarter', 
                            y='Debt to Equity',
                            title="Rapporto Debito/Equity (%)",
                            labels={"Debt to Equity": "Debito/Equity (%)", 'Anno' if period == 'annual' else 'Quarter': "Periodo"},
                            color_discrete_sequence=['#e377c2']
                        )
                        fig_debt_equity.add_hline(y=100, line_dash="dash", line_color="red", 
                                                annotation_text="Debito = Equity (100%)")
                        fig_debt_equity.update_layout(height=350)
                        st.plotly_chart(fig_debt_equity, use_container_width=True)

    # === GRAFICI CASH FLOW ===
    if not cashflow_statements.empty:
        st.subheader("Analisi Cash Flow")
        
        cashflow_reset = cashflow_statements.reset_index()
        if 'Date' in cashflow_reset.columns:
            if period == 'annual':
                cashflow_reset['Anno'] = cashflow_reset['Date'].dt.strftime('%Y')
            else:
                cashflow_reset['Quarter'] = cashflow_reset['Date'].dt.strftime('%Y-Q%q')
        
        cashflow_reset = cashflow_reset.sort_values(by='Date', ascending=False).head(60)
        cashflow_reset = cashflow_reset.sort_values(by='Date')
        
        # Prima coppia: Cash Flow Operativo e da Investimenti
        col_cf1, col_cf2 = st.columns(2)
        
        with col_cf1:
            ocf_field_options = ['operating_cash_flow', 'cash_from_operations', 'cash_flow_from_operations']
            ocf_field = get_balance_sheet_field(cashflow_statements, ocf_field_options)
            if ocf_field:
                fig_ocf = px.bar(
                    cashflow_reset, 
                    x='Anno' if period == 'annual' else 'Quarter', 
                    y=ocf_field,
                    title="Cash Flow Operativo",
                    labels={ocf_field: "Cash Flow Operativo ($)", 'Anno' if period == 'annual' else 'Quarter': "Periodo"},
                    color_discrete_sequence=['#2ca02c']
                )
                fig_ocf.update_layout(height=350)
                st.plotly_chart(fig_ocf, use_container_width=True)
        
        with col_cf2:
            icf_field_options = ['investing_cash_flow', 'cash_flow_from_investing_activities', 'investingCashFlow']
            icf_field = get_balance_sheet_field(cashflow_statements, icf_field_options)
            if icf_field:
                fig_icf = px.bar(
                    cashflow_reset, 
                    x='Anno' if period == 'annual' else 'Quarter', 
                    y=icf_field,
                    title="Cash Flow da Investimenti",
                    labels={icf_field: "Cash Flow da Investimenti ($)", 'Anno' if period == 'annual' else 'Quarter': "Periodo"},
                    color_discrete_sequence=['#d62728']
                )
                fig_icf.update_layout(height=350)
                st.plotly_chart(fig_icf, use_container_width=True)
        
        # Seconda coppia: Cash Flow per Share e Cash Flow da Finanziamenti
        col_cf3, col_cf4 = st.columns(2)
        
        with col_cf3:
            # Cash Flow per Share
            if shares_outstanding:
                cfps_field, source_field = calculate_cash_flow_per_share(cashflow_statements, shares_outstanding)
                if cfps_field:
                    cashflow_reset = cashflow_statements.reset_index()
                    if 'Date' in cashflow_reset.columns:
                        if period == 'annual':
                            cashflow_reset['Anno'] = cashflow_reset['Date'].dt.strftime('%Y')
                        else:
                            cashflow_reset['Quarter'] = cashflow_reset['Date'].dt.strftime('%Y-Q%q')
                    
                    cashflow_reset = cashflow_reset.sort_values(by='Date')
                    
                    title = "Cash Flow per Share"
                    if 'free_cash_flow' in source_field.lower():
                        subtitle = "(Free Cash Flow)"
                    elif 'calcolato' in source_field.lower():
                        subtitle = "(FCF Calcolato)"
                    else:
                        subtitle = "(Operating CF)"
                    
                    fig_cfps = px.bar(
                        cashflow_reset, 
                        x='Anno' if period == 'annual' else 'Quarter', 
                        y='Cash Flow per Share',
                        title=f"{title}<br><sub>{subtitle}</sub>",
                        labels={"Cash Flow per Share": "Cash Flow per Share ($)", 'Anno' if period == 'annual' else 'Quarter': "Periodo"},
                        color_discrete_sequence=['#9467bd']
                    )
                    fig_cfps.update_layout(height=350)
                    st.plotly_chart(fig_cfps, use_container_width=True)
        
        with col_cf4:
            fcf_financing_field_options = ['financing_cash_flow', 'cash_flow_from_financing_activities', 'financingCashFlow']
            fcf_financing_field = get_balance_sheet_field(cashflow_statements, fcf_financing_field_options)
            if fcf_financing_field:
                fig_fcf_financing = px.bar(
                    cashflow_reset, 
                    x='Anno' if period == 'annual' else 'Quarter', 
                    y=fcf_financing_field,
                    title="Cash Flow da Finanziamenti",
                    labels={fcf_financing_field: "Cash Flow da Finanziamenti ($)", 'Anno' if period == 'annual' else 'Quarter': "Periodo"},
                    color_discrete_sequence=['#ff7f0e']
                )
                fig_fcf_financing.update_layout(height=350)
                st.plotly_chart(fig_fcf_financing, use_container_width=True)

    # === GRAFICI MULTIPLI DI VALUTAZIONE ===
    if not income_statements.empty and not balance_sheets.empty:
        st.subheader("Indicatori di Prezzo Storici")
        
        # Recupera dati di prezzo storici degli ultimi 10 anni con frequenza mensile
        try:
            stock = yf.Ticker(symbol)
            
            # Calcola date esatte per gli ultimi 10 anni
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=10*365)  # 10 anni fa
            
            # Recupera dati mensili con date specifiche
            hist_10y = stock.history(start=start_date, end=end_date, interval='1mo')
            
            # Se non abbiamo abbastanza dati mensili, prova con dati settimanali
            if hist_10y.empty or len(hist_10y) < 12:
                st.info("Pochi dati mensili disponibili, utilizzo dati settimanali...")
                hist_10y = stock.history(start=start_date, end=end_date, interval='1wk')
            
            # Se ancora pochi dati, prova con dati giornalieri ma campionati
            if hist_10y.empty or len(hist_10y) < 50:
                st.info("Utilizzo dati giornalieri campionati...")
                hist_daily = stock.history(start=start_date, end=end_date, interval='1d')
                if not hist_daily.empty:
                    # Campiona ogni 30 giorni circa per simulare dati mensili
                    hist_10y = hist_daily.iloc[::30, :]
            
            if not hist_10y.empty:
                st.info(f"Dati recuperati dal {hist_10y.index[0].strftime('%Y-%m-%d')} al {hist_10y.index[-1].strftime('%Y-%m-%d')} ({len(hist_10y)} punti dati)")
                
                # Prepara i dati per i calcoli storici
                valuation_data = []
                
                # Ottieni EPS e Book Value più recenti per i calcoli
                latest_eps = None
                latest_book_value = None
                
                # EPS più recente
                eps_field = 'earnings_per_share_diluted' if 'earnings_per_share_diluted' in income_statements.columns else 'earnings_per_share'
                if eps_field in income_statements.columns:
                    # Ordina per data decrescente e prendi il più recente
                    sorted_income = income_statements.sort_index(ascending=False)
                    latest_eps = sorted_income[eps_field].iloc[0]
                
                # Book Value per Share più recente
                if shares_outstanding:
                    equity_field_options = ['shareholders_equity', 'total_equity', 'stockholders_equity']
                    equity_field = get_balance_sheet_field(balance_sheets, equity_field_options)
                    
                    if equity_field:
                        sorted_balance = balance_sheets.sort_index(ascending=False)
                        latest_equity = sorted_balance[equity_field].iloc[0]
                        latest_book_value = latest_equity / shares_outstanding
                
                # Se non abbiamo EPS dai dati finanziari, prova da Yahoo Finance
                if not latest_eps:
                    latest_eps = information.get('trailingEPS', information.get('ttmEPS'))
                
                # Se non abbiamo Book Value, prova a calcolarlo da Yahoo Finance
                if not latest_book_value:
                    book_value_yahoo = information.get('bookValue')
                    if book_value_yahoo:
                        latest_book_value = book_value_yahoo
                
                # Debug info
                st.info(f"🔍 Dati utilizzati: EPS = ${latest_eps:.2f}, Book Value = ${latest_book_value:.2f}" if latest_eps and latest_book_value else "⚠️ Alcuni dati finanziari mancanti")
                
                # Calcola PE e PBV per ogni periodo
                for date, row in hist_10y.iterrows():
                    close_price = row['Close']
                    
                    # Calcola PE Ratio
                    pe_ratio = None
                    if latest_eps and latest_eps > 0:
                        pe_ratio = close_price / latest_eps
                        # Filtra valori ragionevoli (PE tra 0 e 200)
                        if pe_ratio < 0 or pe_ratio > 200:
                            pe_ratio = None
                    
                    # Calcola PBV Ratio
                    pbv_ratio = None
                    if latest_book_value and latest_book_value > 0:
                        pbv_ratio = close_price / latest_book_value
                        # Filtra valori ragionevoli (PBV tra 0 e 50)
                        if pbv_ratio < 0 or pbv_ratio > 50:
                            pbv_ratio = None
                    
                    valuation_data.append({
                        'Date': date,
                        'Close_Price': close_price,
                        'PE Ratio': pe_ratio,
                        'PBV Ratio': pbv_ratio,
                        'Year_Month': date.strftime('%Y-%m')
                    })
                
                if valuation_data:
                    valuation_df = pd.DataFrame(valuation_data)
                    valuation_df['Date'] = pd.to_datetime(valuation_df['Date'])
                    
                    # Ordina per data per assicurarsi che sia cronologico
                    valuation_df = valuation_df.sort_values('Date')
                    
                    col_pe, col_pbv = st.columns(2)
                    
                    with col_pe:
                        # Grafico PE Ratio
                        pe_data = valuation_df.dropna(subset=['PE Ratio'])
                        if not pe_data.empty and len(pe_data) > 1:
                            fig_pe = px.line(
                                pe_data,
                                x='Date',
                                y='PE Ratio',
                                title=f"PE Ratio - {pe_data['Date'].dt.year.min()} a {pe_data['Date'].dt.year.max()}",
                                labels={"PE Ratio": "PE Ratio", "Date": "Data"}
                            )
                            
                            # Aggiungi linee di riferimento
                            avg_pe = pe_data['PE Ratio'].mean()
                            fig_pe.add_hline(y=avg_pe, line_dash="dash", line_color="orange", 
                                           annotation_text=f"Media: {avg_pe:.1f}")
                            
                            # PE attuale
                            current_pe = information.get('trailingPE')
                            if current_pe:
                                fig_pe.add_hline(y=current_pe, line_dash="dot", line_color="red", 
                                               annotation_text=f"PE Attuale: {current_pe:.1f}")
                            
                            fig_pe.update_layout(
                                height=400,
                                xaxis=dict(
                                    title="Data",
                                    tickformat="%Y-%m",
                                    dtick="M12"  # Tick ogni 12 mesi
                                )
                            )
                            fig_pe.update_traces(line_color='#1f77b4')
                            st.plotly_chart(fig_pe, use_container_width=True)
                            
                            # Mostra statistiche
                            with st.expander("Statistiche PE Ratio"):
                                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                with col_stat1:
                                    st.metric("PE Attuale", f"{current_pe:.2f}" if current_pe else "N/A")
                                with col_stat2:
                                    st.metric("PE Medio", f"{avg_pe:.2f}")
                                with col_stat3:
                                    st.metric("PE Min", f"{pe_data['PE Ratio'].min():.2f}")
                                with col_stat4:
                                    st.metric("PE Max", f"{pe_data['PE Ratio'].max():.2f}")
                        else:
                            st.info("Dati PE Ratio insufficienti per il grafico storico")
                    
                    with col_pbv:
                        # Grafico PBV Ratio
                        pbv_data = valuation_df.dropna(subset=['PBV Ratio'])
                        if not pbv_data.empty and len(pbv_data) > 1:
                            fig_pbv = px.line(
                                pbv_data,
                                x='Date',
                                y='PBV Ratio',
                                title=f"PBV Ratio - {pbv_data['Date'].dt.year.min()} a {pbv_data['Date'].dt.year.max()}",
                                labels={"PBV Ratio": "Price to Book Value", "Date": "Data"}
                            )
                            
                            # Aggiungi linee di riferimento
                            fig_pbv.add_hline(y=1, line_dash="dash", line_color="red", 
                                            annotation_text="PBV = 1")
                            
                            avg_pbv = pbv_data['PBV Ratio'].mean()
                            fig_pbv.add_hline(y=avg_pbv, line_dash="dash", line_color="orange", 
                                            annotation_text=f"Media: {avg_pbv:.1f}")
                            
                            # PBV attuale
                            current_pbv = information.get('priceToBook')
                            if current_pbv:
                                fig_pbv.add_hline(y=current_pbv, line_dash="dot", line_color="red", 
                                                annotation_text=f"PBV Attuale: {current_pbv:.1f}")
                            
                            fig_pbv.update_layout(
                                height=400,
                                xaxis=dict(
                                    title="Data",
                                    tickformat="%Y-%m",
                                    dtick="M12"  # Tick ogni 12 mesi
                                )
                            )
                            fig_pbv.update_traces(line_color='#ff7f0e')
                            st.plotly_chart(fig_pbv, use_container_width=True)
                            
                            # Mostra statistiche
                            with st.expander("Statistiche PBV Ratio"):
                                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                with col_stat1:
                                    st.metric("PBV Attuale", f"{current_pbv:.2f}" if current_pbv else "N/A")
                                with col_stat2:
                                    st.metric("PBV Medio", f"{avg_pbv:.2f}")
                                with col_stat3:
                                    st.metric("PBV Min", f"{pbv_data['PBV Ratio'].min():.2f}")
                                with col_stat4:
                                    st.metric("PBV Max", f"{pbv_data['PBV Ratio'].max():.2f}")
                        else:
                            st.info("Dati PBV Ratio insufficienti per il grafico storico")
                    
                    # Aggiungi analisi del posizionamento attuale
                    col_analysis1, col_analysis2 = st.columns(2)
                    
                    with col_analysis1:
                        if not pe_data.empty and current_pe:
                            pe_percentile = (pe_data['PE Ratio'] < current_pe).mean() * 100
                            if pe_percentile <= 25:
                                pe_status = "🟢 Sottovalutato"
                                pe_color = "green"
                            elif pe_percentile >= 75:
                                pe_status = "🔴 Sopravvalutato"
                                pe_color = "red"
                            else:
                                pe_status = "🟡 Neutrale"
                                pe_color = "orange"
                            
                            st.markdown(f"""
                            **PE Ratio Analysis:**
                            - Percentile storico: **{pe_percentile:.1f}%**
                            - Il PE attuale è superiore al {pe_percentile:.1f}% dei valori storici
                            """)
                    
                    with col_analysis2:
                        if not pbv_data.empty and current_pbv:
                            pbv_percentile = (pbv_data['PBV Ratio'] < current_pbv).mean() * 100
                            if pbv_percentile <= 25:
                                pbv_status = "🟢 Sottovalutato"
                            elif pbv_percentile >= 75:
                                pbv_status = "🔴 Sopravvalutato"
                            else:
                                pbv_status = "🟡 Neutrale"
                            
                            st.markdown(f"""
                            **PBV Ratio Analysis:**
                            - Percentile storico: **{pbv_percentile:.1f}%**
                            - Il PBV attuale è superiore al {pbv_percentile:.1f}% dei valori storici
                            """)
                
                # Informazioni aggiuntive sui multipli
                with st.expander("ℹ️ Come interpretare i multipli di valutazione"):
                    st.markdown("""
                    **PE Ratio (Price-to-Earnings):**
                    - Indica quanto gli investitori pagano per ogni dollaro di utili
                    - PE alto può indicare crescita attesa o sopravvalutazione
                    - PE basso può indicare sottovalutazione o problemi aziendali
                    - Confronta sempre con la media storica e del settore
                    
                    **PBV Ratio (Price-to-Book Value):**
                    - Rapporto tra prezzo di mercato e valore contabile per azione
                    - PBV < 1: azione scambia sotto il valore contabile
                    - PBV > 1: azione scambia sopra il valore contabile
                    - Utile per valutare se l'azione è cara o conveniente
                    
                    **Analisi Percentile:**
                    - 0-25%: Valutazione storicamente bassa (potenziale acquisto)
                    - 25-75%: Valutazione nella norma storica
                    - 75-100%: Valutazione storicamente alta (cautela)
                    """)
            else:
                st.warning("⚠️ Impossibile recuperare dati di prezzo storici per i multipli di valutazione")
                
        except Exception as e:
            st.error(f"Errore nel recupero dei dati storici: {str(e)}")
            st.info("Impossibile calcolare i multipli di valutazione storici")

# === SEZIONE CALCOLO VALORE INTRINSECO ===
st.header('Calcolo Valore Intrinseco')

if not financial_datasets_api_key or financial_datasets_api_key == "YOUR_FINANCIAL_DATASETS_API_KEY_HERE":
    st.warning("⚠️ Configura la chiave API FinancialDatasets nel codice per calcolare il valore intrinseco.")
else:
    # Utilizziamo i dati annuali per il calcolo del valore intrinseco
    if 'period' in locals() and period != 'annual':
        with st.spinner("📊 Caricamento dati annuali per il calcolo..."):
            annual_income = fetch_income_statements(symbol, financial_datasets_api_key, period='annual', limit=60)
            annual_balance = fetch_balance_sheets(symbol, financial_datasets_api_key, period='annual', limit=60)
            annual_cashflow = fetch_cashflow_statements(symbol, financial_datasets_api_key, period='annual', limit=60)
    elif 'income_statements' in locals():
        annual_income = income_statements
        annual_balance = balance_sheets
        annual_cashflow = cashflow_statements
    else:
        with st.spinner("📊 Caricamento dati finanziari..."):
            annual_income = fetch_income_statements(symbol, financial_datasets_api_key, period='annual', limit=60)
            annual_balance = fetch_balance_sheets(symbol, financial_datasets_api_key, period='annual', limit=60)
            annual_cashflow = fetch_cashflow_statements(symbol, financial_datasets_api_key, period='annual', limit=60)

    # Controllo se abbiamo dati sufficienti per il calcolo
    if annual_income.empty or annual_balance.empty or annual_cashflow.empty:
        st.warning("⚠️ Dati insufficienti per il calcolo del valore intrinseco.")
    else:
        # Stima del tasso di crescita degli utili
        earnings_growth = None
        if not annual_income.empty and len(annual_income) > 1:
            eps_column = 'earnings_per_share_diluted' if 'earnings_per_share_diluted' in annual_income.columns else 'earnings_per_share'
            
            if eps_column in annual_income.columns:
                sorted_data = annual_income.sort_index(ascending=False)
                
                if len(sorted_data) >= 2:
                    latest_eps = sorted_data[eps_column].iloc[0]
                    oldest_eps = sorted_data[eps_column].iloc[-1]
                    
                    years = (sorted_data.index[0] - sorted_data.index[-1]).days / 365.25
                    
                    if years > 0 and latest_eps > 0 and oldest_eps > 0:
                        earnings_growth = ((latest_eps / oldest_eps) ** (1/years) - 1) * 100

        # Crea tabs per i diversi metodi di valutazione
        tab1, tab2 = st.tabs(["DCF (Discounted Cash Flow)", "Formula di Graham"])
        
        with tab1:
            st.subheader("Metodo DCF (Discounted Cash Flow)")
            with st.expander("ℹ️ Come funziona il DCF", expanded=False):
                st.markdown("""
                Il **Discounted Cash Flow (DCF)** è un metodo di valutazione che stima il valore intrinseco di un'azienda 
                basandosi sui flussi di cassa futuri attesi, scontati al valore presente.
                
                **Formula**: Valore = Σ(FCF / (1+r)^t) + Valore Terminale
                
                **Parametri chiave**:
                - **Free Cash Flow**: flusso di cassa disponibile dopo investimenti
                - **Tasso di sconto**: rendimento richiesto dagli investitori
                - **Tasso di crescita**: crescita attesa dei flussi futuri
                """)
            
            intrinsic_value_dcf = calculate_dcf_value(information, annual_income, annual_cashflow, annual_balance)
            
        with tab2:
            st.subheader("Formula di Benjamin Graham")
            with st.expander("ℹ️ Come funziona la Formula di Graham", expanded=False):
                st.markdown("""
                La **Formula di Graham** è un metodo di valutazione sviluppato da Benjamin Graham, 
                considerato il padre dell'investimento value.
                
                **Formula**: V = EPS × (8.5 + 2g) × 4.4 / Y
                
                **Dove**:
                - **EPS**: Utile per azione
                - **g**: Tasso di crescita degli utili atteso
                - **Y**: Rendimento dei bond AAA corporate
                - **8.5**: P/E di base per un'azienda senza crescita
                """)
            
            intrinsic_value_graham = calculate_graham_value(information, annual_income, earnings_growth)

# Footer
st.markdown("---")
st.caption("Dati forniti da Yahoo Finance e FinancialDatasets AI")
