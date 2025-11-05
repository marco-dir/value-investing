import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import datetime
from datetime import timedelta
import json
import os

# Configurazione della pagina
st.set_page_config(
    page_title="Analisi Fondamentale Azioni",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS per rimuovere TUTTI i link anchor da tutti i titoli
hide_all_anchor_links = """
    <style>
    /* Nasconde tutti i link anchor su tutti i livelli di titolo */
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Nasconde gli elementi anchor di Streamlit */
    .css-10trblm, .css-16idsys, .css-1dp5vir {
        display: none !important;
    }
    
    /* Metodo universale per tutte le versioni di Streamlit */
    [data-testid="stMarkdownContainer"] a[href^="#"] {
        display: none !important;
    }
    
    /* Nasconde specificamente i viewer badge */
    .viewerBadge_container__1QSob,
    .viewerBadge_link__1S137,
    .styles_viewerBadge__1yB5_,
    [class*="viewerBadge"] {
        display: none !important;
    }
    </style>
"""

st.markdown(hide_all_anchor_links, unsafe_allow_html=True)

# === CONFIGURAZIONE API 
FMP_API_KEY = os.getenv("MY_DATASET_API_KEY")
PERPLEXITY_API_KEY = os.getenv("MY_SONAR_API_KEY")

# ID del Google Sheet (estratto dall'URL)
GOOGLE_SHEET_ID = os.getenv("MY_GOOGLES_ID")
GOOGLE_SHEET_GID = os.getenv("MY_GOOGLES_GID")

# ===== FUNZIONI DI RECUPERO DATI =====
@st.cache_data(ttl=3600)
def search_ticker_score(ticker):
    """Cerca il ticker nel Google Sheet e restituisce il punteggio dalla colonna G"""
    try:
        csv_url = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv&gid={GOOGLE_SHEET_GID}"
        df = pd.read_csv(csv_url)
        
        if len(df.columns) < 7:
            return None
        
        ticker_column = df.columns[2]
        score_column = df.columns[6]
        ticker_upper = ticker.upper()
        
        match = df[df[ticker_column].str.upper() == ticker_upper]
        
        if not match.empty:
            score = match[score_column].iloc[0]
            try:
                return float(score)
            except (ValueError, TypeError):
                return None
        
        return None
        
    except Exception as e:
        st.warning(f"Impossibile recuperare dati dal Google Sheet: {str(e)}")
        return None
        
@st.cache_data(ttl=3600)
def fetch_stock_info_fmp(symbol, api_key):
    """Recupera informazioni dal profilo FMP + quote real-time"""
    try:
        profile_url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={api_key}"
        profile_response = requests.get(profile_url)
        profile_data = profile_response.json()[0] if profile_response.json() else {}
        
        quote_url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={api_key}"
        quote_response = requests.get(quote_url)
        quote_data = quote_response.json()[0] if quote_response.json() else {}
        
        info = {
            'longName': profile_data.get('companyName'),
            'symbol': symbol,
            'currentPrice': quote_data.get('price'),
            'regularMarketChangePercent': quote_data.get('changesPercentage'),
            'marketCap': profile_data.get('mktCap'),
            'sector': profile_data.get('sector'),
            'industry': profile_data.get('industry'),
            'currency': profile_data.get('currency', 'USD'),
            'beta': profile_data.get('beta'),
            'trailingPE': quote_data.get('pe'),
            'priceToBook': quote_data.get('priceToBook'),
        }
        
        return info
    except Exception as e:
        st.error(f"Errore FMP: {str(e)}")
        return {}

def get_currency_symbol(currency_code):
    """Restituisce il simbolo della valuta dal codice"""
    currency_symbols = {
        'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥', 'CHF': 'CHF',
        'CAD': 'C$', 'AUD': 'A$', 'CNY': '¥', 'HKD': 'HK$', 'SEK': 'kr',
        'NOK': 'kr', 'DKK': 'kr', 'INR': '₹', 'BRL': 'R$', 'RUB': '₽',
        'KRW': '₩', 'MXN': '$', 'SGD': 'S$', 'NZD': 'NZ$', 'ZAR': 'R'
    }
    return currency_symbols.get(currency_code, currency_code)

def detect_query_language(query):
    """
    Rileva se la query è probabilmente italiana o inglese
    per prioritizzare il mercato corretto
    
    Returns:
        'IT' per italiano, 'US' per inglese
    """
    query_lower = query.lower().strip()
    
    # Liste di aziende italiane comuni (nomi completi o parziali)
    italian_companies = [
        'intesa', 'unicredit', 'eni', 'enel', 'ferrari', 'telecom', 
        'generali', 'fiat', 'stellantis', 'leonardo', 'poste',
        'saipem', 'tenaris', 'prysmian', 'moncler', 'campari',
        'amplifon', 'buzzi', 'diasorin', 'inwit', 'italgas',
        'recordati', 'snam', 'terna', 'pirelli', 'fineco'
    ]
    
    # Liste di aziende USA/globali comuni in inglese
    english_companies = [
        'apple', 'microsoft', 'google', 'amazon', 'meta', 'facebook',
        'tesla', 'nvidia', 'netflix', 'intel', 'amd', 'oracle',
        'cisco', 'ibm', 'dell', 'hp', 'adobe', 'salesforce',
        'paypal', 'visa', 'mastercard', 'disney', 'nike', 'coca',
        'pepsi', 'mcdonalds', 'starbucks', 'walmart', 'target',
        'boeing', 'ford', 'general motors', 'exxon', 'chevron',
        'pfizer', 'johnson', 'merck', 'abbvie', 'bristol'
    ]
    
    # Check per match esatti o parziali
    for company in italian_companies:
        if company in query_lower:
            return 'IT'
    
    for company in english_companies:
        if company in query_lower:
            return 'US'
    
    # Se non trova match, usa caratteristiche linguistiche
    
    # Caratteristiche tipiche italiane
    italian_chars = ['à', 'è', 'é', 'ì', 'ò', 'ù']
    if any(char in query_lower for char in italian_chars):
        return 'IT'
    
    # Parole chiave italiane comuni
    italian_keywords = ['spa', 's.p.a', 'srl', 'banca', 'gruppo']
    if any(keyword in query_lower for keyword in italian_keywords):
        return 'IT'
    
    # Se il ticker già specifica .MI o .BIT, è italiano
    if '.mi' in query_lower or '.bit' in query_lower:
        return 'IT'
    
    # Parole inglesi comuni nei nomi aziende USA
    english_keywords = ['corp', 'inc', 'ltd', 'technologies', 'systems', 'group']
    if any(keyword in query_lower for keyword in english_keywords):
        return 'US'
    
    # Default: se non siamo sicuri, usiamo inglese (più comune globalmente)
    # Questo aiuta con aziende globali come "Nestle", "Samsung", etc.
    return 'US'

@st.cache_data(ttl=3600)
def search_ticker_by_name(query, api_key, user_country=None):
    """
    Cerca ticker da nome azienda con priorità geografica automatica
    
    Args:
        query: Nome azienda o ticker
        api_key: Chiave API FMP
        user_country: Se None, rileva automaticamente dalla query
    """
    # Se sembra già un ticker completo, restituiscilo
    if query.isupper() and len(query) <= 6 and '.' not in query:
        return query
    
    if '.' in query and query.replace('.', '').replace('-', '').isalnum():
        return query.upper()
    
    # RILEVAMENTO AUTOMATICO PAESE se non specificato
    if user_country is None:
        user_country = detect_query_language(query)
    
    # Cerca via API
    url = f"https://financialmodelingprep.com/api/v3/search?query={query}&limit=25&apikey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        results = response.json()
        
        if not results or len(results) == 0:
            return None
        
        query_lower = query.lower().strip()
        
        # Mercati per paese
        country_exchanges = {
            'IT': ['MIL', 'BIT'],           # Italia
            'US': ['NYSE', 'NASDAQ'],       # USA
            'FR': ['PAR'],                  # Francia
            'DE': ['XETRA', 'F'],          # Germania
            'GB': ['LSE'],                  # Regno Unito
        }
        
        preferred_exchanges = country_exchanges.get(user_country, ['NYSE', 'NASDAQ'])
        main_exchanges = ['NASDAQ', 'NYSE', 'MIL', 'BIT', 'XETRA', 'PAR', 'LSE', 'F']
        
        # PRIORITÀ 0: Match esatto ticker + mercato domestico
        for result in results:
            ticker = result.get('symbol', '')
            exchange = result.get('exchangeShortName', '')
            ticker_base = ticker.split('.')[0]
            
            if ticker_base.upper() == query_lower.upper() and exchange in preferred_exchanges:
                return ticker
        
        # PRIORITÀ 1: Match esatto ticker + mercati principali
        for result in results:
            ticker = result.get('symbol', '')
            exchange = result.get('exchangeShortName', '')
            ticker_base = ticker.split('.')[0]
            
            if ticker_base.upper() == query_lower.upper() and exchange in main_exchanges:
                return ticker
        
        # PRIORITÀ 2: Match esatto nome + mercato domestico
        for result in results:
            name = result.get('name', '').lower()
            exchange = result.get('exchangeShortName', '')
            
            if query_lower == name and exchange in preferred_exchanges:
                return result.get('symbol')
        
        # PRIORITÀ 3: Match esatto nome
        for result in results:
            name = result.get('name', '').lower()
            if query_lower == name:
                return result.get('symbol')
        
        # PRIORITÀ 4: Nome inizia con query + mercato domestico
        for result in results:
            name = result.get('name', '').lower()
            exchange = result.get('exchangeShortName', '')
            
            if name.startswith(query_lower) and exchange in preferred_exchanges:
                return result.get('symbol')
        
        # PRIORITÀ 5: Nome inizia con query
        for result in results:
            name = result.get('name', '').lower()
            if name.startswith(query_lower):
                return result.get('symbol')
        
        # PRIORITÀ 6: Ticker inizia + mercato domestico
        for result in results:
            ticker = result.get('symbol', '').upper()
            exchange = result.get('exchangeShortName', '')
            ticker_base = ticker.split('.')[0]
            
            if ticker_base.startswith(query_lower.upper()) and exchange in preferred_exchanges:
                return result.get('symbol')
        
        # PRIORITÀ 7: Ticker inizia + mercati principali
        for result in results:
            ticker = result.get('symbol', '').upper()
            exchange = result.get('exchangeShortName', '')
            ticker_base = ticker.split('.')[0]
            
            if ticker_base.startswith(query_lower.upper()) and exchange in main_exchanges:
                return result.get('symbol')
        
        # PRIORITÀ 8: Nome contiene + mercato domestico
        for result in results:
            name = result.get('name', '').lower()
            exchange = result.get('exchangeShortName', '')
            
            if query_lower in name and exchange in preferred_exchanges:
                return result.get('symbol')
        
        # PRIORITÀ 9: Primo da mercato domestico
        for result in results:
            exchange = result.get('exchangeShortName', '')
            if exchange in preferred_exchanges:
                return result.get('symbol')
        
        # PRIORITÀ 10: Primo da mercati principali
        for result in results:
            exchange = result.get('exchangeShortName', '')
            if exchange in main_exchanges:
                return result.get('symbol')
        
        # FALLBACK
        return results[0].get('symbol')
            
    except Exception as e:
        st.warning(f"Errore nella ricerca del ticker: {str(e)}")
        return None

def display_search_suggestions(query, api_key, max_results=5, user_country=None):
    """
    Mostra suggerimenti ordinati per rilevanza geografica automatica
    """
    # RILEVAMENTO AUTOMATICO PAESE se non specificato
    if user_country is None:
        user_country = detect_query_language(query)
    
    url = f"https://financialmodelingprep.com/api/v3/search?query={query}&limit=25&apikey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        results = response.json()
        
        if not results or len(results) == 0:
            return []
        
        # Mercati per paese
        country_exchanges = {
            'IT': ['MIL', 'BIT'],
            'US': ['NYSE', 'NASDAQ'],
            'FR': ['PAR'],
            'DE': ['XETRA', 'F'],
            'GB': ['LSE']
        }
        
        preferred_exchanges = country_exchanges.get(user_country, ['NYSE', 'NASDAQ'])
        main_exchanges = ['NASDAQ', 'NYSE', 'MIL', 'BIT', 'XETRA', 'PAR', 'LSE', 'F']
        query_lower = query.lower().strip()
        
        filtered_results = []
        
        for result in results:
            symbol = result.get('symbol', 'N/A')
            name = result.get('name', 'N/A')
            exchange = result.get('exchangeShortName', '')
            
            if exchange not in main_exchanges:
                continue
            
            # Calcola score
            score = 0
            name_lower = name.lower()
            ticker_base = symbol.split('.')[0].lower()
            
            # Match esatto ticker
            if ticker_base == query_lower:
                score += 1000
            elif name_lower == query_lower:
                score += 900
            elif ticker_base.startswith(query_lower):
                score += 800
            elif name_lower.startswith(query_lower):
                score += 700
            elif query_lower in name_lower:
                score += 500
            else:
                score += 100
            
            # BONUS GEOGRAFICO
            if exchange in preferred_exchanges:
                score += 500
            
            # Bonus altri mercati
            exchange_priority = {
                'NYSE': 100, 
                'NASDAQ': 100,
                'MIL': 90,
                'BIT': 90,
                'XETRA': 80,
                'F': 80,
                'PAR': 80,
                'LSE': 80
            }
            
            if exchange not in preferred_exchanges:
                score += exchange_priority.get(exchange, 0)
            
            filtered_results.append({
                'symbol': symbol,
                'name': name,
                'exchange': exchange,
                'score': score,
                'is_domestic': exchange in preferred_exchanges
            })
        
        # Ordina per score
        filtered_results.sort(key=lambda x: x['score'], reverse=True)
        
        if filtered_results:
            # Mostra messaggio con mercato rilevato
            market_emoji = {
                'IT': '🇮🇹 Italia',
                'US': '🇺🇸 USA',
                'FR': '🇫🇷 Francia',
                'DE': '🇩🇪 Germania',
                'GB': '🇬🇧 UK'
            }
            market_label = market_emoji.get(user_country, user_country)
            
            st.info(f"🔍 **Risultati per '{query}'** (mercato prioritario: {market_label})")
            
            # Emoji bandiera per mercato rilevato
            flag_emoji = {
                'IT': '🇮🇹',
                'US': '🇺🇸',
                'FR': '🇫🇷',
                'DE': '🇩🇪',
                'GB': '🇬🇧'
            }
            
            for idx, result in enumerate(filtered_results[:max_results], 1):
                if result['is_domestic']:
                    flag = flag_emoji.get(user_country, '')
                    prefix = f"⭐{flag}"
                elif idx == 1:
                    prefix = "⭐"
                else:
                    prefix = f"{idx}."
                
                st.markdown(f"{prefix} **{result['symbol']}** - {result['name']} *({result['exchange']})*")
            
            return [r['symbol'] for r in filtered_results[:max_results]]
        else:
            return []
            
    except Exception as e:
        st.error(f"Errore nel recupero suggerimenti: {str(e)}")
        return []

@st.cache_data
def fetch_financial_metrics_fmp(symbol, api_key):
    """Recupera metriche finanziarie da FMP"""
    url = f"https://financialmodelingprep.com/api/v3/key-metrics/{symbol}?apikey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data[0] if data else {}
    except Exception as e:
        st.warning(f"Impossibile recuperare metriche aggiuntive da FMP: {str(e)}")
        return {}

@st.cache_data
def fetch_company_profile_fmp(symbol, api_key):
    """Recupera il profilo dell'azienda da FMP"""
    url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data[0] if data else {}
    except Exception as e:
        st.warning(f"Impossibile recuperare il profilo dell'azienda da FMP: {str(e)}")
        return {}

@st.cache_data
def fetch_financial_ratios_fmp(symbol, api_key):
    """Recupera i ratio finanziari da FMP incluso Quick Ratio e ROA"""
    url = f"https://financialmodelingprep.com/api/v3/ratios/{symbol}?apikey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data[0] if data else {}
    except Exception as e:
        st.warning(f"Impossibile recuperare i ratio finanziari da FMP: {str(e)}")
        return {}

@st.cache_data(ttl=3600)
def fetch_price_history_fmp(symbol, api_key):
    """Storico prezzi da FMP"""
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if 'historical' in data:
            df = pd.DataFrame(data['historical'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.sort_index(ascending=False).head(252)
            df = df.sort_index()
            
            column_mapping = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            df = df.rename(columns=column_mapping)
            
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Errore: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def fetch_income_statements_fmp(symbol, api_key, period='annual', limit=60):
    """Recupera i dati del conto economico da FMP"""
    period_param = 'annual' if period == 'annual' else 'quarter'
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period={period_param}&limit={limit}&apikey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        statements = response.json()
        
        if not statements:
            return pd.DataFrame()
        
        df = pd.DataFrame(statements)
        
        if 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'])
            df.set_index('Date', inplace=True)
            df = df.sort_index(ascending=False)
            
        return df
    except Exception as e:
        st.error(f"Errore nel recupero dei dati del conto economico: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def fetch_balance_sheets_fmp(symbol, api_key, period='annual', limit=60):
    """Recupera i dati dello stato patrimoniale da FMP"""
    period_param = 'annual' if period == 'annual' else 'quarter'
    url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}?period={period_param}&limit={limit}&apikey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        statements = response.json()
        
        if not statements:
            return pd.DataFrame()
        
        df = pd.DataFrame(statements)
        
        if 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'])
            df.set_index('Date', inplace=True)
            df = df.sort_index(ascending=False)
            
        return df
    except Exception as e:
        st.error(f"Errore nel recupero dei dati dello stato patrimoniale: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def fetch_cashflow_statements_fmp(symbol, api_key, period='annual', limit=60):
    """Recupera i dati del rendiconto finanziario da FMP"""
    period_param = 'annual' if period == 'annual' else 'quarter'
    url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}?period={period_param}&limit={limit}&apikey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        statements = response.json()
        
        if not statements:
            return pd.DataFrame()
        
        df = pd.DataFrame(statements)
        
        if 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'])
            df.set_index('Date', inplace=True)
            df = df.sort_index(ascending=False)
            
        return df
    except Exception as e:
        st.error(f"Errore nel recupero dei dati del rendiconto finanziario: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def fetch_dividend_history_fmp(symbol, api_key):
    """Recupera lo storico dei dividendi da FMP"""
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{symbol}?apikey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if 'historical' in data and data['historical']:
            df = pd.DataFrame(data['historical'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False)
            return df
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Impossibile recuperare lo storico dei dividendi: {str(e)}")
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
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Errore nell'analisi: {str(e)}"

# ===== FUNZIONI UTILITÀ =====

def get_balance_sheet_field(df, field_options):
    """Trova il primo campo disponibile da una lista di opzioni"""
    for field in field_options:
        if field in df.columns:
            return field
    return None

def safe_format(value, format_str):
    try:
        if value is None:
            return "N/A"
        return format_str.format(value)
    except:
        return "N/A"

def should_be_green(value, condition_type):
    """Determina se un valore soddisfa le condizioni per essere colorato di verde"""
    if value is None or value == "N/A":
        return False
    
    try:
        if isinstance(value, str):
            numeric_value = float(value.replace('%', '').replace(',', '').replace('$', '').replace('€', '').replace('£', '').replace('¥', ''))
            is_percentage_string = '%' in value
        else:
            numeric_value = float(value)
            is_percentage_string = False
        
        if condition_type in ["roic", "roe", "roa"]:
            if is_percentage_string:
                threshold_roic = 10.0
                threshold_roe = 15.0 
                threshold_roa = 10.0
            else:
                threshold_roic = 0.10
                threshold_roe = 0.15
                threshold_roa = 0.10
            
            if condition_type == "roic":
                return numeric_value > threshold_roic
            elif condition_type == "roe":
                return numeric_value > threshold_roe  
            elif condition_type == "roa":
                return numeric_value > threshold_roa
        
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

def format_indicator_value(value, condition_type):
    """Formatta un valore e restituisce sia il valore che se deve essere verde"""
    if condition_type in ["roic", "roe", "roa"]:
        if value is not None and value != 0:
            if value < 1:
                formatted_value = f"{value * 100:.2f}%"
                is_green = should_be_green(value, condition_type)
            else:
                formatted_value = f"{value:.2f}%"
                is_green = should_be_green(formatted_value, condition_type)
        else:
            formatted_value = "N/A"
            is_green = False
    elif condition_type == "debt_equity":
        if value is not None and value != 0:
            formatted_value = f"{value * 100:.2f}%"
            is_green = should_be_green(value, condition_type)
        else:
            formatted_value = "N/A"
            is_green = False
    elif condition_type in ["operating_margin", "profit_margin", "gross_margin"]:
        if value is not None and value != 0:
            formatted_value = f"{value * 100:.2f}%"
            is_green = value > 0.10
        else:
            formatted_value = "N/A"
            is_green = False
    else:
        formatted_value = safe_format(value, "{:.2f}")
        is_green = should_be_green(formatted_value, condition_type)
    
    return formatted_value, is_green

# ========== APPLICAZIONE PRINCIPALE ==========

st.title('Analisi Fondamentale Azioni')
st.caption('Analisi Fondamentale secondo i principi del Value Investing interpretati da DIRAMCO')

# Input Ticker
st.header("Seleziona Titolo del Mercato USA, Europa e Asia")
col_ticker, col_button = st.columns([3, 1])

with col_ticker:
    symbol = st.text_input(
        'Inserisci il Ticker o Nome del Titolo Azionario', 
        'AAPL', 
        help="Es. AAPL, GOOGL, Microsoft, Intesa, Unicredit, G.MI, MC.PA, SAP.DE",
        placeholder="Inserisci ticker o nome azienda..."
    )

with col_button:
    st.write("")
    st.write("")
    analyze_button = st.button("Analizza", type="primary", use_container_width=True)

if not symbol:
    st.info("Inserisci il ticker o nome di un titolo per iniziare l'analisi")
    st.stop()

# Normalizza input (rimuovi spazi extra)
symbol = symbol.strip()

# Se l'input sembra un nome (contiene spazi o lettere minuscole), cerca il ticker
original_input = symbol
if ' ' in symbol or not symbol.isupper():
    with st.spinner(f"🔍 Ricerca ticker per '{symbol}'..."):
        found_ticker = search_ticker_by_name(symbol, FMP_API_KEY)
        
        if found_ticker:
            symbol = found_ticker
            st.success(f"✅ Trovato: **{found_ticker}**")
        else:
            st.warning(f"⚠️ Nessun ticker trovato per '{symbol}'. Prova con questi suggerimenti:")
            suggestions = display_search_suggestions(original_input, FMP_API_KEY, max_results=5)
            
            if suggestions:
                # Permetti selezione
                selected = st.selectbox(
                    "Seleziona il ticker corretto:",
                    options=[""] + suggestions,
                    format_func=lambda x: "Seleziona..." if x == "" else x
                )
                
                if selected:
                    symbol = selected
                    st.rerun()
                else:
                    st.stop()
            else:
                st.error("Nessun risultato trovato. Verifica il nome o ticker inserito.")
                st.stop()

# Converti ticker in maiuscolo per sicurezza
symbol = symbol.upper()

# Recupera informazioni di base
with st.spinner(f" Caricamento dati per {symbol}..."):
    information = fetch_stock_info_fmp(symbol, FMP_API_KEY)

if not information:
    st.error(f"❌ Nessuna informazione trovata per il ticker {symbol}.")
    st.warning("Prova con uno di questi formati:")
    st.markdown("""
    - Ticker USA: `AAPL`, `MSFT`, `GOOGL`
    - Ticker Europa: `ISP.MI` (Italia), `MC.PA` (Francia), `SAP.DE` (Germania)
    - Nome: `Apple`, `Microsoft`, `Intesa Sanpaolo`
    """)
    st.stop()

# Ottieni la valuta del titolo
currency_code = information.get('currency', 'USD')
currency_symbol = get_currency_symbol(currency_code)

# Recupera metriche aggiuntive da FMP
additional_metrics = {}
company_profile = {}
financial_ratios = {}
income_statements_preview = pd.DataFrame()

if FMP_API_KEY and FMP_API_KEY != "YOUR_FMP_API_KEY_HERE":
    additional_metrics = fetch_financial_metrics_fmp(symbol, FMP_API_KEY)
    company_profile = fetch_company_profile_fmp(symbol, FMP_API_KEY)
    financial_ratios = fetch_financial_ratios_fmp(symbol, FMP_API_KEY)
    income_statements_preview = fetch_income_statements_fmp(symbol, FMP_API_KEY, period='annual', limit=1)

# === SEZIONE INFORMAZIONI TITOLO E GRAFICO ===
st.header('Panoramica Titolo')

col_info, col_chart = st.columns([1, 2])

with col_info:
    company_name = information.get("longName", information.get("shortName", symbol))
    current_price = information.get("currentPrice")  # ← Rimosso default 0
    percent_change = information.get("regularMarketChangePercent")
    
    st.subheader(f'{company_name}')
    st.caption(f"Valuta: {currency_code} ({currency_symbol})")
    
    # === FIX: Gestione None ===
    if current_price is not None and current_price > 0:
        st.metric(
            label="Prezzo Attuale",
            value=f"{currency_symbol}{current_price:.2f}",
            delta=f"{'+' if percent_change and percent_change >= 0 else ''}{percent_change:.2f}%" if percent_change is not None else None
        )
    else:
        st.metric(
            label="Prezzo Attuale",
            value="N/A",
            help="Prezzo non disponibile dalle API"
        )
        st.warning("⚠️ Prezzo di mercato non disponibile per questo titolo")

    market_cap = information.get("marketCap", 0)
    if market_cap:
        st.metric(
            label="Capitalizzazione",
            value=f"{currency_symbol}{market_cap/1000000000:.1f}B"
        )

    sector = information.get("sector", company_profile.get("sector", "N/A"))
    st.info(f'**Settore:** {sector}')

    st.subheader("Indicatori Chiave")

    col_ind1, col_ind2 = st.columns(2)

    with col_ind1:
        pe_ratio = information.get("trailingPE", additional_metrics.get("peRatio", 0))
        if pe_ratio:
            st.metric("P/E Ratio", f"{pe_ratio:.2f}")
        else:
            st.metric("P/E Ratio", "N/A")
        
        # FIX: Rendimento Dividendo - Prova più fonti
        dividend_yield = None
        
        # 1. Prova da informazioni base
        dividend_yield = information.get("dividendYield")
        if dividend_yield and dividend_yield > 0:
            st.metric("Rendimento Dividendo", f"{dividend_yield*100:.2f}%")
        else:
            # 2. Calcola da dividend rate e prezzo
            dividend_rate = information.get("dividendRate")
            current_price_val = information.get("currentPrice", information.get("regularMarketPrice"))
            if dividend_rate and current_price_val and current_price_val > 0:
                calculated_yield = (dividend_rate / current_price_val) * 100
                st.metric("Rendimento Dividendo", f"{calculated_yield:.2f}%")
            else:
                # 3. Prova da FMP - calcola da storico dividendi ultimi 12 mesi
                dividend_history = fetch_dividend_history_fmp(symbol, FMP_API_KEY)
                if not dividend_history.empty and current_price_val and current_price_val > 0:
                    one_year_ago = datetime.datetime.now() - timedelta(days=365)
                    recent_divs = dividend_history[dividend_history['date'] >= one_year_ago]
                    if not recent_divs.empty:
                        dividend_field = 'adjDividend' if 'adjDividend' in recent_divs.columns else 'dividend'
                        annual_dividend = recent_divs[dividend_field].sum()
                        if annual_dividend > 0:
                            calculated_yield = (annual_dividend / current_price_val) * 100
                            st.metric("Rendimento Dividendo", f"{calculated_yield:.2f}%")
                        else:
                            st.metric("Rendimento Dividendo", "N/A")
                    else:
                        st.metric("Rendimento Dividendo", "N/A")
                else:
                    st.metric("Rendimento Dividendo", "N/A")

    with col_ind2:
        # FIX: P/B Ratio
        pb_ratio = information.get("priceToBook")
        if not pb_ratio:
            pb_ratio = additional_metrics.get("pbRatio")
        
        if pb_ratio:
            st.metric("P/B Ratio", f"{pb_ratio:.2f}")
        else:
            # Prova a calcolarlo manualmente
            current_price_val = information.get("currentPrice", information.get("regularMarketPrice"))
            book_value = information.get("bookValue")
            
            if current_price_val and book_value and book_value > 0:
                pb_ratio = current_price_val / book_value
                st.metric("P/B Ratio", f"{pb_ratio:.2f}")
            else:
                st.metric("P/B Ratio", "N/A")
        
        beta = information.get("beta", company_profile.get("beta", 0))
        if beta:
            st.metric("Beta", f"{beta:.2f}")
        else:
            st.metric("Beta", "N/A")

with col_chart:
    price_history = fetch_price_history_fmp(symbol, FMP_API_KEY)
    
    if not price_history.empty:
        st.subheader('Grafico 1 Anno')
        
        try:
            price_history_clean = price_history.copy()
            
            # RIMUOVI TIMEZONE dall'indice
            if isinstance(price_history_clean.index, pd.DatetimeIndex):
                if price_history_clean.index.tz is not None:
                    price_history_clean.index = price_history_clean.index.tz_localize(None)
            
            # Filtra solo l'ultimo anno
            end_date = datetime.datetime.now()
            start_date = end_date - timedelta(days=365)
            
            price_history_clean = price_history_clean[price_history_clean.index >= start_date]
            
            if price_history_clean.empty:
                price_history_clean = price_history.copy()
                if isinstance(price_history_clean.index, pd.DatetimeIndex) and price_history_clean.index.tz is not None:
                    price_history_clean.index = price_history_clean.index.tz_localize(None)
            
            # Reset index e prepara dati
            price_history_reset = price_history_clean.reset_index()
            
            if price_history_reset.columns[0] != 'Date':
                price_history_reset.rename(columns={price_history_reset.columns[0]: 'Date'}, inplace=True)
            
            price_history_reset['Date'] = pd.to_datetime(price_history_reset['Date'])
            if hasattr(price_history_reset['Date'].dtype, 'tz') and price_history_reset['Date'].dt.tz is not None:
                price_history_reset['Date'] = price_history_reset['Date'].dt.tz_localize(None)
            
            price_history_reset = price_history_reset.sort_values('Date')
            price_history_reset = price_history_reset.dropna(subset=['Open', 'High', 'Low', 'Close'])
            
            st.caption(f" Dati dal {price_history_reset['Date'].min().strftime('%d/%m/%Y')} al {price_history_reset['Date'].max().strftime('%d/%m/%Y')} ({len(price_history_reset)} giorni)")
            
            if len(price_history_reset) < 2:
                st.error("Dati insufficienti per creare il grafico")
            else:
                fig = go.Figure()
                
                fig.add_trace(go.Candlestick(
                    x=price_history_reset['Date'].tolist(),
                    open=price_history_reset['Open'].tolist(),
                    high=price_history_reset['High'].tolist(),
                    low=price_history_reset['Low'].tolist(),
                    close=price_history_reset['Close'].tolist(),
                    name=symbol,
                    increasing={'line': {'color': '#26a69a'}, 'fillcolor': '#26a69a'},
                    decreasing={'line': {'color': '#ef5350'}, 'fillcolor': '#ef5350'}
                ))
                
                fig.update_layout(
                    height=500,
                    margin=dict(l=50, r=50, t=30, b=50),
                    xaxis_title="Data",
                    yaxis_title=f"Prezzo ({currency_symbol})",
                    xaxis_rangeslider_visible=False,
                    showlegend=False,
                    hovermode='x unified',
                    template='plotly_white'
                )
                
                fig.update_xaxes(
                    type='date',
                    tickformat='%b %Y',
                    tickmode='auto',
                    nticks=10
                )
                
                fig.update_yaxes(
                    tickprefix=currency_symbol,
                    side='right'
                )
                
                st.plotly_chart(fig, use_container_width=True, key='candlestick_chart')
                
                # Calcola YTD
                current_year = datetime.datetime.now().year
                ytd_data = price_history_reset[price_history_reset['Date'].dt.year == current_year]
                
                ytd_change_pct = None
                if not ytd_data.empty and len(ytd_data) > 0:
                    ytd_first_close = ytd_data['Close'].iloc[0]
                    ytd_last_close = ytd_data['Close'].iloc[-1]
                    ytd_change_pct = ((ytd_last_close - ytd_first_close) / ytd_first_close) * 100
                
                # Statistiche
                col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
                
                with col_stat1:
                    st.metric("Massimo 52W", f"{currency_symbol}{price_history_reset['High'].max():.2f}")
                
                with col_stat2:
                    st.metric("Minimo 52W", f"{currency_symbol}{price_history_reset['Low'].min():.2f}")
                
                with col_stat3:
                    if len(price_history_reset) > 1:
                        first_close = price_history_reset['Close'].iloc[0]
                        last_close = price_history_reset['Close'].iloc[-1]
                        change_pct = ((last_close - first_close) / first_close) * 100
                        st.metric("Variazione 1Y", f"{change_pct:+.2f}%", delta_color="normal")
                
                with col_stat4:
                    if ytd_change_pct is not None:
                        st.metric("Variazione YTD", f"{ytd_change_pct:+.2f}%", delta_color="normal")
                    else:
                        st.metric("Variazione YTD", "N/A")
                
                with col_stat5:
                    if 'Volume' in price_history_reset.columns:
                        avg_volume = price_history_reset['Volume'].mean()
                        if avg_volume >= 1000000:
                            st.metric("Volume medio", f"{avg_volume/1000000:.1f}M")
                        elif avg_volume >= 1000:
                            st.metric("Volume medio", f"{avg_volume/1000:.1f}K")
                        else:
                            st.metric("Volume medio", f"{avg_volume:.0f}")
                
        except Exception as e:
            st.error(f"❌ Errore nella creazione del grafico: {str(e)}")
            import traceback
            with st.expander("🔍 Traceback completo"):
                st.code(traceback.format_exc())
        
        st.caption('Visualizza Analisi Tecnica Avanzata tramite IA [qui](https://diramco.com/analisi-tecnica-ai/)')
    else:
        st.warning("⚠️ Dati dei prezzi storici non disponibili.")

# === SEZIONE INDICATORI FINANZIARI DETTAGLIATI ===
st.header('Indicatori Finanziari Dettagliati')

# Recupera dati da FMP prioritariamente
pe_ratio = additional_metrics.get("peRatio")
if not pe_ratio:
    pe_ratio = information.get("trailingPE", 0)

pb_ratio = additional_metrics.get("pbRatio")
if not pb_ratio:
    pb_ratio = information.get("priceToBook", 0)

price_to_sales = additional_metrics.get("priceToSalesRatio")
if not price_to_sales:
    price_to_sales = information.get("priceToSalesTrailing12Months", 0)

# Calcola PEG Ratio
peg_ratio = None
earnings_quarterly_growth = additional_metrics.get("earningsYield")
if pe_ratio and earnings_quarterly_growth and earnings_quarterly_growth > 0:
    eps_growth = earnings_quarterly_growth
    if eps_growth > 1:
        eps_growth = eps_growth / 100
    peg_ratio = pe_ratio / (eps_growth * 100)
else:
    eps_growth = information.get("earningsGrowth", 0)
    if pe_ratio and eps_growth and eps_growth > 0:
        peg_ratio = pe_ratio / (eps_growth * 100)

# ROE, ROA, ROIC da FMP
roe = additional_metrics.get("roe")
if roe and roe > 1:
    roe = roe / 100
if not roe:
    roe = information.get("returnOnEquity", 0)

# FIX ROA - Prova più fonti
roa = additional_metrics.get("returnOnAssets")
if roa and roa > 1:
    roa = roa / 100
if not roa or roa == 0:
    # Prova da financial_ratios
    roa = financial_ratios.get("returnOnAssets")
    if roa and roa > 1:
        roa = roa / 100
if not roa or roa == 0:
    roa = information.get("returnOnAssets", 0)

roic = additional_metrics.get('roic')
if roic and roic > 1:
    roic = roic / 100
if not roic:
    roic = roa

# Debt to Equity
debt_equity_ratio = additional_metrics.get("debtToEquity")
if debt_equity_ratio and debt_equity_ratio > 100:
    debt_equity_ratio = debt_equity_ratio / 100
if not debt_equity_ratio:
    debt_equity_ratio = information.get("debtToEquity", 0)
    if debt_equity_ratio and debt_equity_ratio > 100:
        debt_equity_ratio = debt_equity_ratio / 100

# Margini
gross_margins = None
operating_margin = None
profit_margin = None

if not income_statements_preview.empty:
    latest_income = income_statements_preview.iloc[0]
    
    if 'revenue' in latest_income and 'costOfRevenue' in latest_income and latest_income['revenue'] > 0:
        gross_margins = (latest_income['revenue'] - latest_income['costOfRevenue']) / latest_income['revenue']
    elif 'grossProfit' in latest_income and 'revenue' in latest_income and latest_income['revenue'] > 0:
        gross_margins = latest_income['grossProfit'] / latest_income['revenue']
    
    if 'operatingIncome' in latest_income and 'revenue' in latest_income and latest_income['revenue'] > 0:
        operating_margin = latest_income['operatingIncome'] / latest_income['revenue']
    
    if 'netIncome' in latest_income and 'revenue' in latest_income and latest_income['revenue'] > 0:
        profit_margin = latest_income['netIncome'] / latest_income['revenue']

# Fallback
if gross_margins is None:
    gross_margins = additional_metrics.get("grossProfitMargin")
    if gross_margins and gross_margins > 1:
        gross_margins = gross_margins / 100
    if gross_margins is None:
        gross_margins = information.get("grossMargins", 0)

if operating_margin is None:
    operating_margin = additional_metrics.get("operatingProfitMargin")
    if operating_margin and operating_margin > 1:
        operating_margin = operating_margin / 100
    if operating_margin is None:
        operating_margin = information.get("operatingMargins", 0)

if profit_margin is None:
    profit_margin = additional_metrics.get("netProfitMargin")
    if profit_margin and profit_margin > 1:
        profit_margin = profit_margin / 100
    if profit_margin is None:
        profit_margin = information.get("profitMargins", 0)

# Liquidità
current_ratio = additional_metrics.get("currentRatio")
if not current_ratio:
    current_ratio = information.get("currentRatio", 0)

# FIX Quick Ratio - Prova più fonti
quick_ratio = financial_ratios.get("quickRatio")
if not quick_ratio or quick_ratio == 0:
    quick_ratio = additional_metrics.get("quickRatio")
if not quick_ratio or quick_ratio == 0:
    quick_ratio = information.get("quickRatio", 0)

# Beta
beta = company_profile.get("beta")
if not beta:
    beta = information.get("beta", 0)

# Formattazione valori
pe_value, pe_is_green = format_indicator_value(pe_ratio, "pe")
pb_value, pb_is_green = format_indicator_value(pb_ratio, "pb")
peg_value, peg_is_green = format_indicator_value(peg_ratio, "peg") if peg_ratio else ("N/A", False)
ps_value, ps_is_green = format_indicator_value(price_to_sales, "ps")
debt_eq_value, debt_eq_is_green = format_indicator_value(debt_equity_ratio, "debt_equity")
roic_value, roic_is_green = format_indicator_value(roic, "roic")
roe_value, roe_is_green = format_indicator_value(roe, "roe")
roa_value, roa_is_green = format_indicator_value(roa, "roa")

# Layout tabelle
col_table1, col_table2 = st.columns(2)

with col_table1:
    st.subheader("Indicatori di Prezzo")
    
    indicators_price = [
        ("P/E Ratio", pe_value, pe_is_green, "Prezzo/Utili - Quanto si paga per 1 unità di utili"),
        ("P/B Ratio", pb_value, pb_is_green, "Prezzo/Valore Contabile - Rapporto con patrimonio netto"),
        ("P/S Ratio", ps_value, ps_is_green, "Prezzo/Ricavi - Valutazione basata sui ricavi"),
        ("PEG Ratio", peg_value, peg_is_green, "P/E aggiustato per crescita - Ideale tra 0.5-2.0")
    ]
    
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
    
    indicators_performance = [
        ("ROE", roe_value, roe_is_green, "Return on Equity - Redditività del capitale proprio"),
        ("ROA", roa_value, roa_is_green, "Return on Assets - Efficienza nell'uso degli asset"),
        ("ROIC", roic_value, roic_is_green, "Return on Invested Capital - Redditività capitale investito"),
        ("Debt/Equity", debt_eq_value, debt_eq_is_green, "Rapporto Debito/Equity - Leva finanziaria")
    ]
    
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

# === SEZIONE MARGINI E LIQUIDITÀ ===
st.subheader("Indicatori di Redditività e Liquidità")

col_margin1, col_margin2 = st.columns(2)

with col_margin1:
    st.write("**Margini di Redditività**")
    
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
            if value and value > 0.10:
                st.success(f"✅ {value*100:.2f}%")
            elif value and value > 0:
                st.write(f"{value*100:.2f}%")
            else:
                st.write("❌ N/A")

with col_margin2:
    st.write("**Indicatori di Liquidità**")
    
    liquidity_indicators = [
        ("Current Ratio", current_ratio, "Attività correnti / Passività correnti"),
        ("Quick Ratio", quick_ratio, "(Liquidità + Crediti) / Passività correnti"),
        ("Beta", beta, "Volatilità rispetto al mercato")
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

# === LEGENDA ===
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
    - **Debt/Equity < 100%**: Leva finanziaria controllata ✅
    
    ### Margini e Liquidità
    - **Margini > 10%**: Buona redditività operativa ✅
    - **Current Ratio > 1.5**: Buona liquidità a breve termine ✅
    - **Quick Ratio > 1.0**: Liquidità immediata sufficiente ✅
    - **Beta 0.5-1.5**: Volatilità in linea con il mercato ✅
    
    ### Score DIRAMCO
    - **≥ 9**: Qualità eccellente 🟢🟢
    - **≥ 8**: Qualità molto buona 🟢
    - **≥ 6**: Qualità accettabile 🟡
    - **< 6**: Cautela necessaria 🔴
    """)

def calculate_dcf_value(info, annual_financials, annual_cashflow, annual_balance_sheet, currency_symbol):
    """Calcolo valore intrinseco con metodo DCF"""
    try:
        discount_rate = st.slider("Tasso di Sconto (%)", min_value=5.0, max_value=20.0, value=10.0, step=0.5) / 100
        growth_rate_initial = st.slider("Tasso di Crescita Iniziale (%)", min_value=1.0, max_value=30.0, value=15.0, step=0.5) / 100
        growth_rate_terminal = st.slider("Tasso di Crescita Terminale (%)", min_value=1.0, max_value=5.0, value=2.5, step=0.1) / 100
        forecast_period = st.slider("Periodo di Previsione (anni)", min_value=5, max_value=20, value=10)
        
        # FMP usa 'freeCashFlow' direttamente
        fcf_field_options = ['freeCashFlow']
        fcf_field = get_balance_sheet_field(annual_cashflow, fcf_field_options)
        
        if fcf_field is None:
            ocf_field_options = ['operatingCashFlow', 'netCashProvidedByOperatingActivities']
            fcf_field = get_balance_sheet_field(annual_cashflow, ocf_field_options)
            
            if fcf_field:
                st.info("Free Cash Flow non disponibile. Utilizzando Operating Cash Flow per il calcolo DCF.")
        
        if fcf_field is None:
            ocf_fields = ['operatingCashFlow', 'netCashProvidedByOperatingActivities']
            capex_fields = ['capitalExpenditure', 'investmentsInPropertyPlantAndEquipment']
            
            ocf_field = get_balance_sheet_field(annual_cashflow, ocf_fields)
            capex_field = get_balance_sheet_field(annual_cashflow, capex_fields)
            
            if ocf_field and capex_field:
                annual_cashflow['free_cash_flow_calculated'] = annual_cashflow[ocf_field] - abs(annual_cashflow[capex_field])
                fcf_field = 'free_cash_flow_calculated'
                st.info("Free Cash Flow calcolato come: Operating Cash Flow - Capital Expenditures")
        
        if fcf_field is None:
            st.warning("Dati di Free Cash Flow e Operating Cash Flow non disponibili.")
            return None
            
        fcf = annual_cashflow[fcf_field].iloc[0]
        
        if fcf <= 0 and len(annual_cashflow) >= 3:
            fcf = annual_cashflow[fcf_field].iloc[:3].mean()
            if fcf <= 0:
                st.warning("Free Cash Flow negativo o zero, impossibile calcolare DCF.")
                return None
        elif fcf <= 0:
            st.warning("Free Cash Flow negativo o zero, impossibile calcolare DCF.")
            return None
        
        # Ottieni shares outstanding
        shares_outstanding = (info.get('sharesOutstanding') or 
                            info.get('impliedSharesOutstanding') or
                            info.get('floatShares'))
        
        if not shares_outstanding and additional_metrics:
            shares_outstanding = additional_metrics.get('numberOfShares')
        
        if not shares_outstanding and not annual_financials.empty:
            weighted_avg_field_options = [
                'weightedAverageShsOutDil',
                'weightedAverageShsOut'
            ]
            
            for field in weighted_avg_field_options:
                if field in annual_financials.columns:
                    shares_outstanding = annual_financials[field].iloc[0]
                    if shares_outstanding and shares_outstanding > 0:
                        break
        
        if not shares_outstanding:
            st.warning("Numero di azioni in circolazione non disponibile.")
            return None
        
        projected_cash_flows = []
        
        for year in range(1, forecast_period + 1):
            if forecast_period > 1:
                weight = (forecast_period - year) / (forecast_period - 1)
                growth_rate = weight * growth_rate_initial + (1 - weight) * growth_rate_terminal
            else:
                growth_rate = growth_rate_terminal
                
            projected_cf = fcf * (1 + growth_rate) ** year
            present_value = projected_cf / (1 + discount_rate) ** year
            projected_cash_flows.append(present_value)
        
        terminal_value = (fcf * (1 + growth_rate_terminal) ** forecast_period * (1 + growth_rate_terminal)) / (discount_rate - growth_rate_terminal)
        present_terminal_value = terminal_value / (1 + discount_rate) ** forecast_period
        
        enterprise_value = sum(projected_cash_flows) + present_terminal_value
        
        debt_field_options = [
            'totalDebt',
            'netDebt',
            'longTermDebt'
        ]
        
        cash_field_options = [
            'cashAndCashEquivalents',
            'cashAndShortTermInvestments'
        ]
        
        balance_sheet = annual_balance_sheet.iloc[0] if not annual_balance_sheet.empty else None
        
        if balance_sheet is not None:
            debt_field = get_balance_sheet_field(annual_balance_sheet, debt_field_options)
            cash_field = get_balance_sheet_field(annual_balance_sheet, cash_field_options)
            
            total_debt = balance_sheet[debt_field] if debt_field else 0
            total_cash = balance_sheet[cash_field] if cash_field else 0
            
            equity_value = enterprise_value - total_debt + total_cash
            intrinsic_value_per_share = equity_value / shares_outstanding
            
            current_price = info.get('currentPrice', info.get('price', 0))
            
            st.metric(
                label="Valore Intrinseco per Azione (DCF)",
                value=f"{currency_symbol}{intrinsic_value_per_share:.2f}",
                delta=f"{(intrinsic_value_per_share / current_price - 1) * 100:.1f}% vs prezzo attuale" if current_price > 0 else "N/A"
            )
            
            st.info(f"""
            **Parametri utilizzati:**
            - Free Cash Flow: {currency_symbol}{fcf/1000000:.2f}M
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
        st.error(traceback.format_exc())
        return None

def calculate_graham_value(info, annual_financials, currency_symbol, earnings_growth=None):
    """Calcolo valore intrinseco con Formula di Graham"""
    try:
        eps = (info.get('trailingEPS') or 
               info.get('ttmEPS') or
               additional_metrics.get('revenuePerShare'))
        
        if not eps or eps <= 0:
            if not annual_financials.empty and 'eps' in annual_financials.columns:
                eps = annual_financials['eps'].iloc[0]
            elif not annual_financials.empty and 'epsdiluted' in annual_financials.columns:
                eps = annual_financials['epsdiluted'].iloc[0]
        
        if not eps or eps <= 0:
            st.warning("EPS non disponibile dai dati. Inserisci un valore manualmente.")
            eps = st.number_input("Inserisci EPS manualmente:", value=1.0, step=0.1)
        
        if earnings_growth is None:
            if not annual_financials.empty and len(annual_financials) > 1:
                eps_column = 'epsdiluted' if 'epsdiluted' in annual_financials.columns else 'eps'
                
                if eps_column in annual_financials.columns:
                    sorted_data = annual_financials.sort_index(ascending=False)
                    
                    latest_eps = sorted_data[eps_column].iloc[0]
                    oldest_eps = sorted_data[eps_column].iloc[-1]
                    
                    years = (sorted_data.index[0] - sorted_data.index[-1]).days / 365.25
                    
                    if years > 0 and latest_eps > 0 and oldest_eps > 0:
                        earnings_growth = ((latest_eps / oldest_eps) ** (1/years) - 1) * 100
            
            if earnings_growth is None:
                earnings_growth = 10.0
                st.warning("Tasso di crescita degli utili non disponibile. Utilizzando 10.0% come valore predefinito.")
        
        growth_rate_default = min(max(float(earnings_growth), 0.0), 30.0)
        
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
        
        intrinsic_value = eps * (8.5 + 2 * (growth_rate / 100)) * (4.4 / bond_yield)
        
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        
        st.metric(
            label="Valore Intrinseco per Azione (Graham)",
            value=f"{currency_symbol}{intrinsic_value:.2f}",
            delta=f"{(intrinsic_value / current_price - 1) * 100:.1f}% vs prezzo attuale" if current_price > 0 else "N/A"
        )
        
        st.info(f"""
        **Formula di Graham utilizzata:** V = EPS × (8.5 + 2g) × 4.4 / Y
        
        **Parametri:**
        - EPS: {eps:.2f}
        - Tasso di crescita (g): {growth_rate:.1f}%
        - Rendimento Bond AAA (Y): {bond_yield:.1f}%
        
        **Calcolo:** {currency_symbol}{eps:.2f} × ({8.5 + (2 * growth_rate / 100):.2f}) × ({4.4 / bond_yield:.2f}) = {currency_symbol}{intrinsic_value:.2f}
        """)
        
        return intrinsic_value
        
    except Exception as e:
        st.error(f"Errore nel calcolo con la Formula di Graham: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# SCORE DIRAMCO # ====================================================================
st.write("")  # Spazio di separazione

# Ricerca score nel Google Sheet
ticker_score = search_ticker_score(symbol)

if ticker_score is not None:
    # Determina il colore in base allo score
    if ticker_score >= 9:
        score_color = "#00b300"  # Verde molto scuro
        score_emoji = "🟢🟢"
        score_label = "Eccellente"
    elif ticker_score >= 8:
        score_color = "#00e600"  # Verde
        score_emoji = "🟢"
        score_label = "Molto Buono"
    elif ticker_score >= 6:
        score_color = "#ffd700"  # Giallo
        score_emoji = "🟡"
        score_label = "Accettabile"
    else:
        score_color = "#ff4444"  # Rosso
        score_emoji = "🔴"
        score_label = "Cautela"
    
    st.markdown(f"""
    <div style='padding: 20px; border-radius: 10px; background-color: {score_color}20; border: 2px solid {score_color}; margin-top: 20px; margin-bottom: 20px;'>
        <h2 style='margin: 0; color: {score_color}; text-align: center;'>{score_emoji} Score DIRAMCO: {ticker_score:.1f}/10</h2>
        <p style='margin: 10px 0 0 0; font-size: 1.1em; text-align: center;'>Valutazione qualità Analisi Fondamentale. Indica indirettamente il MOAT del Titolo - {score_label}</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("ℹ️ **Score DIRAMCO:** Non disponibile al momento")

# === SEZIONE ANALISI NEWS ===
if PERPLEXITY_API_KEY and PERPLEXITY_API_KEY != "YOUR_PERPLEXITY_API_KEY_HERE":
    st.header('Analisi delle News e degli ultimi Risultati Finanziari tramite IA')

    with st.expander("Clicca per vedere gli ultimi risultati", expanded=False):
        with st.spinner("Analizzando le notizie di mercato con IA..."):
            ai_analysis = analyze_stock_with_perplexity(symbol, company_name, PERPLEXITY_API_KEY)
            st.markdown(ai_analysis)
            
        st.info("Analisi generata tramite IA basata sulle informazioni di mercato più recenti.")
else:
    st.info("Configura la chiave API Perplexity nel codice per abilitare l'analisi delle notizie con IA.")
    
# === SEZIONE DATI FINANZIARI ===
st.header('Dati Finanziari Storici')

if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_HERE":
    st.warning("Configura la chiave API FMP nel codice per visualizzare i dati finanziari dettagliati.")
else:
    col_period, col_spacer = st.columns([1, 3])
    with col_period:
        selection = st.segmented_control(
            label='Periodo di Analisi', 
            options=['Trimestrale', 'Annuale'], 
            default='Annuale'
        )

    if selection == 'Trimestrale':
        period = 'quarterly'
    else:
        period = 'annual'

    with st.spinner("Caricamento dati finanziari..."):
        income_statements = fetch_income_statements_fmp(symbol, FMP_API_KEY, period=period, limit=60)
        balance_sheets = fetch_balance_sheets_fmp(symbol, FMP_API_KEY, period=period, limit=60)
        cashflow_statements = fetch_cashflow_statements_fmp(symbol, FMP_API_KEY, period=period, limit=60)

    if income_statements.empty or balance_sheets.empty or cashflow_statements.empty:
        st.warning("⚠️ Alcuni dati finanziari non sono disponibili. I grafici potrebbero essere incompleti.")

    # === CALCOLO SHARES OUTSTANDING (PRIMA DI TUTTO) ===
    shares_outstanding = (information.get("sharesOutstanding") or 
                         information.get("impliedSharesOutstanding") or
                         information.get("floatShares") or
                         additional_metrics.get("numberOfShares"))
    
    # Se non trovato, cerca nei financial statements
    if not shares_outstanding and not income_statements.empty:
        shares_field_options = ['weightedAverageShsOutDil', 'weightedAverageShsOut']
        shares_field = get_balance_sheet_field(income_statements, shares_field_options)
        if shares_field:
            sorted_income = income_statements.sort_index(ascending=False)
            shares_outstanding = sorted_income[shares_field].iloc[0]
    
    if shares_outstanding:
        st.info(f"✓ Azioni in circolazione: {shares_outstanding:,.0f}")
    else:
        st.warning("⚠️ Numero di azioni in circolazione non disponibile - alcuni grafici potrebbero non essere mostrati")

    # === GRAFICI CONTO ECONOMICO ===
    if not income_statements.empty:
        st.subheader("Analisi Conto Economico")
        
        income_reset = income_statements.reset_index()
        if 'Date' in income_reset.columns:
            if period == 'annual':
                income_reset['Anno'] = income_reset['Date'].dt.strftime('%Y')
            else:
                income_reset['Quarter'] = income_reset['Date'].apply(lambda x: f"{x.year}-Q{(x.month-1)//3 + 1}")
        
        income_reset = income_reset.sort_values(by='Date', ascending=False).head(60)
        income_reset = income_reset.sort_values(by='Date')
        
        import plotly.express as px
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            if 'revenue' in income_reset.columns:
                fig_revenue = px.bar(
                    income_reset, 
                    x='Anno' if period == 'annual' else 'Quarter', 
                    y='revenue',
                    title="Ricavi Totali",
                    labels={"revenue": f"Ricavi ({currency_symbol})", 'Anno' if period == 'annual' else 'Quarter': "Periodo"},
                    color_discrete_sequence=['#1f77b4']
                )
                fig_revenue.update_layout(height=350)
                st.plotly_chart(fig_revenue, use_container_width=True)
        
        with col_chart2:
            eps_field_options = ['epsdiluted', 'eps']
            eps_field = get_balance_sheet_field(income_statements, eps_field_options)
            if eps_field:
                fig_eps = px.bar(
                    income_reset, 
                    x='Anno' if period == 'annual' else 'Quarter', 
                    y=eps_field,
                    title="Utile per Azione (EPS)",
                    labels={eps_field: f"EPS ({currency_symbol})", 'Anno' if period == 'annual' else 'Quarter': "Periodo"},
                    color_discrete_sequence=['#ff7f0e']
                )
                fig_eps.update_layout(height=350)
                st.plotly_chart(fig_eps, use_container_width=True)
        
        col_chart3, col_chart4 = st.columns(2)
        
        with col_chart3:
            shares_field_options = ['weightedAverageShsOutDil', 'weightedAverageShsOut']
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
            # Recupera e visualizza i dividendi aggiustati per split
            dividend_history = fetch_dividend_history_fmp(symbol, FMP_API_KEY)
            
            if not dividend_history.empty:
                dividend_field = 'adjDividend' if 'adjDividend' in dividend_history.columns else 'dividend'
                dividend_history['Year'] = dividend_history['date'].dt.year
                
                if period == 'annual':
                    dividend_agg = dividend_history.groupby('Year')[dividend_field].sum().reset_index()
                    dividend_agg = dividend_agg.sort_values('Year')
                    dividend_agg['Anno'] = dividend_agg['Year'].astype(str)
                    
                    fig_dividends = px.bar(
                        dividend_agg, 
                        x='Anno', 
                        y=dividend_field,
                        title="Dividendi Annuali per Azione",
                        labels={dividend_field: f"Dividendo ({currency_symbol})", 'Anno': "Anno"},
                        color_discrete_sequence=['#e377c2']
                    )
                else:
                    dividend_history['Quarter'] = dividend_history['date'].apply(lambda x: f"{x.year}-Q{(x.month-1)//3 + 1}")
                    dividend_history = dividend_history.sort_values('date')
                    dividend_history = dividend_history.tail(60)
                    
                    fig_dividends = px.bar(
                        dividend_history, 
                        x='Quarter', 
                        y=dividend_field,
                        title="Dividendi Trimestrali per Azione",
                        labels={dividend_field: f"Dividendo ({currency_symbol})", 'Quarter': "Trimestre"},
                        color_discrete_sequence=['#e377c2']
                    )
                
                fig_dividends.update_layout(height=350)
                st.plotly_chart(fig_dividends, use_container_width=True)
                
                if dividend_field == 'adjDividend':
                    st.caption("✓ Dividendi aggiustati per split azionari")
            else:
                # Fallback: calcola dividendi dal cash flow statement
                dividends_field_options = ['dividendsPaid', 'paymentOfDividends']
                dividends_field = get_balance_sheet_field(cashflow_statements, dividends_field_options)
                
                if dividends_field:
                    shares_field = get_balance_sheet_field(income_statements, ['weightedAverageShsOutDil', 'weightedAverageShsOut'])
                    
                    if shares_field and not income_reset.empty:
                        cashflow_temp = cashflow_statements.reset_index()
                        if 'Date' in cashflow_temp.columns:
                            if period == 'annual':
                                cashflow_temp['Anno'] = cashflow_temp['Date'].dt.strftime('%Y')
                            else:
                                cashflow_temp['Quarter'] = cashflow_temp['Date'].apply(lambda x: f"{x.year}-Q{(x.month-1)//3 + 1}")
                        
                        cashflow_temp = cashflow_temp.sort_values(by='Date', ascending=False).head(60)
                        cashflow_temp = cashflow_temp.sort_values(by='Date')
                        
                        merged_data = cashflow_temp.merge(
                            income_reset[['Date', shares_field]], 
                            on='Date', 
                            how='inner'
                        )
                        
                        if not merged_data.empty:
                            merged_data['Dividendo per Azione'] = abs(merged_data[dividends_field]) / merged_data[shares_field]
                            
                            if period == 'annual':
                                x_col = 'Anno'
                            else:
                                x_col = 'Quarter'
                            
                            fig_dividends = px.bar(
                                merged_data, 
                                x=x_col, 
                                y='Dividendo per Azione',
                                title="Dividendi per Azione",
                                labels={"Dividendo per Azione": f"Dividendo ({currency_symbol})", x_col: "Periodo"},
                                color_discrete_sequence=['#e377c2']
                            )
                            fig_dividends.update_layout(height=350)
                            st.plotly_chart(fig_dividends, use_container_width=True)
                            st.caption("Calcolato da: Dividendi Totali / Azioni in Circolazione")
                        else:
                            st.info("ℹ️ Dati dividendi non disponibili")
                    else:
                        st.info("ℹ️ Dati dividendi o azioni in circolazione non disponibili")
                else:
                    st.info("ℹ️ Dati dividendi non disponibili")
        
        col_chart5, col_chart6 = st.columns(2)
        
        with col_chart5:
            # Calcola Payout Ratio usando dividendi aggiustati
            dividend_history = fetch_dividend_history_fmp(symbol, FMP_API_KEY)
            
            if eps_field and not dividend_history.empty:
                dividend_field = 'adjDividend' if 'adjDividend' in dividend_history.columns else 'dividend'
                dividend_history['Year'] = dividend_history['date'].dt.year
                dividend_agg = dividend_history.groupby('Year')[dividend_field].sum().reset_index()
                
                income_with_year = income_reset.copy()
                income_with_year['Year'] = income_with_year['Date'].dt.year
                
                payout_data = income_with_year.merge(dividend_agg, on='Year', how='inner')
                
                if not payout_data.empty and eps_field in payout_data.columns:
                    payout_data['Payout Ratio'] = (payout_data[dividend_field] / payout_data[eps_field]) * 100
                    payout_data = payout_data[(payout_data['Payout Ratio'] >= 0) & (payout_data['Payout Ratio'] <= 200)]
                    
                    if not payout_data.empty:
                        payout_data = payout_data.sort_values('Year')
                        
                        if period == 'annual':
                            payout_data['Anno'] = payout_data['Year'].astype(str)
                            x_col = 'Anno'
                        else:
                            payout_data['Quarter'] = payout_data['Date'].apply(lambda x: f"{x.year}-Q{(x.month-1)//3 + 1}")
                            x_col = 'Quarter'
                        
                        fig_payout = px.bar(
                            payout_data, 
                            x=x_col, 
                            y='Payout Ratio',
                            title="Payout Ratio (%) - Dividendi/EPS",
                            labels={"Payout Ratio": "Payout Ratio (%)", x_col: "Periodo"},
                            color_discrete_sequence=['#bcbd22']
                        )
                        fig_payout.add_hline(y=50, line_dash="dash", line_color="orange", 
                                           annotation_text="Sostenibile (50%)")
                        fig_payout.add_hline(y=100, line_dash="dash", line_color="red", 
                                           annotation_text="Critico (100%)")
                        fig_payout.update_layout(height=350)
                        st.plotly_chart(fig_payout, use_container_width=True)
                        
                        avg_payout = payout_data['Payout Ratio'].mean()
                        st.caption(f"Payout Ratio medio: {avg_payout:.1f}% | {'✓ Sostenibile' if avg_payout < 70 else '⚠️ Elevato'}")
                    else:
                        st.info("ℹ️ Impossibile calcolare Payout Ratio")
                else:
                    st.info("ℹ️ Dati insufficienti per calcolare Payout Ratio")
            else:
                st.info("ℹ️ Dati EPS o dividendi non disponibili per Payout Ratio")

        with col_chart6:
            # Calcola Margine Lordo
            if 'revenue' in income_reset.columns and 'costOfRevenue' in income_reset.columns:
                income_reset['Margine Lordo'] = ((income_reset['revenue'] - income_reset['costOfRevenue']) / income_reset['revenue']) * 100
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
            elif 'grossProfit' in income_reset.columns and 'revenue' in income_reset.columns:
                income_reset['Margine Lordo'] = (income_reset['grossProfit'] / income_reset['revenue']) * 100
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
                balance_reset['Quarter'] = balance_reset['Date'].apply(lambda x: f"{x.year}-Q{(x.month-1)//3 + 1}")
        
        balance_reset = balance_reset.sort_values(by='Date', ascending=False).head(60)
        balance_reset = balance_reset.sort_values(by='Date')
        
        col_balance1, col_balance2 = st.columns(2)
        
        with col_balance1:
            if shares_outstanding:
                equity_field_options = [
                    'totalStockholdersEquity',
                    'totalEquity',
                    'shareholdersEquity',
                    'stockholdersEquity'
                ]
                equity_field = get_balance_sheet_field(balance_sheets, equity_field_options)
                
                if equity_field and equity_field in balance_reset.columns:
                    balance_reset['Book Value per Share'] = balance_reset[equity_field] / shares_outstanding
                    
                    fig_bvps = px.bar(
                        balance_reset, 
                        x='Anno' if period == 'annual' else 'Quarter', 
                        y='Book Value per Share',
                        title="Book Value per Share",
                        labels={"Book Value per Share": f"Book Value per Share ({currency_symbol})", 
                               'Anno' if period == 'annual' else 'Quarter': "Periodo"},
                        color_discrete_sequence=['#17becf']
                    )
                    fig_bvps.update_layout(height=350)
                    st.plotly_chart(fig_bvps, use_container_width=True)
                else:
                    st.info("ℹ️ Dati Book Value non disponibili")
            else:
                st.info("ℹ️ Numero di azioni in circolazione non disponibile")
        
        with col_balance2:
            debt_fields = ['totalDebt', 'netDebt', 'longTermDebt']
            equity_fields = ['totalStockholdersEquity', 'totalEquity', 'stockholdersEquity']
            
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
                            'Anno' if period == 'annual' else 'Quarter': date.strftime('%Y') if period == 'annual' else f"{date.year}-Q{(date.month-1)//3 + 1}"
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
                cashflow_reset['Quarter'] = cashflow_reset['Date'].apply(lambda x: f"{x.year}-Q{(x.month-1)//3 + 1}")
        
        cashflow_reset = cashflow_reset.sort_values(by='Date', ascending=False).head(60)
        cashflow_reset = cashflow_reset.sort_values(by='Date')
        
        col_cf1, col_cf2 = st.columns(2)
        
        with col_cf1:
            ocf_field_options = ['operatingCashFlow', 'netCashProvidedByOperatingActivities']
            ocf_field = get_balance_sheet_field(cashflow_statements, ocf_field_options)
            if ocf_field:
                fig_ocf = px.bar(
                    cashflow_reset, 
                    x='Anno' if period == 'annual' else 'Quarter', 
                    y=ocf_field,
                    title="Cash Flow Operativo",
                    labels={ocf_field: f"Cash Flow Operativo ({currency_symbol})", 'Anno' if period == 'annual' else 'Quarter': "Periodo"},
                    color_discrete_sequence=['#2ca02c']
                )
                fig_ocf.update_layout(height=350)
                st.plotly_chart(fig_ocf, use_container_width=True)
        
        with col_cf2:
            icf_field_options = ['netCashUsedForInvestingActivites', 'investingCashFlow']
            icf_field = get_balance_sheet_field(cashflow_statements, icf_field_options)
            if icf_field:
                fig_icf = px.bar(
                    cashflow_reset, 
                    x='Anno' if period == 'annual' else 'Quarter', 
                    y=icf_field,
                    title="Cash Flow da Investimenti",
                    labels={icf_field: f"Cash Flow da Investimenti ({currency_symbol})", 'Anno' if period == 'annual' else 'Quarter': "Periodo"},
                    color_discrete_sequence=['#d62728']
                )
                fig_icf.update_layout(height=350)
                st.plotly_chart(fig_icf, use_container_width=True)
        
        col_cf3, col_cf4 = st.columns(2)
        
        with col_cf3:
            if shares_outstanding:
                fcf_field_options = ['freeCashFlow']
                fcf_field = get_balance_sheet_field(cashflow_statements, fcf_field_options)
                
                if not fcf_field:
                    ocf_field_options = ['operatingCashFlow', 'netCashProvidedByOperatingActivities']
                    fcf_field = get_balance_sheet_field(cashflow_statements, ocf_field_options)
                
                if not fcf_field:
                    ocf_fields = ['operatingCashFlow', 'netCashProvidedByOperatingActivities']
                    capex_fields = ['capitalExpenditure', 'investmentsInPropertyPlantAndEquipment']
                    
                    ocf_field = get_balance_sheet_field(cashflow_statements, ocf_fields)
                    capex_field = get_balance_sheet_field(cashflow_statements, capex_fields)
                    
                    if ocf_field and capex_field and ocf_field in cashflow_reset.columns and capex_field in cashflow_reset.columns:
                        cashflow_reset['Free Cash Flow Calcolato'] = cashflow_reset[ocf_field] - abs(cashflow_reset[capex_field])
                        fcf_field = 'Free Cash Flow Calcolato'
                
                if fcf_field and fcf_field in cashflow_reset.columns:
                    cashflow_reset['Cash Flow per Share'] = cashflow_reset[fcf_field] / shares_outstanding
                    
                    title = "Cash Flow per Share"
                    if 'free' in fcf_field.lower():
                        subtitle = "(Free Cash Flow)"
                    elif 'calcolato' in fcf_field.lower():
                        subtitle = "(FCF Calcolato)"
                    else:
                        subtitle = "(Operating CF)"
                    
                    fig_cfps = px.bar(
                        cashflow_reset, 
                        x='Anno' if period == 'annual' else 'Quarter', 
                        y='Cash Flow per Share',
                        title=f"{title}<br><sub>{subtitle}</sub>",
                        labels={"Cash Flow per Share": f"Cash Flow per Share ({currency_symbol})", 
                               'Anno' if period == 'annual' else 'Quarter': "Periodo"},
                        color_discrete_sequence=['#9467bd']
                    )
                    fig_cfps.update_layout(height=350)
                    st.plotly_chart(fig_cfps, use_container_width=True)
                else:
                    st.info("ℹ️ Dati Cash Flow non disponibili")
            else:
                st.info("ℹ️ Numero di azioni in circolazione non disponibile")
        
        with col_cf4:
            fcf_financing_field_options = ['netCashUsedProvidedByFinancingActivities', 'financingCashFlow']
            fcf_financing_field = get_balance_sheet_field(cashflow_statements, fcf_financing_field_options)
            if fcf_financing_field:
                fig_fcf_financing = px.bar(
                    cashflow_reset, 
                    x='Anno' if period == 'annual' else 'Quarter', 
                    y=fcf_financing_field,
                    title="Cash Flow da Finanziamenti",
                    labels={fcf_financing_field: f"Cash Flow da Finanziamenti ({currency_symbol})", 'Anno' if period == 'annual' else 'Quarter': "Periodo"},
                    color_discrete_sequence=['#ff7f0e']
                )
                fig_fcf_financing.update_layout(height=350)
                st.plotly_chart(fig_fcf_financing, use_container_width=True)

# === GRAFICI MULTIPLI DI VALUTAZIONE ===
    if not income_statements.empty and not balance_sheets.empty:
        st.subheader("Indicatori di Prezzo Storici")
        
        try:
            stock = yf.Ticker(symbol)
            
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=10*365)
            
            hist_10y = stock.history(start=start_date, end=end_date, interval='1mo')
            
            if hist_10y.empty or len(hist_10y) < 12:
                st.info("Pochi dati mensili disponibili, utilizzo dati settimanali...")
                hist_10y = stock.history(start=start_date, end=end_date, interval='1wk')
            
            if hist_10y.empty or len(hist_10y) < 50:
                st.info("Utilizzo dati giornalieri campionati...")
                hist_daily = stock.history(start=start_date, end=end_date, interval='1d')
                if not hist_daily.empty:
                    hist_10y = hist_daily.iloc[::30, :]
            
            if not hist_10y.empty:
                st.info(f"Dati recuperati dal {hist_10y.index[0].strftime('%Y-%m-%d')} al {hist_10y.index[-1].strftime('%Y-%m-%d')} ({len(hist_10y)} punti dati)")
                
                valuation_data = []
                
                # Calcola EPS più recente
                latest_eps = None
                eps_field_options = ['epsdiluted', 'eps']
                eps_field = get_balance_sheet_field(income_statements, eps_field_options)
                if eps_field:
                    sorted_income = income_statements.sort_index(ascending=False)
                    latest_eps = sorted_income[eps_field].iloc[0]
                
                if not latest_eps:
                    latest_eps = information.get('trailingEPS', information.get('ttmEPS'))
                
                # Calcola Book Value più recente
                latest_book_value = None
                if shares_outstanding:
                    equity_field_options = ['totalStockholdersEquity', 'totalEquity', 'stockholdersEquity']
                    equity_field = get_balance_sheet_field(balance_sheets, equity_field_options)
                    
                    if equity_field:
                        sorted_balance = balance_sheets.sort_index(ascending=False)
                        latest_equity = sorted_balance[equity_field].iloc[0]
                        if latest_equity and latest_equity > 0:
                            latest_book_value = latest_equity / shares_outstanding

                # Fallback a Yahoo Finance
                if not latest_book_value:
                    book_value_yahoo = information.get('bookValue')
                    if book_value_yahoo and book_value_yahoo > 0:
                        latest_book_value = book_value_yahoo

                # FIX: Calcolo alternativo se ancora non disponibile
                if not latest_book_value:
                    pb_ratio = information.get('priceToBook')
                    current_price = information.get('currentPrice', information.get('regularMarketPrice'))
                    if pb_ratio and pb_ratio > 0 and current_price and current_price > 0:
                        latest_book_value = current_price / pb_ratio
                        st.caption(f" Book Value calcolato da P/B Ratio: {currency_symbol}{latest_book_value:.2f}")
                
                for date, row in hist_10y.iterrows():
                    close_price = row['Close']
                    
                    pe_ratio = None
                    if latest_eps and latest_eps > 0:
                        pe_ratio = close_price / latest_eps
                        if pe_ratio < 0 or pe_ratio > 200:
                            pe_ratio = None
                    
                    pbv_ratio = None
                    if latest_book_value and latest_book_value > 0:
                        pbv_ratio = close_price / latest_book_value
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
                    valuation_df = valuation_df.sort_values('Date')
                    
                    col_pe, col_pbv = st.columns(2)
                    
                    with col_pe:
                        pe_data = valuation_df.dropna(subset=['PE Ratio'])
                        if not pe_data.empty and len(pe_data) > 1:
                            fig_pe = px.line(
                                pe_data,
                                x='Date',
                                y='PE Ratio',
                                title=f"PE Ratio - {pe_data['Date'].dt.year.min()} a {pe_data['Date'].dt.year.max()}",
                                labels={"PE Ratio": "PE Ratio", "Date": "Data"}
                            )
                            
                            avg_pe = pe_data['PE Ratio'].mean()
                            fig_pe.add_hline(y=avg_pe, line_dash="dash", line_color="orange", 
                                        annotation_text=f"Media: {avg_pe:.1f}")
                            
                            current_pe = information.get('trailingPE')
                            if current_pe:
                                fig_pe.add_hline(y=current_pe, line_dash="dot", line_color="red", 
                                            annotation_text=f"PE Attuale: {current_pe:.1f}")
                            
                            fig_pe.update_layout(
                                height=400,
                                xaxis=dict(
                                    title="Data",
                                    tickformat="%Y-%m",
                                    dtick="M12"
                                )
                            )
                            fig_pe.update_traces(line_color='#1f77b4')
                            st.plotly_chart(fig_pe, use_container_width=True)
                            
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
                        pbv_data = valuation_df.dropna(subset=['PBV Ratio'])
                        if not pbv_data.empty and len(pbv_data) > 1:
                            fig_pbv = px.line(
                                pbv_data,
                                x='Date',
                                y='PBV Ratio',
                                title=f"PBV Ratio - {pbv_data['Date'].dt.year.min()} a {pbv_data['Date'].dt.year.max()}",
                                labels={"PBV Ratio": "Price to Book Value", "Date": "Data"}
                            )
                            
                            fig_pbv.add_hline(y=1, line_dash="dash", line_color="red", 
                                            annotation_text="PBV = 1")
                            
                            avg_pbv = pbv_data['PBV Ratio'].mean()
                            fig_pbv.add_hline(y=avg_pbv, line_dash="dash", line_color="orange", 
                                            annotation_text=f"Media: {avg_pbv:.1f}")
                            
                            # FIX: Calcola PBV Attuale con più fonti - PRIORITÀ MULTIPLE
                            current_pbv = None
                            
                            # 1. Prova da information (Yahoo Finance / FMP principale)
                            current_pbv = information.get('priceToBook')
                            
                            # 2. Prova da additional_metrics (FMP key metrics)
                            if not current_pbv or current_pbv == 0:
                                current_pbv = additional_metrics.get('pbRatio')
                            
                            # 3. Usa l'ultimo valore dal grafico storico come approssimazione
                            if not current_pbv or current_pbv == 0:
                                if not pbv_data.empty:
                                    current_pbv = pbv_data['PBV Ratio'].iloc[-1]
                                    st.caption("ℹ️ PBV calcolato dall'ultimo valore storico disponibile")
                            
                            # 4. Calcola manualmente da prezzo e book value
                            if not current_pbv or current_pbv == 0:
                                current_price_for_pbv = information.get('currentPrice', information.get('regularMarketPrice'))
                                if current_price_for_pbv and current_price_for_pbv > 0:
                                    # Cerca book value per azione
                                    book_value = information.get('bookValue')
                                    
                                    # Se non disponibile, calcola da balance sheet
                                    if not book_value and 'balance_sheets' in globals() and not balance_sheets.empty:
                                        equity_field_options = ['totalStockholdersEquity', 'totalEquity', 'stockholdersEquity']
                                        equity_field = get_balance_sheet_field(balance_sheets, equity_field_options)
                                        
                                        if equity_field and shares_outstanding and shares_outstanding > 0:
                                            latest_balance = balance_sheets.iloc[0]
                                            total_equity = latest_balance[equity_field]
                                            if total_equity and total_equity > 0:
                                                book_value = total_equity / shares_outstanding
                                    
                                    # Calcola PBV
                                    if book_value and book_value > 0:
                                        current_pbv = current_price_for_pbv / book_value
                                        st.caption(f"ℹ️ PBV calcolato: Prezzo ({currency_symbol}{current_price_for_pbv:.2f}) / Book Value ({currency_symbol}{book_value:.2f})")
                            
                            # Aggiungi linea al grafico se PBV disponibile
                            if current_pbv and current_pbv > 0:
                                fig_pbv.add_hline(y=current_pbv, line_dash="dot", line_color="red", 
                                                annotation_text=f"PBV Attuale: {current_pbv:.1f}")
                            
                            fig_pbv.update_layout(
                                height=400,
                                xaxis=dict(
                                    title="Data",
                                    tickformat="%Y-%m",
                                    dtick="M12"
                                )
                            )
                            fig_pbv.update_traces(line_color='#ff7f0e')
                            st.plotly_chart(fig_pbv, use_container_width=True)
                            
                            # Statistiche PBV
                            with st.expander("Statistiche PBV Ratio"):
                                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                with col_stat1:
                                    if current_pbv and current_pbv > 0:
                                        st.metric("PBV Attuale", f"{current_pbv:.2f}")
                                    else:
                                        st.metric("PBV Attuale", "N/A")
                                with col_stat2:
                                    st.metric("PBV Medio", f"{avg_pbv:.2f}")
                                with col_stat3:
                                    st.metric("PBV Min", f"{pbv_data['PBV Ratio'].min():.2f}")
                                with col_stat4:
                                    st.metric("PBV Max", f"{pbv_data['PBV Ratio'].max():.2f}")
                        else:
                            st.info("ℹ️ Dati PBV Ratio insufficienti per il grafico storico")

                    # SEZIONE ANALISI - Anche questa va corretta
                    col_analysis1, col_analysis2 = st.columns(2)

                    with col_analysis1:
                        if not pe_data.empty and current_pe:
                            pe_percentile = (pe_data['PE Ratio'] < current_pe).mean() * 100
                            if pe_percentile <= 25:
                                pe_status = "🟢 Sottovalutato"
                            elif pe_percentile >= 75:
                                pe_status = "🔴 Sopravvalutato"
                            else:
                                pe_status = "🟡 Neutrale"
                            
                            st.markdown(f"""
                            **PE Ratio Analysis:**
                            - Stato: {pe_status}
                            - Percentile storico: **{pe_percentile:.1f}%**
                            - Il PE attuale è superiore al {pe_percentile:.1f}% dei valori storici
                            """)

                    with col_analysis2:
                        if not pbv_data.empty:
                            # FIX: Usa lo stesso current_pbv calcolato sopra
                            if current_pbv and current_pbv > 0:
                                pbv_percentile = (pbv_data['PBV Ratio'] < current_pbv).mean() * 100
                                if pbv_percentile <= 25:
                                    pbv_status = "🟢 Sottovalutato"
                                elif pbv_percentile >= 75:
                                    pbv_status = "🔴 Sopravvalutato"
                                else:
                                    pbv_status = "🟡 Neutrale"
                                
                                st.markdown(f"""
                                **PBV Ratio Analysis:**
                                - Stato: {pbv_status}
                                - Percentile storico: **{pbv_percentile:.1f}%**
                                - Il PBV attuale è superiore al {pbv_percentile:.1f}% dei valori storici
                                """)
                            else:
                                # FIX: Mostra statistiche anche senza current_pbv
                                avg_pbv = pbv_data['PBV Ratio'].mean()
                                st.markdown(f"""
                                **PBV Ratio Analysis:**
                                - PBV Attuale: Non disponibile
                                - PBV Medio storico: **{avg_pbv:.2f}**
                                - Range: {pbv_data['PBV Ratio'].min():.2f} - {pbv_data['PBV Ratio'].max():.2f}
                                """)
                
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
            import traceback
            st.error(traceback.format_exc())
            st.info("Impossibile calcolare i multipli di valutazione storici")

# === SEZIONE CALCOLO VALORE INTRINSECO ===
st.header('Calcolo Valore Intrinseco')

if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_HERE":
    st.warning("⚠️ Configura la chiave API FMP nel codice per calcolare il valore intrinseco.")
else:
    if 'period' in locals() and period != 'annual':
        with st.spinner(" Caricamento dati annuali per il calcolo..."):
            annual_income = fetch_income_statements_fmp(symbol, FMP_API_KEY, period='annual', limit=60)
            annual_balance = fetch_balance_sheets_fmp(symbol, FMP_API_KEY, period='annual', limit=60)
            annual_cashflow = fetch_cashflow_statements_fmp(symbol, FMP_API_KEY, period='annual', limit=60)
    elif 'income_statements' in locals():
        annual_income = income_statements
        annual_balance = balance_sheets
        annual_cashflow = cashflow_statements
    else:
        with st.spinner(" Caricamento dati finanziari..."):
            annual_income = fetch_income_statements_fmp(symbol, FMP_API_KEY, period='annual', limit=60)
            annual_balance = fetch_balance_sheets_fmp(symbol, FMP_API_KEY, period='annual', limit=60)
            annual_cashflow = fetch_cashflow_statements_fmp(symbol, FMP_API_KEY, period='annual', limit=60)

    if annual_income.empty or annual_balance.empty or annual_cashflow.empty:
        st.warning("⚠️ Dati insufficienti per il calcolo del valore intrinseco.")
    else:
        earnings_growth = None
        if not annual_income.empty and len(annual_income) > 1:
            eps_column_options = ['epsdiluted', 'eps']
            eps_column = get_balance_sheet_field(annual_income, eps_column_options)
            
            if eps_column:
                sorted_data = annual_income.sort_index(ascending=False)
                
                if len(sorted_data) >= 2:
                    latest_eps = sorted_data[eps_column].iloc[0]
                    oldest_eps = sorted_data[eps_column].iloc[-1]
                    
                    years = (sorted_data.index[0] - sorted_data.index[-1]).days / 365.25
                    
                    if years > 0 and latest_eps > 0 and oldest_eps > 0:
                        earnings_growth = ((latest_eps / oldest_eps) ** (1/years) - 1) * 100

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
            
            intrinsic_value_dcf = calculate_dcf_value(information, annual_income, annual_cashflow, annual_balance, currency_symbol)
            
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
            
            intrinsic_value_graham = calculate_graham_value(information, annual_income, currency_symbol, earnings_growth)

# Footer
st.markdown("---")
st.caption("Sviluppato da DIRAMCO")
