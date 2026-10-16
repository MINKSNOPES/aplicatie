
import os
import json
import math
from datetime import datetime

import pandas as pd
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from google import genai


# --------------------------------------------------
# CONFIGURARE
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="Fundamental Analysis AI",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Fundamental Analysis AI")
st.caption(
    "Analiză fundamentală a acțiunilor cu Yahoo Finance "
    "și Google Gemini"
)


# --------------------------------------------------
# UTILITARE
# --------------------------------------------------

def safe_float(value):
    """Convertește valori numerice, NaN și valori lipsă."""
    try:
        if value is None:
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def get_value(data, *keys):
    """Returnează prima valoare numerică disponibilă."""
    for key in keys:
        value = safe_float(data.get(key))
        if value is not None:
            return value
    return None


def format_number(value, decimals=2):
    if value is None:
        return "N/A"
    return f"{value:,.{decimals}f}"


def format_money(value, currency="USD"):
    if value is None:
        return "N/A"
    return f"{value:,.0f} {currency}"


def latest_balance_value(balance_sheet, candidates):
    """Extrage valoarea din cea mai recentă coloană a bilanțului."""
    if balance_sheet is None or balance_sheet.empty:
        return None

    for candidate in candidates:
        if candidate in balance_sheet.index:
            row = balance_sheet.loc[candidate]

            if isinstance(row, pd.Series):
                for value in row.tolist():
                    number = safe_float(value)
                    if number is not None:
                        return number
            else:
                number = safe_float(row)
                if number is not None:
                    return number

    return None


def dataframe_to_text(df, max_rows=35):
    """Transformă un tabel financiar într-un text pentru AI."""
    if df is None or df.empty:
        return "Date indisponibile."

    output = df.iloc[:, :4].head(max_rows).copy()
    output = output.astype(object).where(pd.notna(output), None)

    return output.to_string()


# --------------------------------------------------
# PRELUARE DATE FINANCIARE
# --------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def fetch_financial_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)

    info = ticker.info or {}

    balance_sheet = ticker.balance_sheet
    income_stmt = ticker.income_stmt
    cash_flow = ticker.cashflow

    if balance_sheet is None:
        balance_sheet = pd.DataFrame()

    if income_stmt is None:
        income_stmt = pd.DataFrame()

    if cash_flow is None:
        cash_flow = pd.DataFrame()

    if not info and balance_sheet.empty and income_stmt.empty:
        raise ValueError(
            "Nu au fost găsite date pentru acest simbol. "
            "Verifică ticker-ul și încearcă din nou."
        )

    currency = info.get("currency", "USD")

    # Multipli și date de piață
    market_data = {
        "Ticker": ticker_symbol,
        "Companie": info.get("longName")
        or info.get("shortName")
        or ticker_symbol,
        "Sector": info.get("sector", "N/A"),
        "Industrie": info.get("industry", "N/A"),
        "Monedă": currency,
        "Preț curent": get_value(info, "currentPrice", "regularMarketPrice"),
        "Capitalizare bursieră": get_value(info, "marketCap"),
        "P/E trailing": get_value(info, "trailingPE"),
        "P/E forward": get_value(info, "forwardPE"),
        "P/B": get_value(info, "priceToBook"),
        "P/S": get_value(info, "priceToSalesTrailing12Months"),
        "EV/EBITDA": get_value(info, "enterpriseToEbitda"),
        "EPS trailing": get_value(info, "trailingEps"),
        "Dividend yield": get_value(info, "dividendYield"),
        "ROE": get_value(info, "returnOnEquity"),
        "ROA": get_value(info, "returnOnAssets"),
        "Debt/Equity": get_value(info, "debtToEquity"),
        "Current ratio": get_value(info, "currentRatio"),
        "Free cash flow": get_value(info, "freeCashflow"),
        "Total revenue": get_value(info, "totalRevenue"),
        "Net income": get_value(info, "netIncomeToCommon"),
        "Profit margin": get_value(info, "profitMargins"),
        "52-week high": get_value(info, "fiftyTwoWeekHigh"),
        "52-week low": get_value(info, "fiftyTwoWeekLow"),
    }

    # Bilanț: valorile din cea mai recentă perioadă disponibilă
    balance_data = {
        "Total assets": latest_balance_value(
            balance_sheet, ["Total Assets"]
        ),
        "Total liabilities": latest_balance_value(
            balance_sheet,
            ["Total Liabilities Net Minority Interest",
             "Total Liabilities"],
        ),
        "Stockholders equity": latest_balance_value(
            balance_sheet,
            ["Stockholders Equity",
             "Total Stockholder Equity",
             "Common Stock Equity"],
        ),
        "Cash and equivalents": latest_balance_value(
            balance_sheet,
            ["Cash And Cash Equivalents",
             "Cash Cash Equivalents And Short Term Investments"],
        ),
        "Current assets": latest_balance_value(
            balance_sheet, ["Current Assets", "Total Current Assets"]
        ),
        "Current liabilities": latest_balance_value(
            balance_sheet,
            ["Current Liabilities", "Total Current Liabilities"],
        ),
        "Long-term debt": latest_balance_value(
            balance_sheet, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]
        ),
    }

    assets = balance_data["Total assets"]
    liabilities = balance_data["Total liabilities"]
    equity = balance_data["Stockholders equity"]
    current_assets = balance_data["Current assets"]
    current_liabilities = balance_data["Current liabilities"]

    # Calculează numai când există date valide
    balance_data["Liabilities / Assets"] = (
        liabilities / assets
        if liabilities is not None and assets not in (None, 0)
        else None
    )

    balance_data["Current ratio (balance sheet)"] = (
        current_assets / current_liabilities
        if current_assets is not None
        and current_liabilities not in (None, 0)
        else None
    )

    balance_data["Equity / Assets"] = (
        equity / assets
        if equity is not None and assets not in (None, 0)
        else None
    )

    return {
        "market": market_data,
        "balance": balance_data,
        "balance_sheet": balance_sheet,
        "income_stmt": income_stmt,
        "cash_flow": cash_flow,
    }


# --------------------------------------------------
# INTEGRARE GEMINI
# --------------------------------------------------

def generate_ai_analysis(financial_data, api_key, model_name):
    client = genai.Client(api_key=api_key)

    market = financial_data["market"]
    balance = financial_data["balance"]

    # Trimitem doar un subset util și serializabil
    payload = {
        "market_data": market,
        "balance_sheet_metrics": balance,
        "balance_sheet": dataframe_to_text(
            financial_data["balance_sheet"]
        ),
        "income_statement": dataframe_to_text(
            financial_data["income_stmt"]
        ),
        "cash_flow_statement": dataframe_to_text(
            financial_data["cash_flow"]
        ),
        "data_timestamp": datetime.now().isoformat(),
    }

    prompt = f"""
Ești un analist financiar fundamental, prudent și obiectiv.
Analizează compania folosind EXCLUSIV datele furnizate mai jos.

DATE FINANCIARE:
{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}

Cerințe:
1. Scrie în limba română.
2. Explică evaluarea folosind P/E trailing, P/E forward,
   P/B și ceilalți multipli disponibili.
3. Analizează bilanțul: lichiditate, datorii, active și capitaluri.
4. Evaluează profitabilitatea și fluxul de numerar, dacă există date.
5. Identifică riscurile fundamentale și ce informații lipsesc.
6. Prezintă argumente atât pentru subevaluare, cât și pentru
   supraevaluare.
7. Compară multiplii cu sectorul numai dacă există date de
   comparație reale în input. Nu inventa medii sectoriale.
8. Nu inventa cifre, rezultate financiare, prețuri-țintă sau
   evenimente. Dacă un indicator lipsește, spune acest lucru.
9. Menționează că P/E negativ, P/B negativ sau indicatorii
   lipsă necesită interpretare specială și nu indică automat
   subevaluare.
10. Nu trata datele Yahoo Finance ca fiind neapărat în timp real.
11. Încheie cu una dintre concluziile:
   - Posibil subevaluată
   - Evaluare aparent rezonabilă
   - Posibil supraevaluată
   - Date insuficiente pentru o concluzie
12. Oferă un nivel de încredere scăzut, mediu sau ridicat și
   justifică-l în funcție de completitudinea datelor.
13. Nu oferi garanții de randament și nu prezenta concluzia
   drept recomandare personalizată de cumpărare sau vânzare.

Structura raportului:
## Rezumat executiv
## Multipli și evaluare
## Analiza bilanțului
## Profitabilitate și flux de numerar
## Riscuri fundamentale
## Argumente pro și contra
## Concluzie și nivel de încredere
## Ce ar trebui verificat în continuare

Separă faptele din date de interpretările tale.
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    result = response.text

    if not result:
        raise RuntimeError(
            "Gemini nu a returnat un răspuns text."
        )

    return result


# --------------------------------------------------
# INTERFAȚĂ
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Setări")

    ticker_input = st.text_input(
        "Ticker companie",
        value="AAPL",
        max_chars=20,
        help="Exemple: AAPL, MSFT, GOOGL, BMW.DE",
    ).strip().upper()

    default_api_key = os.getenv("GEMINI_API_KEY", "")

    try:
        secret_api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_api_key = ""

    api_key = st.text_input(
        "Gemini API Key",
        value=secret_api_key or default_api_key,
        type="password",
        help="Cheia rămâne în sesiunea aplicației.",
    )

    model_name = st.text_input(
        "Model Gemini",
        value="gemini-2.5-flash",
        help=(
            "Dacă modelul nu este disponibil în contul tău, "
            "introdu un ID de model suportat."
        ),
    ).strip()

    run_analysis = st.button(
        "🔍 Analizează compania",
        type="primary",
        use_container_width=True,
    )

    st.caption(
        "Datele financiare sunt furnizate de Yahoo Finance. "
        "Disponibilitatea și întârzierea lor pot varia."
    )


if run_analysis:
    if not ticker_input:
        st.error("Introdu un ticker valid.")
        st.stop()

    if not ticker_input.replace(".", "").replace("-", "").isalnum():
        st.error("Ticker-ul conține caractere nepermise.")
        st.stop()

    with st.spinner("Preiau datele financiare..."):
        try:
            data = fetch_financial_data(ticker_input)
        except Exception as exc:
            st.error(f"Eroare la preluarea datelor: {exc}")
            st.stop()

    market = data["market"]
    balance = data["balance"]

    st.subheader(
        f"{market['Companie']} ({market['Ticker']})"
    )

    st.caption(
        f"Sector: {market['Sector']} | "
        f"Industrie: {market['Industrie']} | "
        f"Monedă: {market['Monedă']}"
    )

    # KPI-uri
    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Preț curent",
        format_money(market["Preț curent"], market["Monedă"]),
    )
    k2.metric("P/E trailing", format_number(market["P/E trailing"]))
    k3.metric("P/E forward", format_number(market["P/E forward"]))
    k4.metric("P/B", format_number(market["P/B"]))

    k5, k6, k7, k8 = st.columns(4)

    k5.metric(
        "Capitalizare",
        format_money(
            market["Capitalizare bursieră"], market["Monedă"]
        ),
    )
    k6.metric("ROE", (
        f"{market['ROE']:.2%}"
        if market["ROE"] is not None else "N/A"
    ))
    k7.metric("Debt / Equity", format_number(market["Debt/Equity"]))
    k8.metric(
        "Free cash flow",
        format_money(market["Free cash flow"], market["Monedă"]),
    )

    st.divider()

    tab_balance, tab_multiples, tab_statements, tab_ai = st.tabs(
        [
            "📊 Bilanț",
            "📐 Multipli",
            "📑 Situații financiare",
            "🤖 Analiză AI",
        ]
    )

    with tab_balance:
        st.subheader("Indicatori din bilanț")

        balance_df = pd.DataFrame(
            [
                {"Indicator": key, "Valoare": value}
                for key, value in balance.items()
            ]
        )

        st.dataframe(
            balance_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Valorile bilanțului provin din cea mai recentă "
            "coloană disponibilă. Raporturile sunt fracții, "
            "nu procente, dacă nu sunt formatate explicit."
        )

    with tab_multiples:
        st.subheader("Multipli de evaluare")

        multiples = {
            "P/E trailing": market["P/E trailing"],
            "P/E forward": market["P/E forward"],
            "P/B": market["P/B"],
            "P/S": market["P/S"],
            "EV/EBITDA": market["EV/EBITDA"],
        }

        multiples_df = pd.DataFrame(
            [
                {
                    "Multiplu": name,
                    "Valoare": value,
                }
                for name, value in multiples.items()
            ]
        )

        st.dataframe(
            multiples_df,
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "Multiplii trebuie interpretați în contextul sectorului, "
            "creșterii, riscurilor și calității profitului. "
            "Un multiplu mic nu înseamnă automat o acțiune ieftină."
        )

    with tab_statements:
        st.subheader("Bilanț")
        if data["balance_sheet"].empty:
            st.warning("Bilanț indisponibil.")
        else:
            st.dataframe(
                data["balance_sheet"],
                use_container_width=True,
            )

        st.subheader("Cont de profit și pierdere")
        if data["income_stmt"].empty:
            st.warning("Cont de profit și pierdere indisponibil.")
        else:
            st.dataframe(
                data["income_stmt"],
                use_container_width=True,
            )

        st.subheader("Flux de numerar")
        if data["cash_flow"].empty:
            st.warning("Situația fluxurilor de numerar indisponibilă.")
        else:
            st.dataframe(
                data["cash_flow"],
                use_container_width=True,
            )

    with tab_ai:
        st.subheader("Analiză fundamentală cu AI")

        if not api_key:
            st.warning(
                "Introdu cheia Gemini în bara laterală pentru "
                "a genera raportul AI. Indicatorii financiari "
                "pot fi consultați fără această cheie."
            )
        elif st.button(
            "Generează raportul AI",
            type="primary",
            key="generate_ai",
        ):
            with st.spinner("Gemini analizează datele..."):
                try:
                    report = generate_ai_analysis(
                        data, api_key, model_name
                    )
                    st.session_state["ai_report"] = report
                    st.session_state["ai_ticker"] = ticker_input
                except Exception as exc:
                    st.error(
                        "Nu am putut genera raportul. Verifică "
                        "API key-ul, numele modelului, cota și "
                        f"conexiunea. Detalii: {exc}"
                    )

        if (
            st.session_state.get("ai_report")
            and st.session_state.get("ai_ticker") == ticker_input
        ):
            report = st.session_state["ai_report"]

            st.markdown(report)

            st.download_button(
                "⬇️ Descarcă raportul AI (.txt)",
                data=report,
                file_name=f"{ticker_input}_analiza.txt",
                mime="text/plain",
            )

    # Export date
    st.divider()
    st.subheader("Export date")

    export_data = {
        "ticker": ticker_input,
        "market": market,
        "balance": balance,
    }

    csv_df = pd.DataFrame(
        [
            {"Categorie": "Piață", "Indicator": key, "Valoare": value}
            for key, value in market.items()
        ] + [
            {"Categorie": "Bilanț", "Indicator": key, "Valoare": value}
            for key, value in balance.items()
        ]
    )

    col_csv, col_json = st.columns(2)

    with col_csv:
        st.download_button(
            "⬇️ Export CSV",
            data=csv_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{ticker_input}_date_financiare.csv",
            mime="text/csv",
        )

    with col_json:
        st.download_button(
            "⬇️ Export JSON",
            data=json.dumps(
                export_data,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            file_name=f"{ticker_input}_date_financiare.json",
            mime="application/json",
        )

else:
    st.info(
        "Introdu ticker-ul companiei în bara laterală și apasă "
        "«Analizează compania» pentru a prelua datele financiare."
    )

    st.markdown(
        """
        ### Ce analizează aplicația?

        - **Evaluare:** P/E, P/B, P/S și EV/EBITDA.
        - **Bilanț:** active, datorii, capitaluri și lichidități.
        - **Profitabilitate:** ROE, ROA, marjă netă și EPS.
        - **Flux de numerar:** free cash flow și situația cash flow.
        - **AI:** argumente pro și contra, riscuri și concluzie.
        """
    )

st.divider()

st.caption(
    "Instrument educațional de analiză. Nu constituie "
    "recomandare personalizată de investiții. Verifică datele "
    "în raportările oficiale ale companiei înainte de a lua "
    "decizii financiare."
)
