"""
SmartPrice — interactive demo (Streamlit).

    streamlit run demo/app.py

A thin client: every number on the page comes from the SmartPrice API.
It never imports the app's code — it talks HTTP, like any other client.

Settings (environment variables):
    SMARTPRICE_API_URL   where the API runs   (default http://localhost:8000)
    CURRENCY_SYMBOL      shown before prices  (default £ — the dataset is a UK wholesaler)
"""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("SMARTPRICE_API_URL", "http://localhost:8000").rstrip("/")
CURRENCY = os.environ.get("CURRENCY_SYMBOL", "£")
TIMEOUT = 60          # seconds — a free-tier API may need ~a minute to wake up
DEFAULT_PRODUCT = "85123A"

STRATEGIES = {
    "demand_based": "Demand-based (ML forecast)",
    "competitive": "Competitive (3% below competitor)",
    "cost_plus": "Cost-plus (40% margin)",
    "dynamic": "Dynamic (blend)",
    "penetration": "Penetration (10% above cost)",
}

# Short chart labels for the model's features (the API's full sentences go in the table)
SHORT_NAMES = {
    "product_id": "Product level",
    "lag_1": "Yesterday",
    "lag_7": "Same day last week",
    "lag_14": "Same day 2 weeks ago",
    "roll_mean_7": "7-day average",
    "roll_std_7": "7-day volatility",
    "roll_mean_28": "28-day average",
    "roll_std_28": "28-day volatility",
    "day_of_week": "Day of week",
    "month": "Month",
    "is_holiday": "Bank holiday",
    "is_christmas_break": "Christmas closure",
    "last_price": "Last price",
    "price_vs_28d": "Price vs usual",
}

st.set_page_config(page_title="SmartPrice", page_icon=":chart_with_upwards_trend:", layout="wide")


# ------------------------------------------------------------------
# API calls
# ------------------------------------------------------------------

def call(path: str, **params) -> tuple[int, dict | list]:
    """GET an API path; return (status, json). Network errors become status 0."""
    try:
        response = requests.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT)
        return response.status_code, response.json()
    except requests.RequestException as exc:
        return 0, {"detail": f"Cannot reach the API at {API_URL} ({type(exc).__name__})"}


@st.cache_data(ttl=300, show_spinner=False)
def load_products() -> list[dict]:
    status, body = call("/api/v1/products/")
    return body if status == 200 else []


@st.cache_data(ttl=300, show_spinner=False)
def load_model_info() -> dict | None:
    status, body = call("/api/v1/models/current")
    return body if status == 200 else None


def money(value) -> str:
    return f"{CURRENCY}{float(value):,.2f}"


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

st.sidebar.title("SmartPrice")
st.sidebar.caption("ML demand forecasting + explainable pricing")

with st.spinner("Connecting to the API (a free-tier server can take up to a minute to wake up)..."):
    health_status, health = call("/health")
    products = load_products()

if health_status != 200 or not products:
    st.error(health.get("detail", "The API is not available.") if isinstance(health, dict) else "The API is not available.")
    st.stop()

st.sidebar.markdown(
    f"**API** `{health['status']}` · database `{health['database']}` · "
    f"cache `{health['cache']}` · model `{health['model']}`"
)

labels = {p["product_id"]: f"{p['name']} ({p['product_id']})" for p in products}
ids = list(labels)
# ?product=22197 in the URL opens that product directly (handy for sharing a demo link)
requested = st.query_params.get("product", DEFAULT_PRODUCT)
product_id = st.sidebar.selectbox(
    "Product",
    ids,
    index=ids.index(requested) if requested in ids else 0,
    format_func=labels.get,
    help="20 real products from the UCI Online Retail II dataset, plus 3 demo SKUs without sales history.",
)
strategy = st.sidebar.selectbox("Pricing strategy", list(STRATEGIES), format_func=STRATEGIES.get)

model = load_model_info()
if model:
    with st.sidebar.expander("Model", expanded=False):
        st.markdown(
            f"**{model['model_name']}** · version `{model['version']}`  \n"
            f"Trained on {model['trained_from']} to {model['trained_to']} ({model['training_rows']:,} rows)  \n"
            f"Held-out test WAPE **{model['test']['wape']:.3f}** · MAE {model['test']['mae']:.1f} units  \n"
            f"Anomaly detector: {model['anomaly_detector']['name']} (test F1 {model['anomaly_detector']['f1']:.3f})  \n"
            f"Dataset SHA-256 `{model['dataset_sha256'][:12]}...`"
        )

st.sidebar.markdown(f"[API docs (Swagger)]({API_URL}/docs)")


# ------------------------------------------------------------------
# Main page
# ------------------------------------------------------------------

st.title(labels[product_id])

with st.spinner("Forecasting and pricing..."):
    status, explanation = call(f"/api/v1/explain/{product_id}", strategy=strategy)

if status == 422:
    st.info(
        f"{explanation['detail']}. This is one of the hand-made demo products - "
        "pick one of the real dataset products to see the forecast."
    )
    st.stop()
if status != 200:
    st.error(explanation.get("detail", f"The API answered {status}."))
    st.stop()

forecast, price = explanation["forecast"], explanation["price"]
predicted, typical = forecast["predicted_units"], forecast["typical_units"]
competitor, suggested = float(price["competitor_price"]), float(price["suggested_price"])

# ---- Headline numbers ------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    f"Forecast for {pd.Timestamp(forecast['forecast_date']):%a %d %b %Y}",
    f"{predicted:.0f} units",
    f"{(predicted / typical - 1) * 100:+.0f}% vs a typical {pd.Timestamp(forecast['forecast_date']):%A}" if typical else None,
)
col2.metric("Demand score", f"{float(forecast['demand_score']):.2f}", help="0.50 = a normal day, 1.00 = double the usual demand or more")
col3.metric(
    "Recommended price",
    money(suggested),
    f"{(suggested / competitor - 1) * 100:+.1f}% vs competitor {money(competitor)}",
    delta_color="off",
)
col4.metric("Margin", f"{float(price['margin_percent']):.1f}%", help=f"Cost {money(price['cost_price'])} (dataset assumption: 60% of median price)")

# ---- Why ---------------------------------------------------------------------
st.subheader("Why")
for line in explanation["summary"]:
    st.markdown(f"- {line}")

left, right = st.columns([3, 2])

with left:
    st.subheader("What drives this forecast")
    factors = pd.DataFrame(forecast["top_factors"])
    factors["Factor"] = factors["feature"].map(SHORT_NAMES).fillna(factors["feature"])
    # Log scale: multipliers range from x0.3 to x27, which a percent axis cannot show side by side.
    # +0.69 means "doubles the forecast", -0.69 "halves it".
    factors["Effect (log scale)"] = factors["contribution"].round(3)
    factors["Multiplier"] = factors["multiplier"].map(lambda m: f"x{m:.2f}")
    st.bar_chart(factors.set_index("Factor")["Effect (log scale)"], horizontal=True, height=240)
    st.dataframe(
        factors[["Factor", "Multiplier", "description"]].rename(columns={"description": "Detail"}),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        f"Exact contributions of the deployed model. The forecast starts from {forecast['base_units']:.0f} units "
        "and each factor multiplies it (bars on a log scale: +0.69 doubles the forecast, -0.69 halves it)."
    )

with right:
    st.subheader("How the price was set")
    rows = {
        "Strategy": price["strategy_description"],
        "Competitor price": money(price["competitor_price"]),
        "Cost": money(price["cost_price"]),
        "Demand factor": f"x{float(price['demand_factor']):.3f}" if price["demand_factor"] else "not used",
        "Guardrails": " ".join(price["guardrails_applied"]) or "none needed",
        "Recommended price": money(price["suggested_price"]),
    }
    st.table(pd.DataFrame(rows.items(), columns=["", "Value"]).set_index(""))

# ---- Anomalies ----------------------------------------------------------------
st.subheader("Unusual days (last 2 weeks)")
anomalies = forecast["recent_anomalies"]
if anomalies:
    st.dataframe(
        pd.DataFrame(anomalies).rename(columns={"sale_date": "Date", "units_sold": "Units sold", "reason": "Why it was flagged"}),
        hide_index=True,
        use_container_width=True,
    )
else:
    st.success("No unusual days - demand in the last two weeks looks normal.")

st.caption(
    f"Data: UCI Online Retail II (real UK wholesale transactions, 2009-2011); history ends {forecast['history_end']}. "
    f"Model {forecast['model_version']}. Everything on this page comes from the SmartPrice API at {API_URL}."
)
