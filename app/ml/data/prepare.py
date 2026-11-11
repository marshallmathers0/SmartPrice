"""
Turn the raw UCI Online Retail II transactions into clean daily demand series.

    python -m app.ml.data.prepare

Input   data/raw/online_retail_ii.zip      (1,067,371 transaction lines, Dec 2009 – Dec 2011)
Output  data/processed/daily_sales.csv     one row per product per calendar day
        data/processed/products.csv        the product catalog derived from it
        data/processed/metadata.json       row counts, date range, SHA-256 hashes

Dataset: "Online Retail II", UCI Machine Learning Repository,
https://archive.ics.uci.edu/dataset/502/online+retail+ii — real transactions
of a UK online gift wholesaler.

Every cleaning rule below is a decision an interviewer may ask about, so each
one says WHY it exists.
"""
import hashlib
import json
import logging
import zipfile
from pathlib import Path

import pandas as pd

from app.config.settings import resolve_path, settings

logger = logging.getLogger(__name__)

# How many products to model. The most CONSISTENT sellers (sold on the most
# days) are chosen, not the biggest sellers: a product sold in one huge order
# has a "series" that is almost all zeros and nothing to learn from.
TOP_N_PRODUCTS = 20

# The dataset has selling prices but no costs and no competitor prices.
# These two rules are ASSUMPTIONS, documented here and in the DEVLOG:
#   competitor_price = the product's median selling price (a "market price" proxy)
#   cost_price       = 60% of that median (a typical wholesale gross margin of ~40%)
COST_RATIO = 0.60

# Real product codes are 5 digits with an optional letter (85123A).
# Everything else is a service line, not a product: POST (postage), M (manual),
# D (discount), BANK CHARGES, AMAZONFEE, DCGS... (gift sets), etc.
PRODUCT_CODE_PATTERN = r"^\d{5}[A-Za-z]?$"

EXCEL_FILE_IN_ZIP = "online_retail_II.xlsx"


# ------------------------------------------------------------------
# Steps (pure functions — each one is unit-tested on tiny frames)
# ------------------------------------------------------------------

def load_raw(zip_path: Path) -> pd.DataFrame:
    """Read both sheets of the workbook (2009-2010 and 2010-2011) into one frame."""
    with zipfile.ZipFile(zip_path) as archive, archive.open(EXCEL_FILE_IN_ZIP) as workbook:
        sheets = pd.read_excel(workbook, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    return pd.concat(sheets.values(), ignore_index=True)


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only real sales of real products.

    1. Exact duplicate lines are dropped. The two sheets overlap on
       1–9 Dec 2010, so those days appear twice in the raw file.
    2. Cancellations are dropped: invoice numbers starting with "C".
    3. Quantity <= 0 (returns, stock adjustments) and price <= 0 (free
       samples, write-offs) are dropped — they are not customer demand.
    4. Non-product codes (postage, fees, discounts) are dropped.
    """
    frame = raw.drop_duplicates()
    is_cancellation = frame["Invoice"].astype(str).str.startswith("C")
    is_product = frame["StockCode"].astype(str).str.match(PRODUCT_CODE_PATTERN)
    frame = frame[~is_cancellation & (frame["Quantity"] > 0) & (frame["Price"] > 0) & is_product]
    return frame.assign(
        StockCode=frame["StockCode"].astype(str).str.upper(),
        day=frame["InvoiceDate"].dt.normalize(),
    )


def select_top_products(clean: pd.DataFrame, n: int = TOP_N_PRODUCTS) -> list[str]:
    """The n products sold on the most distinct days (ties broken by code, so it is deterministic)."""
    days_sold = clean.groupby("StockCode")["day"].nunique().rename("days_sold").reset_index()
    ranked = days_sold.sort_values(["days_sold", "StockCode"], ascending=[False, True])
    return ranked["StockCode"].head(n).tolist()


def build_daily_sales(clean: pd.DataFrame, product_ids: list[str]) -> pd.DataFrame:
    """
    One row per product per CALENDAR day, with zero-sale days filled in.

    Why fill zeros? A day with no sales is information (the shop was closed,
    or nobody bought). If those days were missing, "lag 7" would silently mean
    "7 rows ago" instead of "7 days ago", and the model would learn from a
    distorted calendar.

    avg_price is the quantity-weighted average price actually paid that day.
    On zero-sale days nobody paid anything, so it is left empty (NaN) here —
    the feature pipeline decides later how to fill it, using past data only.
    """
    sold = clean[clean["StockCode"].isin(product_ids)].assign(revenue=lambda f: f["Quantity"] * f["Price"])
    daily = sold.groupby(["StockCode", "day"], as_index=False).agg(
        units_sold=("Quantity", "sum"),
        revenue=("revenue", "sum"),
        orders=("Invoice", "nunique"),
    )

    # The calendar spans the whole dataset, so every product has the same days
    calendar = pd.date_range(clean["day"].min(), clean["day"].max(), freq="D")
    grid = pd.MultiIndex.from_product([sorted(product_ids), calendar], names=["StockCode", "day"])
    dense = daily.set_index(["StockCode", "day"]).reindex(grid).reset_index()

    dense["units_sold"] = dense["units_sold"].fillna(0).astype(int)
    dense["orders"] = dense["orders"].fillna(0).astype(int)
    dense["avg_price"] = (dense["revenue"] / dense["units_sold"].where(dense["units_sold"] > 0)).round(4)

    return dense.rename(columns={"StockCode": "product_id", "day": "sale_date"})[
        ["product_id", "sale_date", "units_sold", "orders", "avg_price"]
    ]


def build_catalog(clean: pd.DataFrame, product_ids: list[str]) -> pd.DataFrame:
    """Product rows for the database: name, competitor price and cost (see COST_RATIO)."""
    sold = clean[clean["StockCode"].isin(product_ids)]
    grouped = sold.groupby("StockCode")
    catalog = pd.DataFrame({
        # The most common description; titled for display ("White Hanging Heart T-Light Holder")
        "name": grouped["Description"].agg(lambda s: s.dropna().mode().iat[0])
                .str.replace(r"\s+", " ", regex=True).str.strip(" .").str.title(),
        "competitor_price": grouped["Price"].median().round(2),
    })
    catalog["cost_price"] = (catalog["competitor_price"] * COST_RATIO).round(2).clip(lower=0.01)
    return catalog.rename_axis("product_id").reset_index()[["product_id", "name", "cost_price", "competitor_price"]]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------
# Command
# ------------------------------------------------------------------

def main() -> None:
    from app.config.logging import setup_logging
    setup_logging()

    raw_path = resolve_path(settings.raw_data_path)
    out_dir = resolve_path(settings.processed_data_dir)
    if not raw_path.exists():
        raise SystemExit(
            f"Raw dataset not found at {raw_path}.\n"
            "Download it from https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
        )

    logger.info("Reading %s (takes about a minute - it is a 45 MB Excel file)", raw_path.name)
    raw = load_raw(raw_path)
    clean = clean_transactions(raw)
    product_ids = select_top_products(clean)
    daily = build_daily_sales(clean, product_ids)
    catalog = build_catalog(clean, product_ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    daily_path = out_dir / "daily_sales.csv"
    catalog_path = out_dir / "products.csv"
    daily.to_csv(daily_path, index=False, date_format="%Y-%m-%d")
    catalog.to_csv(catalog_path, index=False)

    metadata = {
        "source": "UCI Online Retail II — https://archive.ics.uci.edu/dataset/502/online+retail+ii",
        "raw_sha256": sha256_of(raw_path),
        "daily_sales_sha256": sha256_of(daily_path),   # the "dataset version" logged with every training run
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "products": len(product_ids),
        "date_from": daily["sale_date"].min().strftime("%Y-%m-%d"),
        "date_to": daily["sale_date"].max().strftime("%Y-%m-%d"),
        "daily_rows": int(len(daily)),
        "zero_sale_rows": int((daily["units_sold"] == 0).sum()),
        "cost_ratio_assumption": COST_RATIO,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    logger.info("Prepared data | %s", json.dumps(metadata))


if __name__ == "__main__":
    main()
