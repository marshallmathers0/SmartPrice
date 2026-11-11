from dataclasses import dataclass
from datetime import date
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class DailySales:
    """One product's demand on one calendar day (a row of sales_history)."""

    product_id: str
    sale_date: date
    units_sold: int
    avg_price: Decimal | None        # None on zero-sale days: nobody paid anything


@dataclass(frozen=True, slots=True)
class ForecastFactor:
    """One feature's share of a forecast, in plain language and in numbers."""

    feature: str                     # e.g. "lag_7"
    description: str                 # e.g. "Sales on the same weekday last week: 41 units"
    contribution: float              # on the log(1 + units) scale; exact
    multiplier: float                # exp(contribution): x1.35 = raises the forecast ~35%


@dataclass(frozen=True, slots=True)
class AnomalyFlag:
    """A recent day the deployed anomaly detector considers unusual."""

    sale_date: date
    units_sold: int
    reason: str


@dataclass(frozen=True, slots=True)
class DemandForecast:
    """
    Next-day demand for one product, with the reasons behind it.

    demand_score turns the forecast into the 0..1 signal the pricing engine
    already understands (Product.demand_score):
        demand_score = clip(forecast / typical, 0, 2) / 2
    where typical = the model's own forecast for the same weekday over the
    last 4 weeks (so the model's systematic caution cancels out)
    so 0.50 = a normal day, 1.00 = double the usual demand or more,
    0.00 = no demand expected.
    """

    product_id: str
    forecast_date: date
    history_end: date
    predicted_units: float
    typical_units: float             # the model's median forecast for the same weekday, last 4 weeks
    recent_actual_units: float       # what actually sold on those 4 days (median) - for context
    demand_score: Decimal
    model_version: str
    base_units: float                # the forecast before any feature is considered
    factors: tuple[ForecastFactor, ...]
    recent_anomalies: tuple[AnomalyFlag, ...]
