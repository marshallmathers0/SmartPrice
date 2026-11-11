from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field

from app.models.forecast import DemandForecast
from app.models.pricing import PricingStrategy
from app.services.explanation import Explanation


class ForecastFactorResponse(BaseModel):
    feature: str = Field(examples=["lag_7"])
    description: str = Field(examples=["Sales on the same weekday last week: 41 units"])
    effect: str = Field(examples=["lowers"])
    multiplier: float = Field(description="Multiplies (1 + forecast); 1.35 = raises it ~35%")
    contribution: float = Field(description="Exact contribution on the log(1 + units) scale")


class AnomalyFlagResponse(BaseModel):
    sale_date: date
    units_sold: int
    reason: str


class ForecastResponse(BaseModel):
    """Next-day demand forecast with the reasons behind it (GET /forecast/{product_id})."""

    product_id: str
    forecast_date: date = Field(description="The next day the shop is open after the last sales data")
    history_end: date
    predicted_units: float
    typical_units: float = Field(description="The model's median forecast for the same weekday over the last 4 weeks")
    recent_actual_units: float = Field(description="Median units actually sold on those 4 days")
    demand_score: Decimal = Field(description="0.50 = a normal day, 1.00 = double the usual or more, 0.00 = none")
    model_version: str
    base_units: float = Field(description="The forecast before any feature is considered")
    top_factors: list[ForecastFactorResponse]
    recent_anomalies: list[AnomalyFlagResponse]

    @classmethod
    def from_domain(cls, forecast: DemandForecast) -> "ForecastResponse":
        return cls(
            product_id=forecast.product_id,
            forecast_date=forecast.forecast_date,
            history_end=forecast.history_end,
            predicted_units=forecast.predicted_units,
            typical_units=forecast.typical_units,
            recent_actual_units=forecast.recent_actual_units,
            demand_score=forecast.demand_score,
            model_version=forecast.model_version,
            base_units=forecast.base_units,
            top_factors=[
                ForecastFactorResponse(
                    feature=f.feature,
                    description=f.description,
                    effect="raises" if f.contribution > 0 else "lowers",
                    multiplier=f.multiplier,
                    contribution=f.contribution,
                )
                for f in forecast.factors
            ],
            recent_anomalies=[
                AnomalyFlagResponse(sale_date=a.sale_date, units_sold=a.units_sold, reason=a.reason)
                for a in forecast.recent_anomalies
            ],
        )


class PriceExplanationResponse(BaseModel):
    strategy: PricingStrategy
    strategy_description: str
    competitor_price: Decimal
    cost_price: Decimal
    demand_score: Decimal
    demand_factor: Optional[Decimal] = Field(default=None, description="Only for demand_based: 0.90 + 0.20 x demand_score")
    suggested_price: Decimal
    margin_percent: Decimal
    guardrails_applied: list[str]


class ExplanationResponse(BaseModel):
    """Why this forecast, and why this price (GET /explain/{product_id})."""

    product_id: str
    product_name: str
    summary: list[str] = Field(description="The explanation in plain language")
    forecast: ForecastResponse
    price: PriceExplanationResponse

    @classmethod
    def from_domain(cls, explanation: Explanation) -> "ExplanationResponse":
        quote = explanation.quote
        return cls(
            product_id=quote.product.product_id,
            product_name=quote.product.name,
            summary=list(explanation.summary),
            forecast=ForecastResponse.from_domain(explanation.forecast),
            price=PriceExplanationResponse(
                strategy=quote.strategy,
                strategy_description=explanation.strategy_description,
                competitor_price=quote.product.competitor_price,
                cost_price=quote.product.cost_price,
                demand_score=quote.product.demand_score,
                demand_factor=explanation.demand_factor,
                suggested_price=quote.suggested_price,
                margin_percent=quote.margin_percent,
                guardrails_applied=list(quote.guardrails_applied),
            ),
        )
