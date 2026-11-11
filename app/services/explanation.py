"""
Explain a recommendation end to end: demand forecast -> demand score -> price.

The forecast part is exact (per-feature contributions, app/ml/explain.py).
The price part retraces the pricing pipeline: which strategy, what formula,
which guardrails fired. Both are turned into a few plain-language sentences
for a person, alongside the numbers for a program.
"""
from dataclasses import dataclass
from decimal import Decimal

from app.config.settings import settings
from app.factories.strategy_factory import StrategyFactory
from app.models.forecast import DemandForecast
from app.models.pricing import PriceQuote, PricingStrategy
from app.services.forecast import DemandForecastService
from app.services.pricing import PricingService
from app.strategies.demand_based import DemandBasedPricingStrategy


@dataclass(frozen=True)
class Explanation:
    forecast: DemandForecast
    quote: PriceQuote
    strategy_description: str
    demand_factor: Decimal | None            # only for the demand_based strategy
    summary: tuple[str, ...]


class ExplanationService:
    def __init__(self, forecasts: DemandForecastService, pricing: PricingService) -> None:
        self._forecasts = forecasts
        self._pricing = pricing

    async def explain(self, product_id: str, strategy: PricingStrategy) -> Explanation:
        forecast = await self._forecasts.forecast(product_id)
        quote = await self._pricing.calculate_price(product_id, strategy)

        pricing_strategy = StrategyFactory.get_strategy(strategy)
        demand_factor = (
            pricing_strategy.factor(quote.product)
            if isinstance(pricing_strategy, DemandBasedPricingStrategy) else None
        )
        return Explanation(
            forecast=forecast,
            quote=quote,
            strategy_description=pricing_strategy.description,
            demand_factor=demand_factor,
            summary=_summary(forecast, quote, demand_factor),
        )


def money(amount: Decimal) -> str:
    return f"{settings.currency_symbol}{amount}"


def _summary(forecast: DemandForecast, quote: PriceQuote, demand_factor: Decimal | None) -> tuple[str, ...]:
    product = quote.product
    weekday = forecast.forecast_date.strftime("%A")
    lines = []

    # ---- 1. The forecast, compared with a normal day -----------------------------
    if forecast.typical_units > 0:
        change = (forecast.predicted_units / forecast.typical_units - 1) * 100
        comparison = f"{abs(change):.0f}% {'above' if change >= 0 else 'below'} a typical {weekday} ({forecast.typical_units:.0f})"
    else:
        comparison = "with no recent reference to compare against"
    lines.append(
        f"Next open day ({forecast.forecast_date:%a %d %b %Y}): about {forecast.predicted_units:.0f} units of "
        f"{product.name} expected, {comparison} - demand score {forecast.demand_score} (0.50 = normal)."
    )

    # ---- 2. Why ---------------------------------------------------------------------
    reasons = "; ".join(
        f"{f.description} ({'raises' if f.contribution > 0 else 'lowers'} it x{f.multiplier:.2f})"
        for f in forecast.factors[:3]
    )
    lines.append(f"Main reasons: {reasons}.")

    # ---- 3. The price ---------------------------------------------------------------
    if demand_factor is not None:
        lines.append(
            f"Price: competitor {money(product.competitor_price)} x {demand_factor:.3f} "
            f"(0.90 + 0.20 x demand score {product.demand_score})."
        )
    else:
        lines.append(f"Price: strategy '{quote.strategy.value}' applied (this strategy does not use the forecast).")
    if quote.guardrails_applied:
        lines.append("Guardrails then adjusted it: " + " ".join(quote.guardrails_applied))
    else:
        lines.append("No guardrail needed to change it.")
    lines.append(f"Recommended price: {money(quote.suggested_price)} ({quote.margin_percent}% margin).")

    # ---- 4. Anything unusual lately -----------------------------------------------
    if forecast.recent_anomalies:
        days = ", ".join(f"{a.sale_date:%d %b} ({a.units_sold} units: {a.reason.lower()})" for a in forecast.recent_anomalies)
        lines.append(f"Unusual days in the last 2 weeks: {days}.")
    return tuple(lines)
