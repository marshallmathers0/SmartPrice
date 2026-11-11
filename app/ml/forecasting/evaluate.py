"""
Compare forecasting models honestly — on time, the way they would be used.

    python -m app.ml.forecasting.evaluate

Writes reports/forecast_comparison.md and reports/forecast_actual_vs_predicted.png.

The protocol
------------
                              development period                         |   test period
    2009-12 ............................................................. | last 91 days
    [ fold 1 train ][val]                                                 |
    [ fold 2 train      ][val]                                            |   touched ONCE,
    [ fold 3 train           ][val]          walk-forward validation      |   after the model
    [ fold 4 train                ][val]     (model selection happens     |   is chosen
    [ fold 5 train                     ][val] only here)                  |

1. Chronological hold-out: the last TEST_DAYS days are the test set.
2. Walk-forward validation (TimeSeriesSplit) on the development period:
   each fold trains on everything BEFORE its validation window.
3. The model with the lowest mean validation WAPE is selected.
4. Only then is the test set scored — once. Nothing is tuned on it.

Why not a random train/test split?
----------------------------------
A random split puts days from December 2011 in training and days from
November 2011 in test. The model then "predicts" November having already seen
what happened after it — information that does not exist at forecast time.
Its score would be optimistic, and it would not hold up in production.
Splitting by date keeps every evaluation in the real direction: past -> future.

Folds split on DATES, not rows: all 20 products of one day always land on the
same side, so no product's "tomorrow" is used to predict another's "today".
"""
import logging
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")    # write image files; no window / display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from app.config.settings import resolve_path, settings
from app.ml.features import FEATURES, TARGET, build_features, load_daily_sales
from app.ml.forecasting.models import make_candidates, predict_demand
from app.ml.metrics import score

logger = logging.getLogger(__name__)

TEST_DAYS = 91            # ~3 months held out (10 Sep - 9 Dec 2011)
CV_FOLDS = 5
CV_VALIDATION_DAYS = 60   # each walk-forward validation window
SELECTION_METRIC = "wape"


@dataclass
class EvaluationResult:
    cv_scores: pd.DataFrame               # one row per (model, fold)
    cv_summary: pd.DataFrame              # mean/std per model
    test_scores: pd.DataFrame             # one row per model
    best_model_name: str
    test_predictions: pd.DataFrame        # test rows + one prediction column per model
    split_info: dict = field(default_factory=dict)


def split_by_date(features: pd.DataFrame, test_days: int = TEST_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Everything before the last `test_days` days is development; the rest is test."""
    test_start = features["sale_date"].max() - pd.Timedelta(days=test_days - 1)
    development = features[features["sale_date"] < test_start].reset_index(drop=True)
    test = features[features["sale_date"] >= test_start].reset_index(drop=True)
    return development, test


def walk_forward_folds(development: pd.DataFrame, n_splits: int = CV_FOLDS, validation_days: int = CV_VALIDATION_DAYS):
    """
    Yield (train_rows, validation_rows) boolean masks.

    TimeSeriesSplit runs over the sorted UNIQUE DATES, then each date split is
    mapped back to rows — so a whole day (all products) is either train or validation.
    """
    dates = np.sort(development["sale_date"].unique())
    for train_idx, val_idx in TimeSeriesSplit(n_splits=n_splits, test_size=validation_days).split(dates):
        train_dates, val_dates = dates[train_idx], dates[val_idx]
        assert train_dates.max() < val_dates.min()            # the defining property: past -> future
        yield (development["sale_date"].isin(train_dates).to_numpy(),
               development["sale_date"].isin(val_dates).to_numpy())


def run_evaluation(daily: pd.DataFrame) -> EvaluationResult:
    features = build_features(daily)
    development, test = split_by_date(features)

    # ---- 1. Walk-forward validation: model selection ----------------------
    cv_rows = []
    for fold, (train_mask, val_mask) in enumerate(walk_forward_folds(development), start=1):
        train, val = development[train_mask], development[val_mask]
        for name, model in make_candidates().items():
            model.fit(train[FEATURES], train[TARGET])
            cv_rows.append({"model": name, "fold": fold, **score(val[TARGET], predict_demand(model, val))})
        logger.info("Fold %d | train %s..%s | validate %s..%s", fold,
                    train["sale_date"].min().date(), train["sale_date"].max().date(),
                    val["sale_date"].min().date(), val["sale_date"].max().date())

    cv_scores = pd.DataFrame(cv_rows)
    cv_summary = (
        cv_scores.groupby("model", sort=False)[["mae", "rmse", "wape"]]
        .agg(["mean", "std"])
    )
    best_model_name = cv_summary[(SELECTION_METRIC, "mean")].idxmin()
    logger.info("Selected by validation %s: %s", SELECTION_METRIC.upper(), best_model_name)

    # ---- 2. Test set: scored ONCE, after selection ------------------------
    # Every candidate is refit on the full development period and scored, so the
    # report shows the whole picture — but the choice above is already final.
    test_rows, test_predictions = [], test[["product_id", "sale_date", TARGET]].copy()
    for name, model in make_candidates().items():
        model.fit(development[FEATURES], development[TARGET])
        predictions = predict_demand(model, test)
        test_predictions[name] = predictions
        test_rows.append({"model": name, **score(test[TARGET], predictions)})

    return EvaluationResult(
        cv_scores=cv_scores,
        cv_summary=cv_summary,
        test_scores=pd.DataFrame(test_rows).set_index("model"),
        best_model_name=best_model_name,
        test_predictions=test_predictions,
        split_info={
            "development_from": development["sale_date"].min().date().isoformat(),
            "development_to": development["sale_date"].max().date().isoformat(),
            "test_from": test["sale_date"].min().date().isoformat(),
            "test_to": test["sale_date"].max().date().isoformat(),
            "development_rows": len(development),
            "test_rows": len(test),
            "cv_folds": CV_FOLDS,
            "cv_validation_days": CV_VALIDATION_DAYS,
        },
    )


# ------------------------------------------------------------------
# Reports
# ------------------------------------------------------------------

def _fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def comparison_markdown(result: EvaluationResult) -> str:
    info = result.split_info
    baseline = result.test_scores.loc["Seasonal naive (same weekday last week)", "wape"]
    best_test = result.test_scores.loc[result.best_model_name, "wape"]

    lines = [
        "# Forecast Model Comparison",
        "",
        "Generated by `python -m app.ml.forecasting.evaluate` from real data "
        "(UCI Online Retail II, 20 products). Target: units sold per product per day (next-day forecast).",
        "",
        f"- **Development period:** {info['development_from']} to {info['development_to']} ({info['development_rows']:,} rows)",
        f"- **Test period (held out, scored once):** {info['test_from']} to {info['test_to']} ({info['test_rows']:,} rows)",
        f"- **Model selection:** walk-forward validation, {info['cv_folds']} folds x {info['cv_validation_days']} days, "
        f"lowest mean validation {SELECTION_METRIC.upper()}",
        "",
        "## Walk-forward validation (used to choose the model)",
        "",
        "| Model | MAE | RMSE | WAPE (mean ± std over folds) |",
        "|---|---:|---:|---:|",
    ]
    for name, row in result.cv_summary.iterrows():
        marker = " **(selected)**" if name == result.best_model_name else ""
        lines.append(
            f"| {name}{marker} | {_fmt(row[('mae', 'mean')], 2)} | {_fmt(row[('rmse', 'mean')], 2)} "
            f"| {_fmt(row[('wape', 'mean')], 3)} ± {_fmt(row[('wape', 'std')], 3)} |"
        )
    lines += [
        "",
        "## Test set (held out until the end)",
        "",
        "| Model | MAE | RMSE | WAPE |",
        "|---|---:|---:|---:|",
    ]
    for name, row in result.test_scores.iterrows():
        marker = " **(selected)**" if name == result.best_model_name else ""
        lines.append(f"| {name}{marker} | {_fmt(row['mae'], 2)} | {_fmt(row['rmse'], 2)} | {_fmt(row['wape'], 3)} |")
    lines += [
        "",
        f"**Selected model: {result.best_model_name}.** Test WAPE {best_test:.3f} vs. "
        f"{baseline:.3f} for the seasonal-naive baseline "
        f"({(1 - best_test / baseline) * 100:.1f}% lower error).",
        "",
        "Metrics: MAE and RMSE are in units; WAPE = sum(|error|) / sum(actual demand). "
        "WAPE is used instead of MAPE because 22% of product-days have zero sales.",
        "",
        "![Actual vs predicted](forecast_actual_vs_predicted.png)",
        "",
    ]
    return "\n".join(lines)


def plot_actual_vs_predicted(result: EvaluationResult, path, products: int = 4) -> None:
    """Test period, for the highest-volume products: actual vs the selected model vs seasonal naive."""
    predictions = result.test_predictions
    top = predictions.groupby("product_id", observed=True)[TARGET].sum().nlargest(products).index
    fig, axes = plt.subplots(len(top), 1, figsize=(11, 2.6 * len(top)), sharex=True)
    for ax, product_id in zip(np.atleast_1d(axes), top):
        series = predictions[predictions["product_id"] == product_id]
        ax.plot(series["sale_date"], series[TARGET], color="black", linewidth=1.4, label="Actual")
        ax.plot(series["sale_date"], series[result.best_model_name], color="tab:blue", linewidth=1.2,
                label=f"{result.best_model_name} (selected)")
        ax.plot(series["sale_date"], series["Seasonal naive (same weekday last week)"], color="tab:orange",
                linewidth=1, linestyle="--", alpha=0.8, label="Seasonal naive")
        ax.set_title(f"Product {product_id}", fontsize=10, loc="left")
        ax.set_ylabel("Units / day")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Test period: actual vs predicted daily demand", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_reports(result: EvaluationResult, reports) -> list:
    """Write the comparison table and plot; return the files written."""
    reports.mkdir(parents=True, exist_ok=True)
    table, plot = reports / "forecast_comparison.md", reports / "forecast_actual_vs_predicted.png"
    table.write_text(comparison_markdown(result), encoding="utf-8")
    plot_actual_vs_predicted(result, plot)
    return [table, plot]


def main() -> None:
    from app.config.logging import setup_logging
    setup_logging()

    daily = load_daily_sales(resolve_path(settings.processed_data_dir) / "daily_sales.csv")
    result = run_evaluation(daily)
    write_reports(result, resolve_path(settings.reports_dir))
    logger.info("Reports written to %s", resolve_path(settings.reports_dir))
    print(comparison_markdown(result).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
