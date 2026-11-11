"""
Evaluate anomaly detectors on labelled (injected) anomalies.

    python -m app.ml.anomaly.evaluate

Writes reports/anomaly_comparison.md and reports/anomaly_example.png.

The protocol (same discipline as the forecasting evaluation)
------------------------------------------------------------
    ... training history ... | validation (91 days) | test (last 91 days)
                               anomalies injected,     different anomalies injected
                               seed 1 -> choose the    (seed 2) -> report precision,
                               threshold here          recall, F1 ONCE

1. Inject labelled anomalies into the validation window (one copy of the data)
   and, separately, into the test window (another copy, different seed).
2. Isolation Forest is trained on the history BEFORE each window — no labels.
3. Each detector's threshold is the one with the best F1 on VALIDATION.
4. That fixed threshold is applied to the TEST window and scored once.

Why choose the threshold by F1?
-------------------------------
A detector outputs a score; the threshold turns it into yes/no.
    lower threshold  -> more flags -> higher recall, lower precision (alert fatigue)
    higher threshold -> fewer flags -> higher precision, lower recall (missed problems)
F1 (harmonic mean of the two) picks the balance point when both kinds of
error matter equally. Choosing it on validation and freezing it before the
test is what keeps the reported numbers honest.
"""
import logging
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score

from app.config.settings import resolve_path, settings
from app.ml.anomaly.detectors import build_detection_features, make_detectors
from app.ml.anomaly.inject import inject_anomalies
from app.ml.features import load_daily_sales
from app.ml.forecasting.evaluate import TEST_DAYS

logger = logging.getLogger(__name__)

VALIDATION_DAYS = 91
VALIDATION_SEED = 1
# Seed 2 drew the test anomalies during development, and those results were seen
# before the repeated-value rule was added. The final report therefore uses a
# FRESH draw of test anomalies the final design has never been scored on.
TEST_SEED = 3
ANOMALY_TYPES = ("spike", "drop", "flatline")


@dataclass
class Window:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    seed: int


@dataclass
class AnomalyEvaluation:
    results: pd.DataFrame          # one row per detector: threshold, precision, recall, F1, per-type recall
    test_rows: pd.DataFrame        # scored test window (for the plot)
    label_counts: dict
    windows: dict


def windows_for(daily: pd.DataFrame) -> tuple[Window, Window]:
    last = daily["sale_date"].max()
    test_start = last - pd.Timedelta(days=TEST_DAYS - 1)
    validation_start = test_start - pd.Timedelta(days=VALIDATION_DAYS)
    return (
        Window("validation", validation_start, test_start - pd.Timedelta(days=1), VALIDATION_SEED),
        Window("test", test_start, last, TEST_SEED),
    )


def labelled_window(daily: pd.DataFrame, window: Window) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(features for the history before the window, labelled features for the window)."""
    injected, labels = inject_anomalies(daily, window.start, window.end, seed=window.seed)
    features = build_detection_features(injected)
    features = features.merge(labels, on=["product_id", "sale_date"], how="left")
    features["is_anomaly"] = features["anomaly_type"].notna().astype(int)

    history = features[features["sale_date"] < window.start]
    scored = features[features["sale_date"].between(window.start, window.end)].reset_index(drop=True)
    return history, scored


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """The score threshold that maximises F1 on these (validation) labels."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])      # the last P/R point has no threshold


def run_evaluation(daily: pd.DataFrame) -> AnomalyEvaluation:
    validation, test = windows_for(daily)
    val_history, val_rows = labelled_window(daily, validation)
    test_history, test_rows = labelled_window(daily, test)

    results = []
    for val_detector, test_detector in zip(make_detectors(), make_detectors()):
        # 1. threshold chosen on validation
        val_detector.fit(val_history)
        threshold = best_f1_threshold(val_rows["is_anomaly"].to_numpy(), val_detector.score(val_rows))

        # 2. frozen threshold applied to the test window, once
        test_detector.fit(test_history)
        test_scores = test_detector.score(test_rows)
        flagged = (test_scores >= threshold).astype(int)
        test_rows[f"flag::{test_detector.name}"] = flagged

        truth = test_rows["is_anomaly"].to_numpy()
        row = {
            "detector": test_detector.name,
            "threshold": threshold,
            "precision": precision_score(truth, flagged, zero_division=0),
            "recall": recall_score(truth, flagged, zero_division=0),
            "f1": f1_score(truth, flagged, zero_division=0),
            "flags": int(flagged.sum()),
        }
        for anomaly_type in ANOMALY_TYPES:
            of_type = test_rows["anomaly_type"] == anomaly_type
            row[f"recall_{anomaly_type}"] = float(flagged[of_type.to_numpy()].mean()) if of_type.any() else float("nan")
        results.append(row)
        logger.info("%s | threshold %.3f | test P %.3f R %.3f F1 %.3f", row["detector"], threshold,
                    row["precision"], row["recall"], row["f1"])

    return AnomalyEvaluation(
        results=pd.DataFrame(results).set_index("detector"),
        test_rows=test_rows,
        label_counts={
            "validation": val_rows["anomaly_type"].value_counts().to_dict(),
            "test": test_rows["anomaly_type"].value_counts().to_dict(),
            "validation_rows": len(val_rows),
            "test_rows": len(test_rows),
        },
        windows={w.name: (w.start.date().isoformat(), w.end.date().isoformat()) for w in (validation, test)},
    )


# ------------------------------------------------------------------
# Reports
# ------------------------------------------------------------------

def comparison_markdown(evaluation: AnomalyEvaluation) -> str:
    counts = evaluation.label_counts
    val_from, val_to = evaluation.windows["validation"]
    test_from, test_to = evaluation.windows["test"]
    best = evaluation.results["f1"].idxmax()

    def type_counts(name):
        return ", ".join(f"{counts[name].get(t, 0)} {t}" for t in ANOMALY_TYPES)

    lines = [
        "# Anomaly Detection Comparison",
        "",
        "Generated by `python -m app.ml.anomaly.evaluate` from real data (UCI Online Retail II, 20 products) "
        "with injected, labelled anomalies. Only open days are judged (Saturdays, bank holidays and the "
        "Christmas break are planned closures).",
        "",
        f"- **Validation window (threshold chosen here):** {val_from} to {val_to} - "
        f"{counts['validation_rows']:,} open product-days, injected: {type_counts('validation')}",
        f"- **Test window (scored once):** {test_from} to {test_to} - "
        f"{counts['test_rows']:,} open product-days, injected: {type_counts('test')}",
        "- **Threshold rule:** the score cut-off with the best F1 on validation, then frozen.",
        "- **Test-set hygiene:** an earlier development run scored a different draw of test anomalies "
        "(seed 2); the repeated-value rule was added afterwards, based on validation evidence. These final "
        "numbers use a fresh draw (seed 3) that the final design was never tuned on.",
        "",
        "## Test results",
        "",
        "| Detector | Threshold | Flags | Precision | Recall | F1 | Recall: spike | Recall: drop | Recall: flatline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in evaluation.results.iterrows():
        marker = " **(best)**" if name == best else ""
        lines.append(
            f"| {name}{marker} | {r['threshold']:.3f} | {int(r['flags'])} | {r['precision']:.3f} | {r['recall']:.3f} "
            f"| {r['f1']:.3f} | {r['recall_spike']:.2f} | {r['recall_drop']:.2f} | {r['recall_flatline']:.2f} |"
        )
    lines += [
        "",
        "Thresholds are in each detector's own units: |z| in standard deviations (log scale), "
        "IQR fences in interquartile ranges, Isolation Forest in negated `score_samples`.",
        "",
        "Precision is a lower bound: the real data contains unlabelled natural extremes (one-off bulk "
        "orders); flagging one counts as a false positive here.",
        "",
        "![Example](anomaly_example.png)",
        "",
    ]
    return "\n".join(lines)


def plot_example(evaluation: AnomalyEvaluation, path) -> None:
    """One product's test window: demand, injected anomalies, and what the best detector flagged."""
    rows = evaluation.test_rows
    best = evaluation.results["f1"].idxmax()
    product_id = rows[rows["is_anomaly"] == 1]["product_id"].value_counts().index[0]
    series = rows[rows["product_id"] == product_id]
    flags = series[series[f"flag::{best}"] == 1]

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(series["sale_date"], series["units_sold"], color="black", linewidth=1.2, label="Units sold (open days)")
    markers = {"spike": ("^", "tab:red"), "drop": ("v", "tab:purple"), "flatline": ("s", "tab:green")}
    for anomaly_type, (marker, colour) in markers.items():
        injected = series[series["anomaly_type"] == anomaly_type]
        if len(injected):
            ax.scatter(injected["sale_date"], injected["units_sold"], marker=marker, s=70, color=colour,
                       label=f"Injected {anomaly_type}", zorder=3)
    ax.scatter(flags["sale_date"], flags["units_sold"], s=180, facecolors="none", edgecolors="tab:blue",
               linewidths=1.6, label=f"Flagged by {best}", zorder=4)
    ax.set_title(f"Product {product_id} - test window", loc="left", fontsize=10)
    ax.set_ylabel("Units / day")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_reports(evaluation: AnomalyEvaluation, reports) -> list:
    """Write the comparison table and example plot; return the files written."""
    reports.mkdir(parents=True, exist_ok=True)
    table, plot = reports / "anomaly_comparison.md", reports / "anomaly_example.png"
    table.write_text(comparison_markdown(evaluation), encoding="utf-8")
    plot_example(evaluation, plot)
    return [table, plot]


def main() -> None:
    from app.config.logging import setup_logging
    setup_logging()

    daily = load_daily_sales(resolve_path(settings.processed_data_dir) / "daily_sales.csv")
    evaluation = run_evaluation(daily)
    write_reports(evaluation, resolve_path(settings.reports_dir))
    logger.info("Reports written to %s", resolve_path(settings.reports_dir))
    print(comparison_markdown(evaluation).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
