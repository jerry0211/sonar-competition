"""Phase 1 baseline: UCI Sonar (Mines vs Rocks), the full loop end to end.

Run:
    python scripts/download_data.py uci
    python experiments/01_uci_tabular/run.py

This is intentionally tiny. The point is to prove the train -> validate ->
report loop works and to make the metric suite real. Everything here is just
wiring; the actual logic lives in sonar_toolkit.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from sonar_toolkit.loaders import load_tabular
from sonar_toolkit.models import gbm_fit_predict
from sonar_toolkit.validation import cross_validate


def main():
    data = ROOT / "data" / "sonar.all-data"
    if not data.exists():
        raise SystemExit("Run: python scripts/download_data.py uci")

    # 'M' (metal/mine) is the positive/detection class.
    X, y = load_tabular(data, positive_label="M")
    print(f"loaded X={X.shape}, positives={int(y.sum())}/{len(y)}\n")

    print("5-fold CV (HistGradientBoosting):")
    # No groups: each UCI row is an independent measurement.
    summary = cross_validate(X, y, gbm_fit_predict, n_splits=5, max_fpr=0.1)

    print("\nsummary (mean +/- std):")
    for k, v in summary.items():
        print(f"  {k:20s} {v['mean']:.3f} +/- {v['std']:.3f}")


if __name__ == "__main__":
    main()
