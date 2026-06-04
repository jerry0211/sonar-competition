"""Phase 3: synthetic-simulator detection drill, end to end.

Pick any preset scenario (a 'differing initial condition') and run the full loop:
simulate labelled audio -> featurize -> group-aware CV -> detection metrics.

    python experiments/03_synthetic_sim/run.py                      # baseline
    python experiments/03_synthetic_sim/run.py --scenario rare_target
    python experiments/03_synthetic_sim/run.py --scenario quiet_sub_loud_shipping
    python experiments/03_synthetic_sim/run.py --list               # show options

This is the rehearsal engine: change --scenario and you've changed the task.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from sonar_toolkit.simulator import SCENARIOS, generate_dataset
from sonar_toolkit.features import featurize_batch
from sonar_toolkit.models import gbm_fit_predict
from sonar_toolkit.validation import cross_validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="baseline", choices=list(SCENARIOS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list", action="store_true", help="list scenarios and exit")
    args = ap.parse_args()

    if args.list:
        for name, scn in SCENARIOS.items():
            print(f"  {name:24s} mode={scn.mode:7s} "
                  f"snr={scn.snr_db_range} prevalence={scn.target_prevalence} "
                  f"ambient={scn.ambient}")
        return

    scn = SCENARIOS[args.scenario]
    print(f"scenario: {scn.name}  (mode={scn.mode}, ambient={scn.ambient}, "
          f"snr={scn.snr_db_range}, prevalence={scn.target_prevalence})\n")

    data = generate_dataset(scn, seed=args.seed)
    y, groups = data["labels"], data["groups"]
    print(f"simulated {len(np.unique(groups))} recordings -> {len(y)} segments, "
          f"positives={int(y.sum())}/{len(y)}")

    print("featurizing...")
    X = featurize_batch(data["waveforms"], data["sr"])
    print(f"feature matrix: {X.shape}\n")

    print("group-aware 5-fold CV (HistGradientBoosting):")
    summary = cross_validate(X, y, gbm_fit_predict, groups=groups,
                             n_splits=5, max_fpr=0.1, seed=args.seed)

    print("\nsummary (mean +/- std):")
    for k, v in summary.items():
        print(f"  {k:20s} {v['mean']:.3f} +/- {v['std']:.3f}")


if __name__ == "__main__":
    main()
