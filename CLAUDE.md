# CLAUDE.md — guidance for Claude Code

Practice repo for an AI hackathon: **detect submarines / targets from sonar data.**
The task framing is unknown until the day, so this repo is built for *adaptability*.

## Architecture (respect this seam)
- `sonar_toolkit/` is the durable, reusable library. Keep it generic and tested.
  - `loaders/` — csv, wav, segmentation. The seam that absorbs new data formats.
  - `features/` — spectral (stft/mel/mfcc) + narrowband (LOFAR, DEMON).
  - `models/` — GBM baseline + 1D/2D CNN templates. Same fit/predict contract.
  - `validation/` — **group-aware** CV + detection metrics.
- `experiments/NN_name/` are thin: import the toolkit, wire a config, run.
  When adding a practice project, copy `01_uci_tabular/` as the template.

## Non-negotiable rules
1. **Never commit data.** Fetch via `scripts/download_data.py`. `data/` is gitignored.
2. **Group your CV folds by recording** whenever segments share a source, or your
   score is leakage. Pass `groups=` to `cross_validate`. UCI tabular is the only
   case where independent rows make plain stratified K-fold OK.
3. **Watch `recall@fpr`, not accuracy.** This is a detection problem.
4. New shared `sonar_toolkit/` code gets a quick teammate review before merge.

## Workflow
- Branch per task: `feat/<name>` or `exp/<name>`. Small PRs into `main`.
- `python scripts/download_data.py uci` then `python experiments/01_uci_tabular/run.py`
  to confirm the loop works.
- `pytest -q` before pushing.

## Practice roadmap (build these as experiments)
1. UCI tabular GBM baseline ✅ (done — the reference template)
2. ShipsEar/DeepShip audio → spectrogram → CNN (+ transfer learning)
3. Synthetic sonar simulator (tonals + broadband + Doppler + noise; active echoes)
4. Stress drills: imbalance/rare-target, domain shift, few-shot, streaming, localization
5. Timed full dress rehearsal
