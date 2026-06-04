# Sonar Hackathon Toolkit

Practice repo for an AI hackathon: **detect submarines / targets from sonar data.**

The competition task is announced on the day, so this repo optimises for *adaptability*:
a reusable toolkit (`sonar_toolkit/`) plus thin, swappable practice experiments. When the
real task drops, you map it onto a few axes (modality, active/passive, task type, labels,
conditions, metric) and re-point the pipeline — you don't start from scratch.

## Quickstart

```bash
git clone <your-repo-url> && cd sonar-hackathon
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_data.py uci                 # fetch the tiny practice set
python experiments/01_uci_tabular/run.py            # run the end-to-end baseline
pytest -q                                           # sanity-check the toolkit
```

## Layout

```
sonar_toolkit/        # the reusable asset — keep generic & tested
  loaders/   features/   models/   validation/   augment/
experiments/          # thin consumers of the toolkit; copy 01_ as a template
scripts/              # data download etc.
tests/
CLAUDE.md             # conventions for Claude Code sessions
```

## Team workflow

- **Data is never committed.** Everyone runs `scripts/download_data.py`. `data/` is gitignored.
  Share *synthetic* data later via a Drive folder (graduate to DVC only if versioning hurts).
- **Branch per task** (`feat/…`, `exp/…`), small PRs into a protected `main`.
- **Shared `sonar_toolkit/` changes get a quick review** — it's infrastructure everyone depends on.
- Split the roadmap phases (in `CLAUDE.md`) across teammates; each plugs into the same harness.

## Two rules that win hackathons

1. **Trustworthy validation.** Group CV folds by recording when segments share a source, or
   your score is fiction. Don't overfit the public leaderboard.
2. **The metric is detection, not accuracy.** Track `recall@fixed-FPR`; watch false alarms.
