"""Fetch public practice datasets into ./data (gitignored).

Usage:
    python scripts/download_data.py uci

Add ShipsEar / DeepShip fetchers here as you reach Phase 2. Those require
registration, so this script will print instructions rather than auto-download.
"""
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# The official UCI archive often 403s on scripted downloads, so we try a few
# stable mirrors of the classic headerless format (60 features + R/M label).
UCI_MIRRORS = [
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "undocumented/connectionist-bench/sonar/sonar.all-data",
]


def uci():
    DATA.mkdir(exist_ok=True)
    dest = DATA / "sonar.all-data"
    if dest.exists():
        print(f"already present: {dest}")
        return
    req = lambda u: urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
    for url in UCI_MIRRORS:
        try:
            print(f"trying {url}")
            with urllib.request.urlopen(req(url)) as r:
                dest.write_bytes(r.read())
            print("done. 208 rows, 60 features, labels R (rock) / M (mine/metal).")
            return
        except Exception as e:  # noqa
            print(f"  failed: {e}")
    raise SystemExit("all mirrors failed — check your network settings/allowlist")


def shipsear():
    print("ShipsEar requires registration: https://underwaternoise.atlanta.ovh/")
    print("DeepShip: https://github.com/irfankamboh/DeepShip")
    print("Download manually into ./data/ then point an experiment config at it.")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "uci"
    {"uci": uci, "shipsear": shipsear, "deepship": shipsear}.get(which, uci)()
