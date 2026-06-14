"""Phase 4 — audio -> mel-spectrogram -> Audio Spectrogram Transformer (AST).

Thin experiment script following the 01/02 template. Imports a reusable AST
wrapper from sonar_toolkit and wires it to the existing training loop, feature
pipeline, and group-aware CV infrastructure.

Run:
    python experiments/04_audio_ast/run.py
    python experiments/04_audio_ast/run.py --config experiments/04_audio_ast/config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import GroupKFold

from sonar_toolkit.training.device import get_device, seed_everything, describe_device
from sonar_toolkit.training.loop import fit, TrainConfig
from sonar_toolkit.data.dataset import WindowDataset
from sonar_toolkit.augment.spec import SpecAugment, AddNoise, Compose
from sonar_toolkit.features import mel_spectrogram
from sonar_toolkit.validation import recall_at_fpr

from sonar_toolkit.models.ast import ASTClassifier

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


# ---- data loading ---------------------------------------------------------

def _load_audio_files(data_dir: Path):
    """Load real audio files from a labelled directory structure.

    Expected layout:
        data_dir/
            positive/   *.wav
            negative/   *.wav
    Returns (waveform_paths, labels, groups) where each file is one group.
    """
    paths, labels, groups = [], [], []
    group_id = 0
    for label_name, label_val in [("negative", 0), ("positive", 1)]:
        folder = data_dir / label_name
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Expected {folder} to exist. Organise audio as "
                f"{data_dir}/positive/ and {data_dir}/negative/."
            )
        for fp in sorted(folder.glob("*.wav")):
            paths.append(fp)
            labels.append(label_val)
            groups.append(group_id)
            group_id += 1
    if not paths:
        raise FileNotFoundError(f"No .wav files found under {data_dir}")
    return paths, np.array(labels), np.array(groups)


def _generate_synthetic(cfg: dict):
    """Generate synthetic recordings via the sonar_toolkit simulator."""
    from sonar_toolkit import simulator

    waveforms, labels, groups = [], [], []
    sr = cfg["sr"]
    win = int(cfg["window_s"] * sr)
    hop = int(cfg["hop_s"] * sr)
    for rec_id in range(cfg["n_recordings"]):
        y, label = simulator.generate(scenario=cfg["scenario"], sr=sr)
        for i in range(0, max(1, len(y) - win + 1), hop):
            waveforms.append(y[i : i + win])
            labels.append(int(label))
            groups.append(rec_id)
    return waveforms, np.asarray(labels), np.asarray(groups)


def load_data(cfg: dict):
    """Return (waveforms_or_paths, labels, groups) from config."""
    if cfg.get("use_simulator", False):
        return _generate_synthetic(cfg)

    data_dir = ROOT / cfg["data_dir"]
    paths, labels, groups = _load_audio_files(data_dir)

    # Window long files into fixed-length segments.
    sr = cfg["sr"]
    win = int(cfg["window_s"] * sr)
    hop = int(cfg["hop_s"] * sr)
    import librosa

    waveforms, win_labels, win_groups = [], [], []
    for fp, lab, grp in zip(paths, labels, groups):
        y, _ = librosa.load(str(fp), sr=sr, mono=True)
        if len(y) < win:
            # Pad short files to window length.
            y = np.pad(y, (0, win - len(y)))
        for i in range(0, max(1, len(y) - win + 1), hop):
            waveforms.append(y[i : i + win].astype(np.float32))
            win_labels.append(lab)
            win_groups.append(grp)
    return waveforms, np.asarray(win_labels), np.asarray(win_groups)


# ---- feature extraction ---------------------------------------------------

def make_feature_fn(cfg: dict):
    """Return a callable: waveform -> mel spectrogram [n_mels, time]."""
    f = cfg["feature"]
    sr = cfg["sr"]

    def extract(wav: np.ndarray) -> np.ndarray:
        return mel_spectrogram(
            wav, sr=sr, n_mels=f["n_mels"], n_fft=f["n_fft"], hop=f["hop_length"],
        )

    return extract


# ---- model factory --------------------------------------------------------

def make_model(cfg: dict):
    """Build the AST model from config."""
    m = cfg["model"]
    return ASTClassifier(
        n_classes=2,
        pretrained=m.get("pretrained", True),
        target_length=m.get("target_length", 200),
        n_mels=cfg["feature"]["n_mels"],
    )


# ---- backbone freeze/unfreeze schedule ------------------------------------

def _freeze_backbone(model):
    """Freeze all parameters except the classification head."""
    for name, p in model.named_parameters():
        if "head" not in name and "classifier" not in name and "fc" not in name:
            p.requires_grad = False


def _unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


# ---- main loop ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AST experiment")
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG), help="YAML config path",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed_everything(42)
    device = get_device()
    print(f"device: {describe_device(device)}")

    # ---- data ----
    waveforms, labels, groups = load_data(cfg)
    print(f"windows: {len(labels)}, positives: {int(labels.sum())}/{len(labels)}")

    feature_fn = make_feature_fn(cfg)
    cache_dir = cfg["paths"]["cache_dir"]
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- augmentation ----
    aug = Compose([
        SpecAugment(freq_mask=16, time_mask=24, n_freq=2, n_time=2),
        AddNoise(snr_db_range=(5, 25)),
    ])

    # ---- CV ----
    target_fpr = cfg["cv"]["target_fpr"]
    score_fn = lambda yt, ys: recall_at_fpr(yt, ys, max_fpr=target_fpr)

    n_folds = cfg["cv"]["n_folds"]
    gkf = GroupKFold(n_splits=n_folds)
    idx = np.arange(len(labels))
    fold_scores: list[float] = []

    freeze_epochs = cfg["model"].get("freeze_backbone_epochs", 0)

    for fold, (tr, va) in enumerate(gkf.split(idx, labels, groups)):
        print(f"\n{'='*40} fold {fold} {'='*40}")

        train_ds = WindowDataset(
            [waveforms[i] for i in tr], labels[tr], groups[tr],
            feature_fn=feature_fn, cache_dir=cache_dir,
            transform=aug, sr=cfg["sr"],
        )
        val_ds = WindowDataset(
            [waveforms[i] for i in va], labels[va], groups[va],
            feature_fn=feature_fn, cache_dir=cache_dir,
            transform=None, sr=cfg["sr"],
        )

        model = make_model(cfg)

        # Class weighting for imbalance.
        pos = float(labels[tr].mean())
        weights = np.array(
            [1.0 / (1 - pos + 1e-6), 1.0 / (pos + 1e-6)], dtype=np.float32,
        )

        # --- Phase 1: frozen backbone (optional) ---
        if freeze_epochs > 0:
            print(f"  frozen backbone for {freeze_epochs} epochs")
            _freeze_backbone(model)
            frozen_cfg = TrainConfig(
                epochs=freeze_epochs,
                batch_size=cfg["train"]["batch_size"],
                lr=cfg["train"]["lr"] * 10,  # higher LR for head-only warmup
                weight_decay=cfg["train"]["weight_decay"],
                patience=freeze_epochs,       # no early stop during warmup
                use_amp=cfg["train"]["use_amp"],
                scheduler="none",
                grad_clip=cfg["train"].get("grad_clip", 1.0),
                ckpt_path=str(ckpt_dir / f"fold{fold}_frozen.pt"),
            )
            fit(
                model, train_ds, val_ds, score_fn,
                cfg=frozen_cfg, device=device, class_weights=weights,
            )
            _unfreeze_all(model)

        # --- Phase 2: full fine-tuning ---
        remaining = cfg["train"]["epochs"] - freeze_epochs
        tcfg = TrainConfig(
            epochs=max(remaining, 1),
            batch_size=cfg["train"]["batch_size"],
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
            patience=cfg["train"]["patience"],
            use_amp=cfg["train"]["use_amp"],
            scheduler=cfg["train"]["scheduler"],
            grad_clip=cfg["train"].get("grad_clip", 1.0),
            ckpt_path=str(ckpt_dir / f"fold{fold}.pt"),
        )
        out = fit(
            model, train_ds, val_ds, score_fn,
            cfg=tcfg, device=device, class_weights=weights,
        )

        print(f"  [fold {fold}] recall@{target_fpr:.0%}FPR = {out['best_score']:.4f}")
        fold_scores.append(out["best_score"])

    # ---- summary ----
    print(f"\n{'='*60}")
    print(
        f"CV recall@{target_fpr:.0%}FPR: "
        f"{np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}"
    )
    for i, s in enumerate(fold_scores):
        print(f"  fold {i}: {s:.4f}")


if __name__ == "__main__":
    main()
