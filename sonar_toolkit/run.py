"""Phase 2 — audio -> mel-spectrogram -> CNN, trained on GPU with group-aware CV.

A thin consumer of sonar_toolkit, mirroring 01_uci_tabular/run.py. It generates
labelled recordings with the synthetic simulator, windows them, extracts
mel-spectrograms, and runs a CNN (or transfer model) through group-by-recording
CV, scoring recall@fixed-FPR.

The lines tagged [VERIFY] are the *only* coupling to existing toolkit code —
confirm those names/signatures match your repo and adjust if they differ. Run with:
    python experiments/02_audio_cnn/run.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import GroupKFold

from sonar_toolkit.training.device import get_device, seed_everything, describe_device
from sonar_toolkit.training.loop import fit, TrainConfig
from sonar_toolkit.data.dataset import WindowDataset
from sonar_toolkit.augment.spec import SpecAugment, AddNoise, Compose
from sonar_toolkit.models.cnn import CNN2D                  # [VERIFY] exists in models/cnn.py
from sonar_toolkit.models.transfer import TransferClassifier
from sonar_toolkit import simulator                          # [VERIFY] Phase 3 simulator module
from sonar_toolkit.features import mel_spectrogram           # [VERIFY] name in features/
from sonar_toolkit.validation import recall_at_fpr           # [VERIFY] name in validation/

CONFIG = "experiments/02_audio_cnn/config.yaml"


def build_windows(cfg):
    """Generate recordings and slice into fixed windows -> waveforms, labels, groups."""
    waveforms, labels, groups = [], [], []
    win = int(cfg["window_s"] * cfg["sr"])
    hop = int(cfg["hop_s"] * cfg["sr"])
    for rec_id in range(cfg["n_recordings"]):
        # [VERIFY] simulator entry point — adjust to your simulator's actual signature.
        y, label = simulator.generate(scenario=cfg["scenario"], sr=cfg["sr"])
        for i in range(0, max(1, len(y) - win + 1), hop):
            waveforms.append(y[i:i + win])
            labels.append(int(label))
            groups.append(rec_id)          # group = recording, so CV can't leak across windows
    return waveforms, np.asarray(labels), np.asarray(groups)


def make_feature_fn(cfg):
    f = cfg["feature"]
    def fn(wav):
        # [VERIFY] mel_spectrogram signature — should return 2D [n_mels, time].
        return mel_spectrogram(wav, sr=cfg["sr"], n_fft=f["n_fft"],
                               hop_length=f["hop_length"], n_mels=f["n_mels"])
    return fn


def make_model(cfg):
    m = cfg["model"]
    if m["kind"] == "transfer":
        return TransferClassifier(n_classes=2, backbone=m["backbone"], pretrained=True)
    return CNN2D(n_classes=2, base=m["base"])


def main():
    cfg = yaml.safe_load(Path(CONFIG).read_text())
    seed_everything(42)
    device = get_device()
    print("device:", describe_device(device))

    waveforms, labels, groups = build_windows(cfg)
    feature_fn = make_feature_fn(cfg)
    cache_dir = cfg["paths"]["cache_dir"]
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    target_fpr = cfg["cv"]["target_fpr"]
    score_fn = lambda yt, ys: recall_at_fpr(yt, ys, fpr=target_fpr)   # [VERIFY] signature
    aug = Compose([SpecAugment(), AddNoise()])

    gkf = GroupKFold(n_splits=cfg["cv"]["n_folds"])
    idx = np.arange(len(labels))
    fold_scores = []
    for fold, (tr, va) in enumerate(gkf.split(idx, labels, groups)):
        train_ds = WindowDataset([waveforms[i] for i in tr], labels[tr], groups[tr],
                                 feature_fn=feature_fn, cache_dir=cache_dir,
                                 transform=aug, sr=cfg["sr"])
        val_ds = WindowDataset([waveforms[i] for i in va], labels[va], groups[va],
                               feature_fn=feature_fn, cache_dir=cache_dir,
                               transform=None, sr=cfg["sr"])

        pos = float(labels[tr].mean())            # class weights for rare-target imbalance
        weights = np.array([1.0 / (1 - pos + 1e-6), 1.0 / (pos + 1e-6)], dtype=np.float32)

        tcfg = TrainConfig(
            epochs=cfg["train"]["epochs"], batch_size=cfg["train"]["batch_size"],
            lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"],
            patience=cfg["train"]["patience"], use_amp=cfg["train"]["use_amp"],
            scheduler=cfg["train"]["scheduler"], ckpt_path=str(ckpt_dir / f"fold{fold}.pt"),
        )
        print(f"\n[fold {fold}]")
        out = fit(make_model(cfg), train_ds, val_ds, score_fn, cfg=tcfg, device=device,
                  class_weights=weights)
        print(f"[fold {fold}] recall@{target_fpr:.0%}FPR = {out['best_score']:.4f}")
        fold_scores.append(out["best_score"])

    print(f"\nCV recall@{target_fpr:.0%}FPR: "
          f"{np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")


if __name__ == "__main__":
    main()
