"""Synthetic sonar simulator — the engine for practicing 'differing initial conditions'.

It manufactures labelled sonar data on demand so you can rehearse any task framing
before the real one drops. Two physics-motivated generators:

  passive: a target radiates narrowband *tonals* (rotating machinery) plus broadband
           *cavitation* noise amplitude-modulated at the propeller *shaft rate*
           (what DEMON recovers), buried in colored ambient noise at a chosen SNR.
  active:  you transmit a ping; a target returns a delayed echo (target strength)
           amid reverberation and random clutter echoes.

The key object is `Scenario`: change its knobs (SNR, prevalence, ambient, confusers,
doppler, mode) and you get a different practice condition. `generate_dataset` emits
multiple segments per simulated *recording* and tags them with a shared group id, so
the group-aware CV harness is exercised exactly as it must be on real audio.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt, chirp

C_WATER = 1500.0  # speed of sound in seawater, m/s


# --------------------------------------------------------------------------- #
# Vessel profiles: a class is defined by its acoustic signature.
# --------------------------------------------------------------------------- #
@dataclass
class VesselProfile:
    name: str
    tonals: list[tuple[float, float]]   # (frequency_hz, relative_amplitude)
    shaft_rate: float                   # propeller rotations per second
    blade_count: int                    # blades -> blade rate = shaft_rate * blades
    broadband: float                    # cavitation broadband level (relative)


SUBMARINE = VesselProfile(
    name="submarine",
    tonals=[(48, 1.0), (96, 0.5), (144, 0.25), (210, 0.15)],
    shaft_rate=1.8, blade_count=7, broadband=0.25,   # slow, quiet, many blades
)
SURFACE_SHIP = VesselProfile(
    name="surface_ship",
    tonals=[(72, 1.0), (120, 0.7), (240, 0.4)],
    shaft_rate=4.2, blade_count=4, broadband=1.0,    # fast, loud — a confuser
)
FISHING_BOAT = VesselProfile(
    name="fishing_boat",
    tonals=[(60, 0.8), (180, 0.5)],
    shaft_rate=6.0, blade_count=3, broadband=0.7,
)


# --------------------------------------------------------------------------- #
# Noise
# --------------------------------------------------------------------------- #
def colored_noise(n: int, color: str, rng: np.random.Generator) -> np.ndarray:
    """white / pink (1/f) / shipping (steep low-freq emphasis)."""
    white = rng.standard_normal(n)
    if color == "white":
        return white
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n)
    f[0] = f[1] if len(f) > 1 else 1.0
    if color == "pink":
        spec /= np.sqrt(f)
    elif color == "shipping":
        spec /= f                      # ~brown: heavy low-frequency shipping rumble
    out = np.fft.irfft(spec, n=n)
    return out / (out.std() + 1e-12)


def _bandlimit(x, sr, lo, hi, rng=None):
    nyq = sr / 2
    lo = max(lo, 1.0) / nyq
    hi = min(hi, nyq * 0.99) / nyq
    sos = butter(4, [lo, hi], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def _scale_to_snr(target, noise, snr_db):
    """Scale target so target_power / noise_power == 10^(snr_db/10)."""
    tp = np.mean(target ** 2) + 1e-12
    npow = np.mean(noise ** 2) + 1e-12
    desired = 10 ** (snr_db / 10)
    return target * np.sqrt(desired * npow / tp)


# --------------------------------------------------------------------------- #
# Passive signature
# --------------------------------------------------------------------------- #
def synth_passive(profile: VesselProfile, sr: int, duration: float, snr_db: float,
                  ambient: str, rng: np.random.Generator,
                  doppler_velocity: float = 0.0, freq_jitter: float = 1.0,
                  tonal_gain: float = 1.0) -> np.ndarray:
    n = int(sr * duration)
    t = np.arange(n) / sr
    # freq_jitter spreads each recording's lines so classes overlap in frequency
    # (otherwise fixed tonal bins make the classes trivially separable).
    shift = (C_WATER / (C_WATER - doppler_velocity)) * freq_jitter

    # Narrowband tonals (Doppler + jitter), with random drift for realism.
    # tonal_gain < 1 suppresses the (easy) tonals so detection must lean on the
    # (hard) broadband cavitation signature -- the main difficulty knob.
    sig = np.zeros(n)
    for f0, amp in profile.tonals:
        drift = 1 + 0.0008 * np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 6.28))
        sig += tonal_gain * amp * np.sin(2 * np.pi * f0 * shift * drift * t)

    # Broadband cavitation in a high band, amplitude-modulated at the shaft rate
    # (this modulation is exactly what DEMON analysis demodulates back out).
    cav = _bandlimit(rng.standard_normal(n), sr, 0.25 * sr / 2, 0.45 * sr / 2)
    shaft = profile.shaft_rate * shift
    mod = 1.0
    for h in range(1, profile.blade_count + 1):
        amp = 0.6 if h == profile.blade_count else 0.3 / h   # blade rate emphasized
        mod = mod + amp * np.cos(2 * np.pi * shaft * h * t)
    sig = sig + profile.broadband * cav * np.clip(mod, 0, None)

    noise = colored_noise(n, ambient, rng)
    sig = _scale_to_snr(sig, noise, snr_db)
    out = sig + noise
    return out / (np.abs(out).max() + 1e-12)


# --------------------------------------------------------------------------- #
# Active signature
# --------------------------------------------------------------------------- #
def synth_active(sr: int, duration: float, snr_db: float, rng: np.random.Generator,
                 target_present: bool, target_range: float = 600.0,
                 n_clutter: int = 6, reverb_tau: float = 0.4) -> np.ndarray:
    n = int(sr * duration)
    ping_len = int(0.02 * sr)
    tp = np.arange(ping_len) / sr
    ping = chirp(tp, f0=0.1 * sr / 2, f1=0.35 * sr / 2, t1=tp[-1], method="linear")
    ping *= np.hanning(ping_len)

    echoes = np.zeros(n)

    def place(delay_samp, amp):
        i = int(delay_samp)
        if 0 <= i < n - ping_len:
            echoes[i:i + ping_len] += amp * ping

    # Reverberation: many weak, exponentially-decaying random returns.
    rev = colored_noise(n, "pink", rng) * np.exp(-np.arange(n) / (reverb_tau * sr))
    echoes += 0.3 * rev
    # Clutter: random false targets.
    for _ in range(n_clutter):
        place(rng.integers(ping_len, n - ping_len), rng.uniform(0.1, 0.4))
    # The real target echo, if present.
    if target_present:
        delay = 2 * target_range / C_WATER * sr     # two-way travel time
        place(delay, 1.0)

    noise = colored_noise(n, "white", rng)
    echoes = _scale_to_snr(echoes, noise, snr_db)
    out = echoes + noise
    return out / (np.abs(out).max() + 1e-12)


# --------------------------------------------------------------------------- #
# Scenario: the bundle of knobs that defines a practice condition.
# --------------------------------------------------------------------------- #
@dataclass
class Scenario:
    name: str = "baseline"
    mode: str = "passive"                       # 'passive' | 'active'
    sr: int = 16000
    duration: float = 3.0
    n_recordings: int = 60
    segments_per_recording: int = 4
    snr_db_range: tuple[float, float] = (5.0, 15.0)
    target_prevalence: float = 0.5             # fraction of recordings with a target
    ambient: str = "white"                      # 'white' | 'pink' | 'shipping'
    confusers: list[VesselProfile] = field(default_factory=list)
    doppler: bool = False
    doppler_velocity_range: tuple[float, float] = (-8.0, 8.0)
    freq_jitter: float = 0.0                     # +/- fractional tonal-frequency spread
    tonal_gain: float = 1.0                      # <1 suppresses tonals -> harder
    target: VesselProfile = field(default_factory=lambda: SUBMARINE)


def generate_dataset(scn: Scenario, seed: int = 0) -> dict:
    """Return {waveforms, labels, groups, sr, scenario}.

    Each recording shares one group id across its segments, so segments from the
    same recording never leak across CV folds.
    """
    rng = np.random.default_rng(seed)
    waveforms, labels, groups = [], [], []

    for rec in range(scn.n_recordings):
        is_target = rng.random() < scn.target_prevalence
        snr = rng.uniform(*scn.snr_db_range)
        vel = rng.uniform(*scn.doppler_velocity_range) if scn.doppler else 0.0
        confuser = rng.choice(scn.confusers) if scn.confusers else None
        # per-recording frequency jitter: segments share it, classes overlap.
        jitter = rng.uniform(1 - scn.freq_jitter, 1 + scn.freq_jitter)

        for _ in range(scn.segments_per_recording):
            if scn.mode == "active":
                w = synth_active(scn.sr, scn.duration, snr, rng, target_present=is_target)
            elif is_target:
                w = synth_passive(scn.target, scn.sr, scn.duration, snr,
                                  scn.ambient, rng, doppler_velocity=vel,
                                  freq_jitter=jitter, tonal_gain=scn.tonal_gain)
            elif confuser is not None:
                # negative class still contains a (non-target) vessel: harder.
                w = synth_passive(confuser, scn.sr, scn.duration, snr,
                                  scn.ambient, rng, freq_jitter=jitter,
                                  tonal_gain=scn.tonal_gain)
            else:
                noise = colored_noise(int(scn.sr * scn.duration), scn.ambient, rng)
                w = noise / (np.abs(noise).max() + 1e-12)
            waveforms.append(w)
            labels.append(int(is_target))
            groups.append(rec)

    return {
        "waveforms": waveforms,
        "labels": np.array(labels),
        "groups": np.array(groups),
        "sr": scn.sr,
        "scenario": scn.name,
    }


# --------------------------------------------------------------------------- #
# Preset scenarios = ready-made "differing initial conditions" to practice on.
# --------------------------------------------------------------------------- #
SCENARIOS = {
    # easy: strong tonals, mild noise
    "baseline": Scenario(name="baseline", snr_db_range=(-8.0, 2.0), freq_jitter=0.05),
    # imbalance drill: target is rare
    "rare_target": Scenario(name="rare_target", target_prevalence=0.12,
                            n_recordings=90, snr_db_range=(-12.0, -2.0),
                            freq_jitter=0.08),
    # hard: quiet sub (tonals suppressed -> broadband-only), loud shipping, confusers
    "quiet_sub_loud_shipping": Scenario(
        name="quiet_sub_loud_shipping", snr_db_range=(-22.0, -12.0),
        ambient="shipping", confusers=[SURFACE_SHIP, FISHING_BOAT],
        freq_jitter=0.25, tonal_gain=0.12, duration=1.5, n_recordings=80),
    # moving targets: large Doppler shifts
    "doppler": Scenario(name="doppler", doppler=True, snr_db_range=(-14.0, -4.0),
                        doppler_velocity_range=(-30.0, 30.0), freq_jitter=0.05),
    # active sonar: echo in reverberation + clutter
    "active_clutter": Scenario(name="active_clutter", mode="active",
                               snr_db_range=(-8.0, 0.0)),
}
