"""Standalone UAV air-to-ground fading-channel simulator.

A dependency-light stand-in for the Sionna 3GPP channel used in SemanticHDC
(communication.py). It reproduces the same observable quantities -- received
SNR (dB), channel power gain (dB), Doppler -- with realistic, non-Gaussian,
*drifting* statistics as the UAV flies a trajectory:

  * path loss     : free-space, PL = 20log10(d_3d) + 20log10(fc) - 147.55
                    (the exact formula in communication.py)
  * small-scale   : Rician fading (LoS-dominated for an elevated UAV),
                    |h|^2 normalized so E|h|^2 = 1 -> non-central chi-square
  * shadowing     : temporally-correlated log-normal (AR(1))
  * noise         : thermal kTB * noise-figure
  * mobility      : horizontal distance varies over time -> the SNR
                    distribution drifts (mean shifts, fading regime changes)

Swap this out for logged Sionna `metrics['snr_db']` when running on real data.
"""

import numpy as np

# Physical constants / defaults mirroring communication.py
K_BOLTZMANN = 1.38e-23
TEMP_K = 290.0


def _rician_power_gain(k_factor, size, rng):
    """|h|^2 for Rician fading, normalized so E[|h|^2] = 1."""
    s = np.sqrt(k_factor / (k_factor + 1.0))        # LoS component
    sigma = np.sqrt(1.0 / (2.0 * (k_factor + 1.0))) # scatter per real dim
    re = s + sigma * rng.standard_normal(size)
    im = sigma * rng.standard_normal(size)
    return re ** 2 + im ** 2


def simulate_channel_stream(
    n_steps=6000,
    carrier_frequency=2.4e9,
    drone_height_m=50.0,
    drone_velocity_ms=10.0,
    tx_power_dbm=20.0,
    noise_figure_db=6.0,
    bandwidth_hz=20e6,
    shadow_sigma_db=4.0,
    shadow_corr=0.99,
    seed=0,
):
    """Return a dict of per-step streams; `snr_db` is the primary signal."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)

    # UAV trajectory: fly out and back (patrol), distance 40 m -> 320 m -> 40 m,
    # plus a small random walk so it is not perfectly periodic.
    base = 180 + 140 * np.sin(2 * np.pi * t / n_steps - np.pi / 2)
    jitter = np.cumsum(rng.normal(0, 1.5, n_steps))
    horizontal_dist = np.clip(base + jitter, 40, 400)
    d_3d = np.sqrt(horizontal_dist ** 2 + drone_height_m ** 2)

    # Elevation-dependent Rician K-factor: steeper look angle -> stronger LoS.
    elevation = np.arcsin(drone_height_m / d_3d)            # radians
    k_factor = 2.0 + 12.0 * (elevation / (np.pi / 2))       # K in [2, 14]

    # Path loss (free space), matching communication.py
    path_loss_db = 20 * np.log10(d_3d) + 20 * np.log10(carrier_frequency) - 147.55

    # Temporally-correlated log-normal shadowing via AR(1)
    shadow = np.zeros(n_steps)
    for i in range(1, n_steps):
        shadow[i] = shadow_corr * shadow[i - 1] + \
            np.sqrt(1 - shadow_corr ** 2) * rng.normal(0, shadow_sigma_db)

    # Small-scale fading power gain (dB)
    gain_db = 10 * np.log10(_rician_power_gain(k_factor, n_steps, rng) + 1e-12)

    # Doppler from velocity (max Doppler shift)
    doppler_hz = drone_velocity_ms * carrier_frequency / 3e8 * np.ones(n_steps)

    # Received power and SNR
    rx_dbm = tx_power_dbm + gain_db - path_loss_db - shadow
    noise_dbm = 10 * np.log10(K_BOLTZMANN * TEMP_K * bandwidth_hz * 1000) + noise_figure_db
    snr_db = rx_dbm - noise_dbm

    return {
        "snr_db": snr_db,
        "channel_power_gain_db": gain_db,
        "path_loss_db": path_loss_db,
        "horizontal_dist_m": horizontal_dist,
        "doppler_hz": doppler_hz,
        "k_factor": k_factor,
    }


if __name__ == "__main__":
    s = simulate_channel_stream(n_steps=6000)
    snr = s["snr_db"]
    print(f"SNR(dB): mean={snr.mean():.1f} std={snr.std():.1f} "
          f"min={snr.min():.1f} max={snr.max():.1f}")
    print(f"distance(m): {s['horizontal_dist_m'].min():.0f} -> "
          f"{s['horizontal_dist_m'].max():.0f}")
