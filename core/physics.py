"""
POSEIDON Physics-Guided Module
-------------------------------
Contains:
- Haversine distance calculation
- Magnitude-scaled spatiotemporal aftershock labeling
- Physics-guided history feature engineering (Omori-like + Utsu-like proxies)
"""

import numpy as np
import pandas as pd
from collections import deque

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TIME_COL = "time"
LAT_COL  = "latitude"
LON_COL  = "longitude"
MAG_COL  = "mag"
DEP_COL  = "depth"

SUMATRA_BBOX = {
    "lat_low": -7.5, "lat_high": 7.5,
    "lon_low": 92.0,  "lon_high": 107.0
}


# ─────────────────────────────────────────────
# HAVERSINE
# ─────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    R = 6371.0
    lat1, lon1 = np.deg2rad(lat1), np.deg2rad(lon1)
    lat2, lon2 = np.deg2rad(lat2), np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


# ─────────────────────────────────────────────
# MAGNITUDE-SCALED WINDOWS
# ─────────────────────────────────────────────
def R_of_M_km(M):
    """Rupture-scaling spatial window (km)."""
    return 10 ** (0.5 * M - 1.5)


def T_of_M_days(M, T_cap_days=None):
    """Rupture-scaling temporal window (days) with optional cap."""
    T = 10 ** (0.5 * M - 1.0)
    if T_cap_days is not None:
        return np.minimum(T, T_cap_days)
    return T


# ─────────────────────────────────────────────
# COLUMN DETECTION HELPER
# ─────────────────────────────────────────────
def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_columns(df):
    """Auto-detect key seismological columns."""
    return {
        "time":  pick_col(df, ["time", "timestamp", "date", "datetime"]),
        "lat":   pick_col(df, ["latitude", "lat"]),
        "lon":   pick_col(df, ["longitude", "lon", "lng"]),
        "mag":   pick_col(df, ["mag", "magnitude", "mw", "ml"]),
        "depth": pick_col(df, ["depth", "depth_km", "kedalaman"]),
    }


# ─────────────────────────────────────────────
# Mc ESTIMATION (MAXC + GFT)
# ─────────────────────────────────────────────
def maxc_mc(mags, bin_width=0.1):
    mags = mags[np.isfinite(mags)]
    m_min = np.floor(mags.min() / bin_width) * bin_width
    m_max = np.ceil(mags.max() / bin_width) * bin_width
    bins = np.arange(m_min, m_max + bin_width, bin_width)
    counts, _ = np.histogram(mags, bins=bins)
    centers = bins[:-1] + bin_width / 2
    return float(centers[int(np.argmax(counts))])


def b_value_mle(mags, mc, bin_width=0.1, min_n=50):
    mags_cut = mags[mags >= mc]
    if len(mags_cut) < min_n:
        return None
    mean_mag = np.mean(mags_cut)
    b = (np.log10(np.e)) / (mean_mag - (mc - bin_width / 2))
    return float(b)


def gft_score(mags, mc, bin_width=0.1, min_n=50):
    mags_cut = mags[mags >= mc]
    if len(mags_cut) < min_n:
        return None
    b = b_value_mle(mags, mc, bin_width, min_n)
    if b is None:
        return None
    m_vals = np.arange(mc, mags_cut.max() + 1e-9, bin_width)
    obs = np.array([np.sum(mags_cut >= m) for m in m_vals])
    if np.sum(obs) == 0:
        return None
    N0 = obs[0]
    pred = N0 * (10 ** (-b * (m_vals - mc)))
    R = 100.0 - (np.sum(np.abs(obs - pred)) / np.sum(obs)) * 100.0
    return float(R), float(b), int(len(mags_cut))


# ─────────────────────────────────────────────
# AFTERSHOCK LABELING
# ─────────────────────────────────────────────
def label_aftershocks_magnitude_scaled(df_input, MC_FINAL=4.45, T_cap_days=365):
    """
    Magnitude-Scaled Deterministic Spatiotemporal Labeling.
    Assigns is_aftershock=1 if an event falls within the spatiotemporal
    window of a prior mainshock candidate (M >= MC_FINAL).
    """
    df = df_input.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce")
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    needed = [TIME_COL, LAT_COL, LON_COL, MAG_COL]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df = df.dropna(subset=needed).reset_index(drop=True)

    t = df[TIME_COL].values.astype("datetime64[s]").astype(np.int64)
    lat = df[LAT_COL].to_numpy(dtype=float)
    lon = df[LON_COL].to_numpy(dtype=float)
    mag = df[MAG_COL].to_numpy(dtype=float)

    is_mainshock_candidate = mag >= MC_FINAL

    if np.any(is_mainshock_candidate):
        Tmax_global_days = float(np.max(T_of_M_days(mag[is_mainshock_candidate], T_cap_days)))
    else:
        Tmax_global_days = 0.0
    Tmax_global_seconds = Tmax_global_days * 86400.0

    n = len(df)
    is_aftershock = np.zeros(n, dtype=np.int8)
    mainshock_idx = np.full(n, -1, dtype=np.int32)
    dt_days_out   = np.full(n, np.nan, dtype=float)
    dr_km_out     = np.full(n, np.nan, dtype=float)
    R_km_out      = np.full(n, np.nan, dtype=float)
    T_days_out    = np.full(n, np.nan, dtype=float)

    candidates = deque()

    for i in range(n):
        ti = t[i]

        # Prune old candidates
        while candidates and (ti - t[candidates[0]]) > Tmax_global_seconds:
            candidates.popleft()

        # Check against all active candidates
        best_dt = np.inf
        best_j = -1
        for j in candidates:
            dt_sec = ti - t[j]
            dt_day = dt_sec / 86400.0

            T_j = float(T_of_M_days(mag[j], T_cap_days))
            if dt_day > T_j:
                continue

            R_j = float(R_of_M_km(mag[j]))
            dr = float(haversine_km(lat[j], lon[j], lat[i], lon[i]))
            if dr > R_j:
                continue

            # Inside window → pick closest in time
            if dt_day < best_dt:
                best_dt = dt_day
                best_j = j

        if best_j >= 0:
            is_aftershock[i] = 1
            mainshock_idx[i] = best_j
            dt_days_out[i] = best_dt
            dr_km_out[i] = float(haversine_km(lat[best_j], lon[best_j], lat[i], lon[i]))
            R_km_out[i] = float(R_of_M_km(mag[best_j]))
            T_days_out[i] = float(T_of_M_days(mag[best_j], T_cap_days))

        # Add as candidate if qualifies
        if mag[i] >= MC_FINAL:
            candidates.append(i)

    df["is_aftershock"] = is_aftershock
    df["mainshock_idx"] = mainshock_idx
    df["dt_days"]       = dt_days_out
    df["dr_km"]         = dr_km_out
    df["R_km"]          = R_km_out
    df["T_days"]        = T_days_out

    return df


# ─────────────────────────────────────────────
# FEATURE ENGINEERING (Physics-Guided History)
# ─────────────────────────────────────────────
def build_history_features_phys(
    df_input,
    mc_feature_big=5.0,
    near_radius_km=100.0,
    lookback_days_big=14.0,
    r_local_km=50.0,
    win_local_days=30.0,
    win_short_days=7.0,
):
    """
    Build physics-guided history features:
    - Omori-like: dt to nearest big event (temporal decay proxy)
    - Spatial: dr to nearest big event
    - Utsu-like productivity: activity counts in spatiotemporal windows
    - Short-term max magnitude
    """
    df = df_input.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce")
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    t = df[TIME_COL].values.astype("datetime64[s]").astype(np.int64)
    lat = df[LAT_COL].to_numpy(float)
    lon = df[LON_COL].to_numpy(float)
    mag = df[MAG_COL].to_numpy(float)

    n = len(df)

    dt_last_big_near_days = np.full(n, np.nan, float)
    dr_last_big_near_km   = np.full(n, np.nan, float)
    mag_last_big_near     = np.full(n, np.nan, float)

    n_prev_local     = np.zeros(n, int)
    n_prev_local_r   = np.zeros(n, int)
    n_big_prev_local = np.zeros(n, int)
    max_mag_prev_short = np.full(n, np.nan, float)

    q_local = deque()
    q_short = deque()
    q_short_max = deque()
    big_cand = deque()

    sec_local = win_local_days * 86400.0
    sec_short = win_short_days * 86400.0
    sec_big   = lookback_days_big * 86400.0

    for i in range(n):
        ti = t[i]

        while q_local and (ti - t[q_local[0]] > sec_local):
            q_local.popleft()

        while q_short and (ti - t[q_short[0]] > sec_short):
            out = q_short.popleft()
            if q_short_max and q_short_max[0] == out:
                q_short_max.popleft()

        while big_cand and (ti - t[big_cand[0]] > sec_big):
            big_cand.popleft()

        # ── FEATURES FROM PAST ONLY ──
        n_prev_local[i] = len(q_local)
        if q_local:
            idxL = np.array(q_local, dtype=int)
            n_big_prev_local[i] = int((mag[idxL] >= mc_feature_big).sum())
            drs = haversine_km(lat[i], lon[i], lat[idxL], lon[idxL])
            n_prev_local_r[i] = int((drs <= r_local_km).sum())

        if q_short_max:
            max_mag_prev_short[i] = mag[q_short_max[0]]

        if big_cand:
            idxB = np.array(big_cand, dtype=int)
            big_mask = mag[idxB] >= mc_feature_big
            if big_mask.any():
                idxBig = idxB[big_mask]
                drs_big = haversine_km(lat[i], lon[i], lat[idxBig], lon[idxBig])
                near_mask = drs_big <= near_radius_km
                if near_mask.any():
                    dts = (ti - t[idxBig[near_mask]]) / 86400.0
                    best = np.argmin(dts)
                    dt_last_big_near_days[i] = dts[best]
                    dr_last_big_near_km[i]   = drs_big[near_mask][best]
                    mag_last_big_near[i]     = mag[idxBig[near_mask][best]]

        # ── UPDATE QUEUES (add current) ──
        q_local.append(i)
        q_short.append(i)
        while q_short_max and mag[q_short_max[-1]] <= mag[i]:
            q_short_max.pop()
        q_short_max.append(i)
        if mag[i] >= mc_feature_big:
            big_cand.append(i)

    eps = 1e-6
    df["dt_last_big_near_days"] = dt_last_big_near_days
    df["dr_last_big_near_km"]   = dr_last_big_near_km
    df["mag_last_big_near"]     = mag_last_big_near
    df["n_prev_30d"]            = n_prev_local
    df["n_prev_30d_r50"]        = n_prev_local_r
    df["n_big_prev_30d"]        = n_big_prev_local
    df["max_mag_prev_7d"]       = max_mag_prev_short

    # Log transforms
    df["log_dt_big_near"]     = np.log1p(df["dt_last_big_near_days"].fillna(999))
    df["log_dr_big_near"]     = np.log1p(df["dr_last_big_near_km"].fillna(999))
    df["log_n_prev_30d"]      = np.log1p(df["n_prev_30d"])
    df["log_n_prev_30d_r50"]  = np.log1p(df["n_prev_30d_r50"])
    df["log_n_big_prev_30d"]  = np.log1p(df["n_big_prev_30d"])

    return df


# ─────────────────────────────────────────────
# FEATURE COLUMNS
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "mag",
    "log_dt_big_near",
    "log_dr_big_near",
    "log_n_prev_30d",
    "log_n_prev_30d_r50",
    "log_n_big_prev_30d",
    "max_mag_prev_7d",
]

FEATURE_PRETTY = {
    "mag": "Magnitude (M)",
    "log_dt_big_near": "log Δt to nearest large event",
    "log_dr_big_near": "log Δr to nearest large event",
    "log_n_prev_30d": "log count prev. 30 days",
    "log_n_prev_30d_r50": "log count prev. 30d (r≤50 km)",
    "log_n_big_prev_30d": "log count large events prev. 30d",
    "max_mag_prev_7d": "Max magnitude prev. 7 days",
}

LABEL_COL = "is_aftershock"
