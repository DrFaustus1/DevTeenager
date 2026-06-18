import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED = Path(__file__).parent / "processed"
SRC  = PROCESSED / "panel.csv"
DEST = PROCESSED / "panel.csv"
BCK  = PROCESSED / "panel_original_backup.csv"


CLIP_RANGES = {
    "gini":                (0.28,  0.52),
    "decile_coef":         (4.0,   10.5),
    "wage_to_subsistence": (200,   750),
    "crime_murder":        (1,     600),
    "crime_hooliganism":   (0,     250),
    "crime_drugs":         (10,  15000),
    "alcohol_per100k":     (100,  5500),
    "narco_per100k":       (5,     700),
}

LATE_COLS = [
    "gini", "decile_coef", "wage_to_subsistence",
    "crime_murder", "crime_hooliganism", "crime_drugs",
]

INTERP_COLS = [
    "avg_income", "real_income_index", "poverty_rate",
    "unemployment_rate", "urbanization",
    "alcohol_per100k", "narco_per100k",
    "population_total", "pop_working_age_pct",
    "juvenile_crime_share",
]


def extrapolate_backwards(group: pd.DataFrame, col: str) -> pd.Series:
    s = group[col].copy()
    known = s.dropna()

    if known.empty:
        return s

    if len(known) == 1:
        s = s.fillna(known.iloc[0])
        return s

    x = known.index.get_level_values("year").astype(float).values
    y = known.values.astype(float)
    coeffs = np.polyfit(x, y, 1)  # [slope, intercept]

    for idx in s[s.isna()].index:
        yr = idx[1] if isinstance(idx, tuple) else group.loc[idx, "year"] if "year" in group.columns else idx
        s.loc[idx] = np.polyval(coeffs, float(yr))

    if col in CLIP_RANGES:
        lo, hi = CLIP_RANGES[col]
        s = s.clip(lower=lo, upper=hi)

    return s


def interpolate_col(group: pd.DataFrame, col: str) -> pd.Series:
    s = group[col].copy().astype(float)
    s = s.interpolate(method="linear", limit_direction="both")
    s = s.ffill().bfill()
    if col in CLIP_RANGES:
        lo, hi = CLIP_RANGES[col]
        s = s.clip(lower=lo, upper=hi)
    return s


def main():
    panel = pd.read_csv(SRC, sep=";", encoding="utf-8")
    print(f"Loaded: {len(panel)} rows x {panel.shape[1]} cols")

    if not BCK.exists():
        panel.to_csv(BCK, sep=";", index=False, encoding="utf-8")
        print(f"Backup saved: {BCK.name}")

    before = panel.isnull().sum()

    panel = panel.sort_values(["region", "year"]).reset_index(drop=True)
    panel = panel.set_index(["region", "year"])

    for col in INTERP_COLS:
        if col not in panel.columns:
            continue
        panel[col] = (
            panel.groupby(level="region")[col]
            .transform(lambda s: interpolate_col(s.to_frame(col), col) if s.isna().any() else s)
        )

    for col in LATE_COLS:
        if col not in panel.columns:
            continue

        def _extrap(grp, c=col):
            if grp[c].isna().all():
                return grp[c]
            s = grp[c].copy().astype(float)
            known = s.dropna()
            if len(known) < 2:
                s = s.fillna(known.iloc[0] if len(known) == 1 else np.nan)
                return s
            years_known = np.array([idx[1] for idx in known.index], dtype=float)
            vals_known  = known.values.astype(float)
            coeffs = np.polyfit(years_known, vals_known, 1)
            for idx in s[s.isna()].index:
                yr = float(idx[1])
                s.loc[idx] = np.polyval(coeffs, yr)
            if c in CLIP_RANGES:
                lo, hi = CLIP_RANGES[c]
                s = s.clip(lower=lo, upper=hi)
            return s

        panel[col] = panel.groupby(level="region").apply(
            lambda grp: _extrap(grp)
        ).droplevel(0)

    panel = panel.reset_index()

    numeric_cols = [c for c in INTERP_COLS + LATE_COLS if c in panel.columns]
    for col in numeric_cols:
        if panel[col].isna().any():
            year_median = panel.groupby("year")[col].transform("median")
            panel[col] = panel[col].fillna(year_median)

    after = panel.isnull().sum()
    print("\nMissing before / after:")
    for col in before.index:
        b, a = before[col], after.get(col, 0)
        if b > 0 or a > 0:
            print(f"  {col:<30} {b:>4} -> {a:>4}")

    panel.to_csv(DEST, sep=";", index=False, encoding="utf-8")
    print(f"\nSaved: {DEST}")


if __name__ == "__main__":
    main()
