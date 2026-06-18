import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

RAW  = Path(__file__).parent / "raw"
OUT  = Path(__file__).parent / "processed"
OUT.mkdir(exist_ok=True)

PARQUET = RAW / "data_regions_collection_102_v20260313.parquet"

INDICATORS = {
    "avg_income":          ("Y477110374", None),
    "real_income_index":   ("Y477110363", None),
    "poverty_rate":        ("Y477110463", None),
    "unemployment_rate":   ("Y477100008", None),
    "urbanization":        ("Y477110408", None),
    "grp_per_capita":      ("Y477110006", None),
    "investment_raw":      ("Y477050015", None),
    "gini":                ("Y477130031", "Коэффициент Джини"),
    "decile_coef":         ("Y477130031", "Децильный коэффициент, раз"),
    "wage_to_subsistence": ("Y477130043", None),
    "alcohol_per100k":     ("Y477040014", "На 100 000 человек населения"),
    "narco_per100k":       ("Y477040016", "На 100 000 человек населения"),
    "population_total":    ("Y477110373", None),
    "pop_working_age_pct": ("Y477110050", None),
    "crime_murder":        ("Y477130012", "Убийство или покушение на убийство"),
    "crime_hooliganism":   ("Y477130012", "Хулиганство"),
    "crime_drugs":         ("Y477130012", "Преступления, связанные с незаконным оборотом наркотиков"),
}

EXCLUDED_REGIONS = {
    "Республика Дагестан",
    "Республика Ингушетия",
    "Кабардино-Балкарская Республика",
    "Карачаево-Черкесская Республика",
    "Республика Северная Осетия — Алания",
    "Чеченская Республика",
}

YEAR_FROM = 2010
YEAR_TO   = 2024


def load_indicator(df: pd.DataFrame, code: str, subsec: str | None) -> pd.DataFrame:
    mask = df["indicator_code"] == code
    if subsec:
        mask &= df["subsection"] == subsec
    sub = df.loc[mask, ["object_name", "year", "indicator_value"]].copy()
    sub = sub.rename(columns={"object_name": "region", "indicator_value": "value"})
    sub = sub.drop_duplicates(subset=["region", "year"], keep="first")
    return sub.reset_index(drop=True)


def build_panel() -> pd.DataFrame:
    print(f"Загрузка {PARQUET.name} …")
    df = pd.read_parquet(PARQUET)

    df = df[df["object_level"] == "Регион"].copy()
    df = df[(df["year"] >= YEAR_FROM) & (df["year"] <= YEAR_TO)]
    df = df[~df["object_name"].isin(EXCLUDED_REGIONS)]

    n_regions_expected = df["object_name"].nunique()
    print(f"Регионов в выборке: {n_regions_expected}")

    panel = None
    for col, (code, subsec) in INDICATORS.items():
        ind = load_indicator(df, code, subsec)
        if ind.empty:
            print(f"  [!] {col} — не найден (code={code}, subsec={subsec})")
            continue
        ind = ind.rename(columns={"value": col})
        if panel is None:
            panel = ind
        else:
            panel = panel.merge(ind, on=["region", "year"], how="outer")
        years_in = sorted(ind["year"].unique())
        print(f"  [+] {col:<28} {years_in[0]}–{years_in[-1]}, n={len(ind)}")

    panel = panel.sort_values(["region", "year"]).reset_index(drop=True)

    numeric_cols = panel.select_dtypes(include="number").columns
    panel[numeric_cols] = panel[numeric_cols].where(panel[numeric_cols] >= 0)

    panel["investment_pc"] = (panel["investment_raw"] * 1000) / panel["population_total"]
    panel = panel.drop(columns=["investment_raw"])
    panel["is_excluded"] = False

    jc_parsed = OUT / "juvenile_crime_parsed.csv"
    xls_src    = RAW / "juvenile_crime.xls"

    if xls_src.exists():
        from parse_juvenile_crime import parse_juvenile_crime
        print("\nПарсинг juvenile_crime.xls …")
        jc = parse_juvenile_crime()
        jc = jc[(jc["year"] >= YEAR_FROM) & (jc["year"] <= YEAR_TO)]
        panel = panel.merge(jc, on=["region", "year"], how="left")
        filled = panel["juvenile_crime_share"].notna().sum()
        print(f"juvenile_crime_share заполнено: {filled}/{len(panel)} строк")
    elif jc_parsed.exists():
        jc = pd.read_csv(jc_parsed, sep=";", encoding="utf-8")
        jc = jc[(jc["year"] >= YEAR_FROM) & (jc["year"] <= YEAR_TO)]
        panel = panel.merge(jc, on=["region", "year"], how="left")
        filled = panel["juvenile_crime_share"].notna().sum()
        print(f"juvenile_crime_share заполнено из кэша: {filled}/{len(panel)} строк")
    else:
        panel["juvenile_crime_share"] = float("nan")
        print("juvenile_crime_share: XLS не найден, поле пустое")

    out_path = OUT / "panel.csv"
    panel.to_csv(out_path, sep=";", index=False, encoding="utf-8")
    print(f"\nГотово: {len(panel)} строк × {panel.shape[1]} столбцов")
    print(f"Сохранено в {out_path}")
    print(f"\nПропуски по столбцам:")
    print(panel.isnull().sum().to_string())
    return panel


if __name__ == "__main__":
    build_panel()
