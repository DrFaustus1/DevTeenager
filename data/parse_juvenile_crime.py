import pandas as pd
from pathlib import Path

RAW = Path(__file__).parent / "raw"
OUT = Path(__file__).parent / "processed"

XLS        = RAW / "juvenile_crime.xls"
PANEL_CSV  = OUT / "panel.csv"
OUT_CSV    = OUT / "juvenile_crime_parsed.csv"

REGION_NAME_MAP = {
    "г. Москва":             "Москва",
    "г. Санкт-Петербург":    "Санкт-Петербург",
    "г. Севастополь":        "Севастополь",
    "Республика Саха /Якутия/":          "Республика Саха (Якутия)",
    "Ханты-Мансийский автономный округ - Югра":
        "Ханты-Мансийский автономный округ — Югра",
    "Кемеровская область - Кузбасс":     "Кемеровская область",
    "Архангельская область (без а/о)":   "Архангельская область (без автономного округа)",
    "Архангельская область (с а/о )":    "Архангельская область (с автономным округом)",
    "Тюменская область (без а/о)":       "Тюменская область (без автономных округов)",
    "Тюменская область (с а/о)":         "Тюменская область (с автономными округами)",
    "Новгородская  область":             "Новгородская область",
}


def find_annual_columns(raw: pd.DataFrame) -> dict[int, int]:
    year_row   = raw.iloc[3]
    period_row = raw.iloc[4]

    annual_cols: dict[int, int] = {}
    current_year: int | None = None

    for col_idx in range(2, len(raw.columns)):
        yr = year_row.iloc[col_idx]
        if pd.notna(yr):
            try:
                current_year = int(float(str(yr)))
            except (ValueError, TypeError):
                pass

        period = str(period_row.iloc[col_idx]).strip()
        if period == "январь-декабрь" and current_year is not None:
            annual_cols[current_year] = col_idx

    return annual_cols


def parse_juvenile_crime() -> pd.DataFrame:
    raw = pd.read_excel(XLS, sheet_name="Данные", header=None, engine="xlrd")

    annual_cols = find_annual_columns(raw)
    print(f"Годовые столбцы: {sorted(annual_cols.keys())}")

    data = raw.iloc[5:].copy()
    data = data[data.iloc[:, 1].astype(str).str.strip() == "процент"].copy()

    skip_kw = ["федеральный округ", "российская федерация"]
    data = data[
        ~data.iloc[:, 0].astype(str).apply(
            lambda x: any(kw in x.lower() for kw in skip_kw)
        )
    ].copy()

    records = []
    for _, row in data.iterrows():
        raw_name = " ".join(str(row.iloc[0]).split()) 
        raw_name = raw_name.strip()
        region   = REGION_NAME_MAP.get(raw_name, raw_name)
        for year, col_idx in annual_cols.items():
            val = row.iloc[col_idx]
            if pd.notna(val):
                records.append({
                    "region":                region,
                    "year":                  int(year),
                    "juvenile_crime_share":  float(val),
                })

    result = (
        pd.DataFrame(records)
        .sort_values(["region", "year"])
        .reset_index(drop=True)
    )

    OUT.mkdir(exist_ok=True)
    result.to_csv(OUT_CSV, sep=";", index=False, encoding="utf-8")
    print(f"Сохранено {len(result)} записей → {OUT_CSV}")
    print(f"Регионов: {result['region'].nunique()}, "
          f"лет: {result['year'].min()}–{result['year'].max()}")

    if PANEL_CSV.exists():
        panel = pd.read_csv(PANEL_CSV, sep=";", encoding="utf-8")
        panel_regions = set(panel["region"].unique())
        xls_regions   = set(result["region"].unique())

        missing_in_xls   = panel_regions - xls_regions
        missing_in_panel = xls_regions - panel_regions

        if missing_in_xls:
            print(f"\n[!] Регионы из panel.csv, НЕ найденные в XLS ({len(missing_in_xls)}):")
            for r in sorted(missing_in_xls):
                print(f"    {r!r}")
        if missing_in_panel:
            print(f"\n[!] Регионы из XLS, НЕ найденные в panel.csv ({len(missing_in_panel)}):")
            for r in sorted(missing_in_panel):
                print(f"    {r!r}")
        if not missing_in_xls and not missing_in_panel:
            print("\n[OK] Все регионы совпали!")

    return result


if __name__ == "__main__":
    parse_juvenile_crime()
