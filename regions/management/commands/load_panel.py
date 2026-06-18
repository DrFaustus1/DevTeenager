"""
Импортирует data/processed/panel.csv в таблицу PanelObservation.

Запуск:
    python manage.py load_panel
    python manage.py load_panel --csv path/to/other.csv
    python manage.py load_panel --clear   # удалить старые наблюдения перед загрузкой
"""
import math
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from regions.models import PanelObservation, Region

CSV_TO_FIELD = {
    "avg_income":          "avg_income",
    "real_income_index":   "real_income_index",
    "poverty_rate":        "poverty_rate",
    "unemployment_rate":   "unemployment_rate",
    "gini":                "gini",
    "decile_coef":         "decile_coef",
    "wage_to_subsistence": "wage_to_subsistence",
    "urbanization":        "urbanization",
    "alcohol_per100k":     "alcohol_per100k",
    "narco_per100k":       "narco_per100k",
    "population_total":    "population_total",
    "pop_working_age_pct": "pop_working_age_pct",
    "crime_murder":        "crime_murder",
    "crime_hooliganism":   "crime_hooliganism",
    "crime_drugs":         "crime_drugs",
    "juvenile_crime_share": "juvenile_crime_share",
}

DEFAULT_CSV = Path(__file__).parents[3] / "data" / "processed" / "panel.csv"


def _nan_to_none(val):
    """Преобразует NaN/float('nan') в None для Django."""
    if val is None:
        return None
    try:
        if math.isnan(float(val)):
            return None
    except (TypeError, ValueError):
        pass
    return val


class Command(BaseCommand):
    help = "Импортирует panel.csv в PanelObservation"

    def add_arguments(self, parser):
        parser.add_argument(
            "--csv", default=str(DEFAULT_CSV),
            help="Путь к CSV-файлу (по умолчанию: data/processed/panel.csv)"
        )
        parser.add_argument(
            "--clear", action="store_true",
            help="Удалить все PanelObservation перед загрузкой"
        )

    def handle(self, *args, **options):
        csv_path = Path(options["csv"])
        if not csv_path.exists():
            raise CommandError(f"Файл не найден: {csv_path}")

        if options["clear"]:
            deleted, _ = PanelObservation.objects.all().delete()
            self.stdout.write(f"Удалено {deleted} старых наблюдений.")

        df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
        self.stdout.write(f"Загружаю {csv_path.name}: {len(df)} строк …")

        region_cache: dict[str, Region] = {
            r.name: r for r in Region.objects.all()
        }

        created_count = updated_count = skipped_count = 0

        for _, row in df.iterrows():
            region_name = str(row["region"]).strip()
            region = region_cache.get(region_name)
            if region is None:
                self.stderr.write(f"  [!] Регион не найден в БД: {region_name!r} — пропущен")
                skipped_count += 1
                continue

            defaults = {
                field: _nan_to_none(row.get(csv_col))
                for csv_col, field in CSV_TO_FIELD.items()
                if csv_col in row.index
            }

            _, created = PanelObservation.objects.update_or_create(
                region=region,
                year=int(row["year"]),
                defaults=defaults,
            )
            if created:
                created_count += 1
            else:
                updated_count += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Готово: создано {created_count}, обновлено {updated_count}, "
                f"пропущено {skipped_count} строк."
            )
        )
