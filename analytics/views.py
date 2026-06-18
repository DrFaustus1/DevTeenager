import csv
import os
import re

from django.conf import settings
from django.shortcuts import render
from django.utils.html import format_html

from .models import ModelRun


def _format_coef_cell(cell):
    if '\n' not in cell:
        return cell
    coef_part, se_part = cell.split('\n', 1)
    coef_part = coef_part.strip()
    se_part = se_part.strip()
    m = re.match(r'^(-?[\d.]+)(\*{1,3})?$', coef_part)
    if not m:
        return f'{coef_part}<br><small class="text-muted">{se_part}</small>'
    coef, stars = m.group(1), m.group(2) or ''
    stars_colors = {'***': 'danger', '**': 'warning', '*': 'info'}
    stars_html = (
        f'<sup class="fw-bold text-{stars_colors.get(stars, "danger")}">{stars}</sup>'
        if stars else ''
    )
    return f'<strong>{coef}</strong>{stars_html}<br><small class="text-muted">{se_part}</small>'


def _read_csv(rel_path, delimiter=","):
    path = os.path.join(settings.BASE_DIR, rel_path)
    if not os.path.exists(path):
        return [], []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=delimiter)
        rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _read_forecast_summary():
    path = os.path.join(settings.BASE_DIR, "data/processed/forecast_results.csv")
    if not os.path.exists(path):
        return [], []
    import csv as _csv
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        reader = _csv.DictReader(f, delimiter=";")
        for row in reader:
            if row.get("scenario") != "Факт":
                rows.append(row)
    if not rows:
        return [], []
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        key = (r["year"], r["scenario"])
        try:
            buckets[key].append(float(r["jcs_forecast"]))
        except (ValueError, KeyError):
            pass
    years    = sorted({k[0] for k in buckets})
    scenarios = ["Базовый", "Оптимистичный", "Пессимистичный"]
    headers = ["Сценарий"] + [f"{y}" for y in years]
    table_rows = []
    for sc in scenarios:
        row_data = [sc]
        for yr in years:
            vals = buckets.get((yr, sc), [])
            row_data.append(f"{sum(vals)/len(vals):.3f}" if vals else "—")
        table_rows.append(row_data)
    return headers, table_rows


def dashboard(request):
    reg_headers, raw_reg_rows = _read_csv("data/processed/regression_table.csv")
    reg_rows = [[_format_coef_cell(cell) for cell in row] for row in raw_reg_rows]
    ml_headers, ml_rows   = _read_csv("data/processed/ml_metrics.csv")
    model_runs = ModelRun.objects.all()[:20]

    forecast_headers, forecast_summary = _read_forecast_summary()
    forecast_table_headers, forecast_table_rows = _read_csv("data/processed/forecast_table.csv", delimiter=";")

    eda_images = [
        (
            "02_target_dist.png",
            "Распределение уровня подростковой преступности",
            "Гистограмма и кривая плотности показывают, как распределён уровень подростковой преступности "
            "(число зарегистрированных преступлений несовершеннолетних на 100 тыс. населения) "
            "по всем регионам и годам наблюдений. Позволяет оценить типичные значения, "
            "разброс и наличие регионов-выбросов.",
        ),
        (
            "03_target_trend.png",
            "Динамика подростковой преступности по годам (2010–2023)",
            "Медианное значение показателя по всем регионам России за каждый год. "
            "Линия тренда отражает общую направленность изменений на протяжении периода наблюдений — "
            "снижение или рост преступности в целом по стране.",
        ),
        (
            "04_econ_trends.png",
            "Динамика ключевых социально-экономических показателей",
            "На графике представлены медианные значения по регионам для четырёх показателей: "
            "безработица (% от рабочей силы), среднедушевые денежные доходы населения (руб./мес.), "
            "уровень бедности (% населения с доходами ниже прожиточного минимума), "
            "потребление алкоголя (литров чистого спирта на человека в год). "
            "Показатели нормированы для отображения на общей шкале.",
        ),
        (
            "05_corr_matrix.png",
            "Матрица корреляций между переменными (коэффициент Спирмена)",
            "Расшифровка переменных: "
            "juvenile_crime_rate — уровень подростковой преступности (целевая); "
            "unemployment — уровень безработицы; "
            "avg_income — среднедушевые денежные доходы; "
            "poverty_rate — уровень бедности; "
            "alcohol — потребление алкоголя; "
            "decile_coef / gini — коэффициенты неравенства доходов; "
            "crime_murder — уровень убийств и покушений на убийство; "
            "urban_share — доля городского населения; "
            "year — год наблюдения. "
            "Чем темнее синий цвет — тем сильнее положительная связь, красный — отрицательная.",
        ),
        (
            "06_scatter_features.png",
            "Связь каждого предиктора с уровнем подростковой преступности",
            "Каждая точка — один регион в один год. Линия — локальный тренд (LOWESS). "
            "Названия предикторов расшифрованы в описании к матрице корреляций выше. "
            "Графики помогают увидеть характер связи (линейная, нелинейная) и наличие выбросов.",
        ),
        (
            "07_top_regions.png",
            "Регионы с наибольшим и наименьшим уровнем подростковой преступности",
            "Топ-10 регионов с самым высоким и топ-10 с самым низким средним уровнем "
            "подростковой преступности за весь период наблюдений (2010–2023). "
            "Позволяет выявить устойчивую пространственную неоднородность проблемы.",
        ),
        (
            "08_heatmap_region_year.png",
            "Тепловая карта: уровень подростковой преступности по регионам и годам",
            "Каждая ячейка соответствует одному региону в один год. "
            "Интенсивность цвета отражает уровень подростковой преступности: "
            "чем темнее — тем выше значение. "
            "График позволяет одновременно увидеть межрегиональную и временную вариацию показателя, "
            "а также выявить регионы с устойчиво высоким или низким уровнем на протяжении всего периода.",
        ),
    ]

    panel_images = [
        ("01_fe_coefs.png",       "Коэффициенты FE-регрессии"),
        ("02_diagnostics.png",    "Диагностика остатков"),
        ("03_region_effects.png", "Фиксированные эффекты регионов"),
        ("04_marginal_effects.png", "Предельные эффекты"),
    ]

    ml_images = [
        ("ml_cv_comparison.png",      "Сравнение моделей (CV)"),
        ("ml_feature_importance.png", "Важность признаков (Random Forest)"),
        ("ml_pred_vs_actual.png",     "Предсказанные vs Фактические значения"),
        ("shap_rf_beeswarm.png",      "SHAP beeswarm (Random Forest)"),
        ("shap_rf_bar.png",           "SHAP bar chart (Random Forest)"),
        ("shap_comparison.png",       "Сравнение SHAP RF vs XGBoost"),
        ("shap_dependence.png",       "SHAP dependence plot"),
    ]

    cv_headers, cv_rows = _read_csv("data/processed/forecast_cv_summary.csv")

    validation_headers, validation_rows = _read_csv(
        "data/processed/forecast_vs_actual_2024.csv", ";")
    val_m_h, val_m_r = _read_csv("data/processed/validation_metrics_2024.csv", ";")
    validation_metrics = dict(zip(val_m_h, val_m_r[0])) if val_m_h and val_m_r else None

    forecast_images = [
        ("forecast_avg.png",
         "Прогноз доли подростковой преступности (среднее по РФ, 2015–2026)"),
        ("forecast_regions.png",
         "Прогноз для регионов с наибольшим ожидаемым ростом показателя"),
    ]

    return render(request, "analytics/dashboard.html", {
        "reg_headers":  reg_headers,
        "reg_rows":     reg_rows,
        "ml_headers":   ml_headers,
        "ml_rows":      ml_rows,
        "model_runs":   model_runs,
        "eda_images":   eda_images,
        "panel_images": panel_images,
        "ml_images":    ml_images,
        "cv_headers":            cv_headers,
        "cv_rows":               cv_rows,
        "validation_headers":    validation_headers,
        "validation_rows":       validation_rows,
        "validation_metrics":    validation_metrics,
        "forecast_images":       forecast_images,
        "forecast_headers":      forecast_headers,
        "forecast_summary":      forecast_summary,
        "forecast_table_headers": forecast_table_headers,
        "forecast_table_rows":   forecast_table_rows,
    })
