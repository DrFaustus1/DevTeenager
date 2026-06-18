import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['font.family'] = 'DejaVu Sans'

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data', 'processed')

TARGET        = 'juvenile_crime_share'
TREND_YEARS   = 5
FORECAST_YEARS = [2024, 2025, 2026]
SCENARIO_SHIFT = 0.10

FEATURES = [
    'unemployment_rate',
    'poverty_rate',
    'real_income_index',
    'narco_per100k',
    'alcohol_per100k',
    'avg_income',
    'grp_per_capita',
]

FEATURES_XGB = FEATURES + ['year', 'region_enc', 'jcs_lag1']
FEATURES_CAT = FEATURES + ['year', 'jcs_lag1']

df = pd.read_csv(os.path.join(DATA, 'panel.csv'), sep=';')
df = df.sort_values(['region', 'year']).reset_index(drop=True)

df['jcs_lag1'] = df.groupby('region')[TARGET].shift(1)

le_region = LabelEncoder()
le_region.fit(df['region'])
df['region_enc'] = le_region.transform(df['region'])

df_hist = df.dropna(subset=[TARGET, 'jcs_lag1'] + FEATURES).copy()
df_hist = df_hist.sort_values(['region', 'year']).reset_index(drop=True)

print(f'Данные для обучения: {df_hist.shape[0]} строк, '
      f'{df_hist.region.nunique()} регионов, '
      f'годы {df_hist.year.min()}–{df_hist.year.max()}')

y_all = df_hist[TARGET].values

unique_years = sorted(df_hist['year'].unique())
wf_results = []

XGB_PARAMS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
)
CAT_PARAMS = dict(
    iterations=300, depth=4, learning_rate=0.05,
    l2_leaf_reg=3.0, random_seed=42, verbose=0,
)

print(f'\nWalk-Forward CV ({len(unique_years) - 3} шагов)...')
for i in range(3, len(unique_years)):
    train_mask = df_hist['year'].isin(unique_years[:i])
    test_mask  = df_hist['year'] == unique_years[i]
    if test_mask.sum() == 0:
        continue

    y_tr = df_hist.loc[train_mask, TARGET].values
    y_te = df_hist.loc[test_mask,  TARGET].values

    X_tr_xgb = df_hist.loc[train_mask, FEATURES_XGB].values
    X_te_xgb = df_hist.loc[test_mask,  FEATURES_XGB].values
    m_xgb = xgb.XGBRegressor(**XGB_PARAMS)
    m_xgb.fit(X_tr_xgb, y_tr)
    yp_xgb = m_xgb.predict(X_te_xgb).clip(min=0)

    X_tr_cat = df_hist.loc[train_mask, FEATURES_CAT + ['region']].copy()
    X_te_cat = df_hist.loc[test_mask,  FEATURES_CAT + ['region']].copy()
    m_cat = CatBoostRegressor(**CAT_PARAMS)
    m_cat.fit(X_tr_cat, y_tr, cat_features=['region'])
    yp_cat = m_cat.predict(X_te_cat).clip(min=0)

    wf_results.append({
        'test_year': unique_years[i],
        'mae_xgb':  mean_absolute_error(y_te, yp_xgb),
        'rmse_xgb': np.sqrt(mean_squared_error(y_te, yp_xgb)),
        'r2_xgb':   r2_score(y_te, yp_xgb),
        'mae_cat':  mean_absolute_error(y_te, yp_cat),
        'rmse_cat': np.sqrt(mean_squared_error(y_te, yp_cat)),
        'r2_cat':   r2_score(y_te, yp_cat),
    })

wf_df = pd.DataFrame(wf_results)
print(f'\n{"Модель":<30} {"MAE":>7} {"RMSE":>7} {"R²":>7}')
print('-' * 50)
print(f'{"XGBoost + регион + лаг":<30} '
      f'{wf_df.mae_xgb.mean():>7.3f} {wf_df.rmse_xgb.mean():>7.3f} {wf_df.r2_xgb.mean():>7.3f}')
print(f'{"CatBoost + регион + лаг":<30} '
      f'{wf_df.mae_cat.mean():>7.3f} {wf_df.rmse_cat.mean():>7.3f} {wf_df.r2_cat.mean():>7.3f}')

if wf_df.r2_cat.mean() >= wf_df.r2_xgb.mean():
    BEST = 'CatBoost'
else:
    BEST = 'XGBoost'
print(f'\n=> Лучшая модель для прогноза: {BEST}')

if BEST == 'CatBoost':
    X_all = df_hist[FEATURES_CAT + ['region']].copy()
    final_model = CatBoostRegressor(**CAT_PARAMS)
    final_model.fit(X_all, y_all, cat_features=['region'])

    def _predict(df_input: pd.DataFrame) -> np.ndarray:
        X = df_input[FEATURES_CAT + ['region']].copy()
        return final_model.predict(X).clip(min=0)
else:
    X_all = df_hist[FEATURES_XGB].values
    final_model = xgb.XGBRegressor(**XGB_PARAMS)
    final_model.fit(X_all, y_all)

    def _predict(df_input: pd.DataFrame) -> np.ndarray:
        X = df_input[FEATURES_XGB].values
        return final_model.predict(X).clip(min=0)

y_insample = _predict(df_hist)
print(f'In-sample: R²={r2_score(y_all, y_insample):.3f}, '
      f'MAE={mean_absolute_error(y_all, y_insample):.3f}')

def extrapolate_region(region_df: pd.DataFrame, feature: str, forecast_years: list) -> dict:
    sub = region_df[['year', feature]].dropna().sort_values('year')
    if len(sub) < 2:
        last_val = sub[feature].iloc[-1] if len(sub) == 1 else np.nan
        return {y: last_val for y in forecast_years}
    sub = sub.tail(TREND_YEARS)
    x = sub['year'].values.astype(float)
    y = sub[feature].values.astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    return {yr: slope * yr + intercept for yr in forecast_years}

regions = df_hist['region'].unique()
print(f'\nЭкстраполяция предикторов для {len(regions)} регионов...')

forecast_rows = []
for region in regions:
    reg_df = df[df['region'] == region].sort_values('year')
    for yr in FORECAST_YEARS:
        row = {'region': region, 'year': yr,
               'region_enc': int(le_region.transform([region])[0])}
        for feat in FEATURES:
            proj = extrapolate_region(reg_df, feat, FORECAST_YEARS)
            row[feat] = proj[yr]
        forecast_rows.append(row)

df_future_base = pd.DataFrame(forecast_rows)

non_negative = ['unemployment_rate', 'poverty_rate', 'narco_per100k',
                'alcohol_per100k', 'avg_income', 'grp_per_capita']
for col in non_negative:
    df_future_base[col] = df_future_base[col].clip(lower=0)

last_hist_year = df_hist['year'].max()

last_actuals = (df_hist[df_hist['year'] == last_hist_year]
                .set_index('region')[TARGET].to_dict())


def forecast_chained(df_scenario: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
    results = []
    lag_prev = dict(last_actuals)

    for yr in FORECAST_YEARS:
        yr_rows = df_scenario[df_scenario['year'] == yr].copy()
        yr_rows['jcs_lag1'] = yr_rows['region'].map(lag_prev)

        preds = _predict(yr_rows)

        out = yr_rows[['region', 'year']].copy()
        out['jcs_forecast'] = preds.round(4)
        out['scenario'] = scenario_name
        results.append(out)

        lag_prev = dict(zip(yr_rows['region'], preds))

    return pd.concat(results, ignore_index=True)


pred_base    = forecast_chained(df_future_base, 'Базовый')
forecast_all = pred_base.copy()

hist_actual = df_hist[['region', 'year', TARGET]].rename(columns={TARGET: 'jcs_forecast'})
hist_actual['scenario'] = 'Факт'
forecast_full = pd.concat([hist_actual, forecast_all], ignore_index=True)

out_path = os.path.join(DATA, 'forecast_results.csv')
forecast_full.to_csv(out_path, sep=';', index=False, encoding='utf-8-sig')
print(f'\nСохранено: {out_path}  ({len(forecast_full)} строк)')

forecast_all.to_csv(os.path.join(DATA, 'forecast_only.csv'),
                    sep=';', index=False, encoding='utf-8-sig')

print('\n=== Средний прогноз по России ===')
summary = (forecast_all.groupby(['year', 'scenario'])['jcs_forecast']
           .agg(['mean', 'min', 'max']).round(3))
print(summary.to_string())

last_hist = (df_hist[df_hist['year'] == last_hist_year]
             [['region', TARGET]].rename(columns={TARGET: 'jcs_hist'}))
last_fc_year = FORECAST_YEARS[-1]
base_last = pred_base[pred_base['year'] == last_fc_year][['region', 'jcs_forecast']]
comparison = last_hist.merge(base_last, on='region')
comparison['delta'] = (comparison['jcs_forecast'] - comparison['jcs_hist']).round(3)
comparison = comparison.sort_values('delta', ascending=False)
print(f'\nТоп-10 регионов с наибольшим ростом JCS к {last_fc_year}:')
print(comparison.head(10)[['region', 'jcs_hist', 'jcs_forecast', 'delta']].to_string(index=False))

hist_avg = (df_hist.groupby('year')[TARGET].mean().reset_index()
            .rename(columns={TARGET: 'jcs'}))
anchor_avg = hist_avg[hist_avg['year'] == last_hist_year]['jcs'].values[0]

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(hist_avg['year'], hist_avg['jcs'],
        color='steelblue', linewidth=2.5, marker='o', label='Факт (среднее по РФ)')

avg_base = pred_base.groupby('year')['jcs_forecast'].mean()
years_plot  = [last_hist_year] + list(avg_base.index)
values_plot = [anchor_avg]     + list(avg_base.values)
ax.plot(years_plot, values_plot,
        color='tomato', linewidth=2.5, marker='s', label='Прогноз')

ax.set_xlabel('Год')
ax.set_ylabel('Доля подростковой преступности, %')
ax.set_title(f'Прогноз доли подростковой преступности в России\n({BEST})')
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(DATA, 'forecast_avg.png'), dpi=150, bbox_inches='tight')
plt.close()
print('\nСохранён: forecast_avg.png')

top5_regions = comparison.head(5)['region'].tolist()
fig, axes = plt.subplots(1, min(5, len(top5_regions)), figsize=(16, 4), sharey=False)
if len(top5_regions) == 1:
    axes = [axes]

for ax, region in zip(axes, top5_regions):
    hist_r = df_hist[df_hist['region'] == region].sort_values('year')
    ax.plot(hist_r['year'], hist_r[TARGET],
            color='steelblue', linewidth=2, marker='o', label='Факт')

    anchor_r_arr = hist_r[hist_r['year'] == last_hist_year][TARGET].values
    anchor_val = anchor_r_arr[0] if len(anchor_r_arr) > 0 else hist_r[TARGET].iloc[-1]

    fc = pred_base[pred_base['region'] == region].sort_values('year')
    years_plot  = [last_hist_year] + list(fc['year'])
    values_plot = [anchor_val]     + list(fc['jcs_forecast'])
    ax.plot(years_plot, values_plot,
            color='tomato', linewidth=2, marker='s', label='Прогноз')

    ax.set_title(region[:20], fontsize=8)
    ax.set_xlabel('Год', fontsize=8)
    ax.tick_params(labelsize=7)

axes[0].set_ylabel('JCS, %')
plt.suptitle('Прогноз по регионам с наибольшим ростом JCS', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(DATA, 'forecast_regions.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Сохранён: forecast_regions.png')

test_years_str = f"{int(wf_df['test_year'].min())}–{int(wf_df['test_year'].max())}"
cv_summary = pd.DataFrame([
    {
        'Модель': 'XGBoost + регион + лаг',
        'MAE':    round(wf_df.mae_xgb.mean(), 3),
        'RMSE':   round(wf_df.rmse_xgb.mean(), 3),
        'R²':     round(wf_df.r2_xgb.mean(), 3),
    },
    {
        'Модель': 'CatBoost + регион + лаг',
        'MAE':    round(wf_df.mae_cat.mean(), 3),
        'RMSE':   round(wf_df.rmse_cat.mean(), 3),
        'R²':     round(wf_df.r2_cat.mean(), 3),
    },
])
cv_summary.to_csv(os.path.join(DATA, 'forecast_cv_summary.csv'), index=False, encoding='utf-8-sig')
print('Сохранён: forecast_cv_summary.csv')

pivot = (forecast_all[forecast_all['scenario'] == 'Базовый']
         .pivot_table(index='region', columns='year', values='jcs_forecast')
         .round(3))
pivot.columns = [f'Прогноз {y}' for y in pivot.columns]
pivot = pivot.reset_index()
pivot.insert(1, f'Факт {last_hist_year}',
             last_hist.set_index('region')['jcs_hist'].reindex(pivot['region']).values)
pivot.to_csv(os.path.join(DATA, 'forecast_table.csv'), sep=';', index=False, encoding='utf-8-sig')
print('Сохранён: forecast_table.csv')

actual_2024_path = os.path.join(DATA, 'actual_2024.csv')

if not os.path.exists(actual_2024_path):
    pd.DataFrame({
        'region': sorted(df_hist['region'].unique()),
        'actual_jcs_2024': ''
    }).to_csv(actual_2024_path, sep=';', index=False, encoding='utf-8-sig')
    print(f'\n[Валидация 2024] Создан шаблон: actual_2024.csv')
    print('  Заполните колонку actual_jcs_2024 из crimestat.ru и запустите скрипт повторно')
else:
    act = pd.read_csv(actual_2024_path, sep=';', encoding='utf-8-sig')
    act = act[act['actual_jcs_2024'].notna() &
              (act['actual_jcs_2024'].astype(str).str.strip() != '')]
    act['actual_jcs_2024'] = pd.to_numeric(act['actual_jcs_2024'], errors='coerce')
    act = act.dropna(subset=['actual_jcs_2024'])

    if len(act) > 0:
        pred_2024 = pred_base[pred_base['year'] == 2024][['region', 'jcs_forecast']].copy()
        pred_2024.rename(columns={'jcs_forecast': 'predicted_2024'}, inplace=True)
        val = act.merge(pred_2024, on='region', how='inner')
        val['abs_error'] = (val['predicted_2024'] - val['actual_jcs_2024']).abs().round(4)
        val['pct_error'] = ((val['abs_error'] /
                             val['actual_jcs_2024'].replace(0, np.nan)) * 100).round(2)

        y_true = val['actual_jcs_2024'].values
        y_pred = val['predicted_2024'].values
        val_mae  = mean_absolute_error(y_true, y_pred)
        val_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        val_r2   = r2_score(y_true, y_pred)
        val_mape = float(np.nanmean(
            np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, np.nan))) * 100)

        print(f'\n[Валидация 2024] {len(val)} регионов:  '
              f'MAE={val_mae:.3f}  RMSE={val_rmse:.3f}  R²={val_r2:.3f}  MAPE={val_mape:.1f}%')

        out_val = val[['region', 'actual_jcs_2024', 'predicted_2024',
                        'abs_error', 'pct_error']].copy()
        out_val.columns = ['Регион', 'Факт 2024', 'Прогноз 2024', 'Абс. ошибка', 'Ошибка %']
        out_val.sort_values('Абс. ошибка', ascending=False).to_csv(
            os.path.join(DATA, 'forecast_vs_actual_2024.csv'),
            sep=';', index=False, encoding='utf-8-sig')

        pd.DataFrame([{
            'MAE': round(val_mae, 3), 'RMSE': round(val_rmse, 3),
            'R²': round(val_r2, 3), 'MAPE': round(val_mape, 1),
            'N': len(val)
        }]).to_csv(os.path.join(DATA, 'validation_metrics_2024.csv'),
                   sep=';', index=False, encoding='utf-8-sig')
        print('Сохранено: forecast_vs_actual_2024.csv, validation_metrics_2024.csv')
    else:
        print('\n[Валидация 2024] actual_2024.csv пуст — заполните данными и запустите повторно')

print(f'\n Прогнозный модуль v2 завершён. Лучшая модель: {BEST}')
print(f'  Walk-Forward R²: XGBoost={wf_df.r2_xgb.mean():.3f}  CatBoost={wf_df.r2_cat.mean():.3f}')

import shutil
STATIC_FORECAST = os.path.join(BASE, 'static', 'forecast')
os.makedirs(STATIC_FORECAST, exist_ok=True)
for fname in ['forecast_avg.png', 'forecast_regions.png']:
    src = os.path.join(DATA, fname)
    dst = os.path.join(STATIC_FORECAST, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f'Скопирован в static/forecast/: {fname}')
