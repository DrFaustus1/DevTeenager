import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import xgboost as xgb
import shap

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_theme(style='whitegrid', palette='muted')

RANDOM_STATE = 42
TARGET = 'juvenile_crime_share'

df_raw = pd.read_csv('data/processed/panel.csv', sep=';')
print(f'Исходный датасет: {df_raw.shape}')

FEATURES = [
    'avg_income',
    'poverty_rate',
    'unemployment_rate',
    'urbanization',
    'alcohol_per100k',
    'narco_per100k',
    'real_income_index',
    'pop_working_age_pct',
    'grp_per_capita',
    'investment_pc',
]

FEATURE_LABELS = {
    'avg_income':          'Среднедушевой доход',
    'poverty_rate':        'Уровень бедности',
    'unemployment_rate':   'Безработица',
    'urbanization':        'Урбанизация',
    'alcohol_per100k':     'Алкоголизм (на 100 тыс.)',
    'narco_per100k':       'Наркомания (на 100 тыс.)',
    'real_income_index':   'Индекс реальных доходов',
    'pop_working_age_pct': 'Доля нас. тр. возраста',
    'grp_per_capita':      'ВРП на душу населения',
    'investment_pc':       'Инвестиции на душу',
    'year':                'Год (тренд)',
}

FEATURES_WITH_YEAR = FEATURES + ['year']

df = df_raw.dropna(subset=[TARGET]).copy()

keep_cols = list(dict.fromkeys(FEATURES_WITH_YEAR + [TARGET, 'region']))
df_ml = df[keep_cols].dropna(subset=FEATURES_WITH_YEAR + [TARGET])

print(f'Рабочий датасет ML: {df_ml.shape}')
print(f'Регионов: {df_ml["region"].nunique()}')
print(f'Годы: {sorted(df_ml["year"].unique())}')
print(f'\nПропуски в признаках:')
print(df_ml[FEATURES_WITH_YEAR].isnull().sum().to_string())

X = df_ml[FEATURES_WITH_YEAR].values
y = df_ml[TARGET].values
years = df_ml['year'].values
regions = df_ml['region'].values

print(f'X: {X.shape}, y: {y.shape}')
print(f'y: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}')


def walk_forward_cv(model, X, y, years, min_train_years=3):
    """Walk-forward validation по годам панели."""
    unique_years = sorted(np.unique(years))
    results = []

    for i in range(min_train_years, len(unique_years)):
        train_years = unique_years[:i]
        test_year   = unique_years[i]

        train_mask = np.isin(years, train_years)
        test_mask  = years == test_year

        if test_mask.sum() == 0:
            continue

        model.fit(X[train_mask], y[train_mask])
        y_pred = model.predict(X[test_mask])

        results.append({
            'test_year': test_year,
            'n_train':   train_mask.sum(),
            'n_test':    test_mask.sum(),
            'mae':       mean_absolute_error(y[test_mask], y_pred),
            'rmse':      np.sqrt(mean_squared_error(y[test_mask], y_pred)),
            'r2':        r2_score(y[test_mask], y_pred),
        })

    return pd.DataFrame(results)


print('Walk-forward CV запущен...')

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=5,
    max_features=0.7,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

cv_rf = walk_forward_cv(rf, X, y, years, min_train_years=3)
print('Random Forest — Walk-Forward CV:')
print(cv_rf.to_string(index=False))
print(f'\nСредние метрики:')
print(f'  MAE  = {cv_rf.mae.mean():.3f} ± {cv_rf.mae.std():.3f}')
print(f'  RMSE = {cv_rf.rmse.mean():.3f} ± {cv_rf.rmse.std():.3f}')
print(f'  R²   = {cv_rf.r2.mean():.3f} ± {cv_rf.r2.std():.3f}')

xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    verbosity=0,
)

cv_xgb = walk_forward_cv(xgb_model, X, y, years, min_train_years=3)
print('XGBoost — Walk-Forward CV:')
print(cv_xgb.to_string(index=False))
print(f'\nСредние метрики:')
print(f'  MAE  = {cv_xgb.mae.mean():.3f} ± {cv_xgb.mae.std():.3f}')
print(f'  RMSE = {cv_xgb.rmse.mean():.3f} ± {cv_xgb.rmse.std():.3f}')
print(f'  R²   = {cv_xgb.r2.mean():.3f} ± {cv_xgb.r2.std():.3f}')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for metric, title, ax in zip(['mae', 'rmse', 'r2'], ['MAE', 'RMSE', 'R²'], axes):
    ax.plot(cv_rf['test_year'],  cv_rf[metric],  marker='o', label='Random Forest', linewidth=2)
    ax.plot(cv_xgb['test_year'], cv_xgb[metric], marker='s', label='XGBoost',       linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('Тестовый год')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Walk-Forward CV: сравнение моделей по годам', y=1.01)
plt.tight_layout()
plt.savefig('data/processed/ml_cv_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

compare_df = pd.DataFrame({
    'Модель':  ['Random Forest', 'XGBoost'],
    'MAE':     [cv_rf.mae.mean(),  cv_xgb.mae.mean()],
    'RMSE':    [cv_rf.rmse.mean(), cv_xgb.rmse.mean()],
    'R²':      [cv_rf.r2.mean(),   cv_xgb.r2.mean()],
    'MAE std': [cv_rf.mae.std(),   cv_xgb.mae.std()],
}).round(3)
print(compare_df.to_string(index=False))

rf_full = RandomForestRegressor(
    n_estimators=300, max_depth=6, min_samples_leaf=5,
    max_features=0.7, random_state=RANDOM_STATE, n_jobs=-1,
)
rf_full.fit(X, y)

xgb_full = xgb.XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE, verbosity=0,
)
xgb_full.fit(X, y)

y_pred_rf  = rf_full.predict(X)
y_pred_xgb = xgb_full.predict(X)

print('In-sample (полный датасет):')
print(f'  RF  R²={r2_score(y, y_pred_rf):.3f}  MAE={mean_absolute_error(y, y_pred_rf):.3f}')
print(f'  XGB R²={r2_score(y, y_pred_xgb):.3f}  MAE={mean_absolute_error(y, y_pred_xgb):.3f}')

feat_names = [FEATURE_LABELS.get(f, f) for f in FEATURES_WITH_YEAR]

rf_imp  = pd.Series(rf_full.feature_importances_,  index=feat_names).sort_values(ascending=True)
xgb_imp = pd.Series(xgb_full.feature_importances_, index=feat_names).sort_values(ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].barh(rf_imp.index,  rf_imp.values,  color='steelblue', alpha=0.8)
axes[0].set_title('Random Forest — встроенная важность')
axes[0].set_xlabel('Feature Importance')

axes[1].barh(xgb_imp.index, xgb_imp.values, color='tomato', alpha=0.8)
axes[1].set_title('XGBoost — встроенная важность')
axes[1].set_xlabel('Feature Importance')

plt.suptitle('Важность признаков (встроенная)')
plt.tight_layout()
plt.savefig('data/processed/ml_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

explainer_rf = shap.TreeExplainer(rf_full)

sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X), size=min(len(X), 300), replace=False)
X_sample = X[sample_idx]

shap_values_rf = explainer_rf.shap_values(X_sample)
print(f'SHAP values shape: {shap_values_rf.shape}')

X_sample_df = pd.DataFrame(X_sample, columns=FEATURES_WITH_YEAR)

fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    shap_values_rf, X_sample_df,
    feature_names=feat_names,
    show=False, plot_size=None,
)
plt.title('SHAP Beeswarm — Random Forest')
plt.tight_layout()
plt.savefig('data/processed/shap_rf_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()

mean_shap_rf = pd.Series(
    np.abs(shap_values_rf).mean(axis=0),
    index=feat_names,
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(mean_shap_rf.index, mean_shap_rf.values, color='steelblue', alpha=0.8)
ax.set_xlabel('Средний |SHAP|')
ax.set_title('SHAP — средний вклад признаков (Random Forest)')
plt.tight_layout()
plt.savefig('data/processed/shap_rf_bar.png', dpi=150, bbox_inches='tight')
plt.close()

print('Топ признаки по SHAP (RF):')
print(mean_shap_rf[::-1].round(4).to_string())

explainer_xgb = shap.TreeExplainer(xgb_full)
shap_values_xgb = explainer_xgb.shap_values(X_sample)

mean_shap_xgb = pd.Series(
    np.abs(shap_values_xgb).mean(axis=0),
    index=feat_names,
).sort_values(ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].barh(mean_shap_rf.index,  mean_shap_rf.values,  color='steelblue', alpha=0.8)
axes[0].set_title('SHAP — Random Forest')
axes[0].set_xlabel('Средний |SHAP|')

axes[1].barh(mean_shap_xgb.index, mean_shap_xgb.values, color='tomato', alpha=0.8)
axes[1].set_title('SHAP — XGBoost')
axes[1].set_xlabel('Средний |SHAP|')

plt.suptitle('Сравнение SHAP-важности признаков')
plt.tight_layout()
plt.savefig('data/processed/shap_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

top2_labels = mean_shap_rf[::-1].head(2).index.tolist()
top2_feats  = [FEATURES_WITH_YEAR[feat_names.index(lbl)] for lbl in top2_labels]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, feat, label in zip(axes, top2_feats, top2_labels):
    feat_idx = FEATURES_WITH_YEAR.index(feat)
    shap.dependence_plot(
        feat_idx, shap_values_rf, X_sample_df,
        feature_names=feat_names,
        ax=ax, show=False,
    )
    ax.set_title(f'SHAP Dependence: {label}')

plt.tight_layout()
plt.savefig('data/processed/shap_dependence.png', dpi=150, bbox_inches='tight')
plt.close()

last_year  = df_ml['year'].max()
prev_years = sorted(df_ml['year'].unique())[:-1]

train_mask_final = df_ml['year'].isin(prev_years).values
test_mask_final  = (df_ml['year'] == last_year).values

rf_final = RandomForestRegressor(
    n_estimators=300, max_depth=6, min_samples_leaf=5,
    max_features=0.7, random_state=RANDOM_STATE, n_jobs=-1,
)
rf_final.fit(X[train_mask_final], y[train_mask_final])
y_pred_final = rf_final.predict(X[test_mask_final])
y_true_final = y[test_mask_final]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_true_final, y_pred_final, alpha=0.6, s=30, color='steelblue')
lims = [min(y_true_final.min(), y_pred_final.min()) - 0.5,
        max(y_true_final.max(), y_pred_final.max()) + 0.5]
axes[0].plot(lims, lims, 'r--', linewidth=1.2)
axes[0].set_xlabel('Фактическое')
axes[0].set_ylabel('Предсказанное')
axes[0].set_title(f'RF: тест на {last_year}  R²={r2_score(y_true_final, y_pred_final):.2f}')

resid_final = y_true_final - y_pred_final
axes[1].hist(resid_final, bins=20, color='steelblue', edgecolor='white')
axes[1].axvline(0, color='crimson', linewidth=1.2)
axes[1].set_title(f'Остатки (тест {last_year})')
axes[1].set_xlabel('Ошибка')

plt.tight_layout()
plt.savefig('data/processed/ml_pred_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()

print(f'Тест {last_year}:')
print(f'  MAE  = {mean_absolute_error(y_true_final, y_pred_final):.3f}')
print(f'  RMSE = {np.sqrt(mean_squared_error(y_true_final, y_pred_final)):.3f}')
print(f'  R²   = {r2_score(y_true_final, y_pred_final):.3f}')

shap_rank_rf  = mean_shap_rf[::-1].reset_index()
shap_rank_rf.columns  = ['Признак', 'SHAP_RF']
shap_rank_rf['Ранг RF'] = range(1, len(shap_rank_rf) + 1)

shap_rank_xgb = mean_shap_xgb[::-1].reset_index()
shap_rank_xgb.columns  = ['Признак', 'SHAP_XGB']
shap_rank_xgb['Ранг XGB'] = range(1, len(shap_rank_xgb) + 1)

shap_summary = shap_rank_rf.merge(shap_rank_xgb, on='Признак')
shap_summary['SHAP_RF']  = shap_summary['SHAP_RF'].round(4)
shap_summary['SHAP_XGB'] = shap_summary['SHAP_XGB'].round(4)
print('SHAP-ранги признаков:')
print(shap_summary.to_string(index=False))

metrics_summary = pd.DataFrame({
    'Модель':  ['Random Forest', 'XGBoost'],
    'CV MAE':  [round(cv_rf.mae.mean(), 3),  round(cv_xgb.mae.mean(), 3)],
    'CV RMSE': [round(cv_rf.rmse.mean(), 3), round(cv_xgb.rmse.mean(), 3)],
    'CV R²':   [round(cv_rf.r2.mean(), 3),   round(cv_xgb.r2.mean(), 3)],
})
print('\nМетрики Walk-Forward CV:')
print(metrics_summary.to_string(index=False))

shap_summary.to_csv('data/processed/ml_shap_summary.csv', index=False, encoding='utf-8-sig')
metrics_summary.to_csv('data/processed/ml_metrics.csv', index=False, encoding='utf-8-sig')
print('\nАртефакты сохранены: ml_shap_summary.csv, ml_metrics.csv')
