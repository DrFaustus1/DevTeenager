import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from linearmodels import PanelOLS, RandomEffects, PooledOLS
from linearmodels.panel import compare
from scipy import stats
import statsmodels.api as sm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(os.path.dirname(__file__), 'panel_output')
os.makedirs(OUT, exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted')

df = pd.read_csv(os.path.join(BASE, 'data', 'processed', 'panel.csv'), sep=';')

df['log_jcs'] = np.log(df['juvenile_crime_share'])

PREDICTORS = ['avg_income', 'poverty_rate', 'unemployment_rate', 'urbanization',
              'grp_per_capita', 'investment_pc']
TARGET = 'log_jcs'

df['log_income']        = np.log(df['avg_income'])
df['log_grp_pc']        = np.log(df['grp_per_capita'])
df['log_investment_pc'] = np.log(df['investment_pc'])

FEATURES = ['log_income', 'poverty_rate', 'unemployment_rate', 'urbanization',
            'log_grp_pc', 'log_investment_pc']

cols_needed = ['region', 'year', TARGET] + FEATURES
work = df[cols_needed].dropna().copy()
print(f"Рабочая выборка: {work.shape[0]} наблюдений, "
      f"{work.region.nunique()} регионов, годы {work.year.min()}–{work.year.max()}")

work = work.set_index(['region', 'year'])
y = work[TARGET]
X = work[FEATURES]

print(f"\nДисбаланс панели: {work.groupby('region').size().describe().to_dict()}")

X_c = sm.add_constant(X)
pols = PooledOLS(y, X_c).fit(cov_type='clustered', cluster_entity=True)
print("\n" + "="*60)
print("POOLED OLS (cluster-robust SE по регионам)")
print("="*60)
print(pols.summary.tables[1])

fe = PanelOLS(y, X_c, entity_effects=True, time_effects=False).fit(
    cov_type='clustered', cluster_entity=True)
print("\n" + "="*60)
print("FIXED EFFECTS — entity (Within estimator)")
print("="*60)
print(fe.summary.tables[1])
print(f"\nF-тест по entity FE: p = {fe.f_pooled.pval:.4f}")

fe2 = PanelOLS(y, X_c, entity_effects=True, time_effects=True).fit(
    cov_type='clustered', cluster_entity=True)
print("\n" + "="*60)
print("TWO-WAY FIXED EFFECTS (entity + time)")
print("="*60)
print(fe2.summary.tables[1])

re = RandomEffects(y, X_c).fit(cov_type='unadjusted')
print("\n" + "="*60)
print("RANDOM EFFECTS")
print("="*60)
print(re.summary.tables[1])

fe_b  = fe.params[FEATURES]
re_b  = re.params[FEATURES]
fe_v  = fe.cov.loc[FEATURES, FEATURES]
re_v  = re.cov.loc[FEATURES, FEATURES]
diff  = fe_b - re_b
V_diff = fe_v - re_v

try:
    chi2 = float(diff @ np.linalg.inv(V_diff) @ diff)
except np.linalg.LinAlgError:
    chi2 = float(diff @ np.linalg.pinv(V_diff) @ diff)

df_h = len(FEATURES)
p_hausman = 1 - stats.chi2.cdf(abs(chi2), df_h)

print("\n" + "="*60)
print("HAUSMAN TEST (FE vs RE)")
print("="*60)
print(f"chi2({df_h}) = {chi2:.3f},  p = {p_hausman:.4f}")
if p_hausman < 0.05:
    print("=> Отвергаем RE (p<0.05). Предпочтительна модель FIXED EFFECTS.")
else:
    print("=> Не отвергаем RE (p>=0.05). Предпочтительна модель RANDOM EFFECTS.")

print("\n" + "="*60)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("="*60)
comp = compare({'Pooled OLS': pols, 'FE (entity)': fe,
                'FE (2-way)': fe2, 'Random Effects': re},
               stars=True, precision='std_errors')
print(comp)

coef_df = pd.DataFrame({
    'coef': fe.params[FEATURES],
    'ci_lo': fe.params[FEATURES] - 1.96 * fe.std_errors[FEATURES],
    'ci_hi': fe.params[FEATURES] + 1.96 * fe.std_errors[FEATURES],
    'pval': fe.pvalues[FEATURES],
})
labels_map = {
    'log_income':         'log(Среднедушевой доход)',
    'poverty_rate':       'Уровень бедности',
    'unemployment_rate':  'Безработица',
    'urbanization':       'Урбанизация',
    'log_grp_pc':         'log(ВРП на душу)',
    'log_investment_pc':  'log(Инвестиции на душу)',
}
coef_df.index = [labels_map.get(i, i) for i in coef_df.index]

fig, ax = plt.subplots(figsize=(9, 4))
colors = ['tomato' if p < 0.05 else 'steelblue' for p in coef_df['pval']]
ax.barh(coef_df.index, coef_df['coef'], color=colors, alpha=0.8, height=0.5)
ax.errorbar(coef_df['coef'], coef_df.index,
            xerr=[coef_df['coef'] - coef_df['ci_lo'],
                  coef_df['ci_hi'] - coef_df['coef']],
            fmt='none', color='black', capsize=4, linewidth=1.5)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Коэффициент (Δlog_jcs на 1 единицу)')
ax.set_title('Fixed Effects: коэффициенты с 95% CI\n(красный = p<0.05)')
for i, (c, p) in enumerate(zip(coef_df['coef'], coef_df['pval'])):
    stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
    ax.text(c + (0.005 if c >= 0 else -0.005), i,
            f'{c:.3f}{stars}', va='center',
            ha='left' if c >= 0 else 'right', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT}/01_fe_coefs.png', dpi=130)
plt.close()
print("\nСохранён: 01_fe_coefs.png")

fitted = fe.fitted_values.squeeze()
resid  = fe.resids.squeeze()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(fitted, y.values, alpha=0.3, s=10, color='steelblue')
lo, hi = min(fitted.min(), y.min()), max(fitted.max(), y.max())
axes[0].plot([lo, hi], [lo, hi], 'r--', linewidth=1.5)
axes[0].set_xlabel('Fitted log(jcs)')
axes[0].set_ylabel('Actual log(jcs)')
axes[0].set_title(f'Fitted vs Actual  (R²={fe.rsquared:.3f})')

axes[1].scatter(fitted, resid, alpha=0.3, s=10, color='steelblue')
axes[1].axhline(0, color='red', linewidth=1.5, linestyle='--')
axes[1].set_xlabel('Fitted log(jcs)')
axes[1].set_ylabel('Остатки')
axes[1].set_title('Остатки vs Fitted')

stats.probplot(resid, dist='norm', plot=axes[2])
axes[2].set_title('QQ-plot остатков')

plt.tight_layout()
plt.savefig(f'{OUT}/02_diagnostics.png', dpi=130)
plt.close()
print("Сохранён: 02_diagnostics.png")

fe_effects = pd.Series(fe.estimated_effects['estimated_effects'].values,
                       index=work.reset_index()['region'].values)
fe_by_region = fe_effects.groupby(fe_effects.index).mean().sort_values()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
top = fe_by_region.tail(12)
bot = fe_by_region.head(12)

axes[0].barh(bot.index[::-1], bot.values[::-1], color='steelblue', alpha=0.8)
axes[0].axvline(0, color='black', linewidth=0.7, linestyle='--')
axes[0].set_title('Регионы с отрицательными региональными эффектами (низкая JCS)')
axes[0].set_xlabel('Fixed Effect (αᵢ)')

axes[1].barh(top.index, top.values, color='tomato', alpha=0.8)
axes[1].axvline(0, color='black', linewidth=0.7, linestyle='--')
axes[1].set_title('Регионы с положительными региональными эффектами (высокая JCS)')
axes[1].set_xlabel('Fixed Effect (αᵢ)')

plt.suptitle('Региональные фиксированные эффекты (после контроля экономических факторов)')
plt.tight_layout()
plt.savefig(f'{OUT}/03_region_effects.png', dpi=130)
plt.close()
print("Сохранён: 03_region_effects.png")

med = work[FEATURES].median()
coef_fe   = fe.params
intercept = coef_fe.get('const', 0)

def predict_jcs(varying_feat, values):
    base = (intercept
            + sum(coef_fe[f] * (values if f == varying_feat else med[f])
                  for f in FEATURES))
    return np.exp(base)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

grp_range = np.linspace(work['log_grp_pc'].min(), work['log_grp_pc'].max(), 100)
axes[0].plot(np.exp(grp_range) / 1000, predict_jcs('log_grp_pc', grp_range),
             color='steelblue', linewidth=2)
axes[0].set_xlabel('ВРП на душу населения, тыс. руб.')
axes[0].set_ylabel('Прогноз juvenile_crime_share, %')
axes[0].set_title('Предельный эффект ВРП на душу\n(прочие — на медиане)')

inv_range = np.linspace(work['log_investment_pc'].min(), work['log_investment_pc'].max(), 100)
axes[1].plot(np.exp(inv_range) / 1000, predict_jcs('log_investment_pc', inv_range),
             color='tomato', linewidth=2)
axes[1].set_xlabel('Инвестиции на душу населения, тыс. руб.')
axes[1].set_ylabel('Прогноз juvenile_crime_share, %')
axes[1].set_title('Предельный эффект инвестиций на душу\n(прочие — на медиане)')

plt.tight_layout()
plt.savefig(f'{OUT}/04_marginal_effects.png', dpi=130)
plt.close()
print("Сохранён: 04_marginal_effects.png")

print("\n" + "="*60)
print("ИТОГ: ЛУЧШАЯ МОДЕЛЬ — Fixed Effects (entity)")
print("="*60)
print(f"  R²  (within) = {fe.rsquared:.4f}")
print(f"  R²  (between) = {fe.rsquared_between:.4f}")
print(f"  R²  (overall) = {fe.rsquared_overall:.4f}")
print(f"  N obs         = {int(fe.nobs)}")
print(f"  N entities    = {fe.entity_info['total']:.0f}")
print()
for feat in FEATURES:
    c = fe.params[feat]
    p = fe.pvalues[feat]
    se = fe.std_errors[feat]
    stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '(n.s.)'
    print(f"  {feat:25s}  β={c:+.4f}  SE={se:.4f}  p={p:.4f} {stars}")
print(f"\n  Hausman test: chi2={chi2:.3f}, p={p_hausman:.4f}")
print("\n✓ Все графики сохранены в notebooks/panel_output/")
