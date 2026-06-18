import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

OUT = os.path.join(os.path.dirname(__file__), 'eda_output')
os.makedirs(OUT, exist_ok=True)

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_theme(style='whitegrid', palette='muted')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, 'data', 'processed', 'panel.csv'), sep=';')
print(f'Размер: {df.shape}  ({df.region.nunique()} регионов, {df.year.nunique()} лет)')

miss = df.isnull().sum()
miss_pct = (miss / len(df) * 100).round(1)
miss_df = pd.DataFrame({'missing': miss, 'pct': miss_pct}).query('missing > 0').sort_values('pct', ascending=False)
print("\n=== Пропуски ===")
print(miss_df.to_string())

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.barh(miss_df.index, miss_df['pct'], color='steelblue')
ax.bar_label(bars, labels=[f'{v:.1f}%' for v in miss_df['pct']], padding=3)
ax.set_xlabel('Доля пропусков, %')
ax.set_title('Пропущенные значения по столбцам')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUT}/01_missing.png', dpi=120)
plt.close()
print("Сохранён: 01_missing.png")


num_cols = df.select_dtypes(include='number').columns.drop('year')
desc = df[num_cols].describe().T.round(2)
print("\n=== Описательная статистика ===")
print(desc.to_string())


target = 'juvenile_crime_share'
t = df[target].dropna()
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(t, bins=30, color='steelblue', edgecolor='white')
axes[0].set_title('Распределение juvenile_crime_share')
axes[0].set_xlabel('%')
data_by_year = [df.loc[df.year == y, target].dropna().values for y in sorted(df.year.unique())]
axes[1].boxplot(data_by_year, labels=sorted(df.year.unique()), patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.5))
axes[1].set_title('Boxplot по годам')
axes[1].set_xticklabels(sorted(df.year.unique()), rotation=45)
stats.probplot(t, dist='norm', plot=axes[2])
axes[2].set_title('QQ-plot (нормальность)')
plt.tight_layout()
plt.savefig(f'{OUT}/02_target_dist.png', dpi=120)
plt.close()

stat, p = stats.shapiro(t.sample(min(len(t), 5000), random_state=42))
print(f"\n=== Целевая переменная ===")
print(f'Shapiro-Wilk: W={stat:.4f}, p={p:.4f} → {"нормальное" if p>0.05 else "не нормальное"} распределение')
print(f'Skewness={t.skew():.3f}, Kurtosis={t.kurt():.3f}')
print(f'min={t.min():.2f}, max={t.max():.2f}, mean={t.mean():.2f}, median={t.median():.2f}')
print("Сохранён: 02_target_dist.png")


trend = df.groupby('year')[target].agg(['median', 'mean', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
trend.columns = ['median', 'mean', 'q25', 'q75']
trend = trend.dropna()
fig, ax = plt.subplots()
ax.fill_between(trend.index, trend['q25'], trend['q75'], alpha=0.2, label='IQR')
ax.plot(trend.index, trend['median'], marker='o', label='Медиана', linewidth=2)
ax.plot(trend.index, trend['mean'], marker='s', linestyle='--', label='Среднее', linewidth=1.5)
ax.set_title('Динамика доли несовершеннолетних в преступлениях')
ax.set_xlabel('Год'); ax.set_ylabel('%'); ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/03_target_trend.png', dpi=120)
plt.close()
print("Сохранён: 03_target_trend.png")

print("\n=== Тренд по годам ===")
print(trend.round(3).to_string())


econ_cols = {
    'avg_income': 'Среднедушевой доход, руб.',
    'poverty_rate': 'Уровень бедности, %',
    'unemployment_rate': 'Безработица, %',
    'alcohol_per100k': 'Алкоголизм (на 100 тыс.)',
    'narco_per100k': 'Наркомания (на 100 тыс.)',
    'urbanization': 'Урбанизация, %',
}
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, (col, title) in zip(axes.flat, econ_cols.items()):
    trend_e = df.groupby('year')[col].median().dropna()
    ax.plot(trend_e.index, trend_e.values, marker='o', linewidth=2)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Год')
    ax.tick_params(axis='x', rotation=45)
plt.suptitle('Медиана по регионам РФ', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/04_econ_trends.png', dpi=120, bbox_inches='tight')
plt.close()
print("Сохранён: 04_econ_trends.png")


corr_cols = [target,
             'unemployment_rate', 'poverty_rate', 'real_income_index',
             'avg_income', 'grp_per_capita', 'investment_pc',
             'urbanization', 'alcohol_per100k', 'narco_per100k',
             'gini', 'decile_coef', 'wage_to_subsistence',
             'crime_murder', 'crime_hooliganism', 'crime_drugs']

corr_labels = {
    'juvenile_crime_share': 'Доля несоверш.',
    'unemployment_rate':    'Безработица',
    'poverty_rate':         'Бедность',
    'real_income_index':    'Инд. реал. дох.',
    'avg_income':           'Доходы',
    'grp_per_capita':       'ВРП на душу',
    'investment_pc':        'Инвестиции',
    'urbanization':         'Урбанизация',
    'alcohol_per100k':      'Алкоголизм',
    'narco_per100k':        'Наркомания',
    'gini':                 'Джини',
    'decile_coef':          'Децильный к-т',
    'wage_to_subsistence':  'ЗП/ПМ',
    'crime_murder':         'Убийства',
    'crime_hooliganism':    'Хулиганство',
    'crime_drugs':          'Наркопреступления',
}

available = [c for c in corr_cols if c in df.columns]
corr = df[available].corr(method='spearman')
corr.index   = [corr_labels.get(c, c) for c in corr.index]
corr.columns = [corr_labels.get(c, c) for c in corr.columns]

n = len(available)
fig, ax = plt.subplots(figsize=(n + 2, n))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, square=True, cbar_kws={'shrink': 0.7},
            annot_kws={'size': 8})
ax.set_title('Матрица корреляций Спирмена', pad=12)
ax.tick_params(axis='x', rotation=45, labelsize=9)
ax.tick_params(axis='y', rotation=0,  labelsize=9)
plt.tight_layout()
plt.savefig(f'{OUT}/05_corr_matrix.png', dpi=120)
plt.close()

print("\n=== Корреляции с juvenile_crime_share ===")
target_label = corr_labels[target]
print(corr[target_label].drop(target_label).sort_values(key=abs, ascending=False).round(3).to_string())
print("Сохранён: 05_corr_matrix.png")


top_features = ['poverty_rate', 'unemployment_rate', 'avg_income', 'alcohol_per100k', 'urbanization']
fig, axes = plt.subplots(1, len(top_features), figsize=(18, 4))
for ax, feat in zip(axes, top_features):
    sub = df[[feat, target]].dropna()
    ax.scatter(sub[feat], sub[target], alpha=0.3, s=12, color='steelblue')
    r, p = stats.spearmanr(sub[feat], sub[target])
    m, b = np.polyfit(sub[feat], sub[target], 1)
    x_line = np.linspace(sub[feat].min(), sub[feat].max(), 100)
    ax.plot(x_line, m * x_line + b, color='crimson', linewidth=1.5)
    ax.set_xlabel(feat, fontsize=9)
    ax.set_ylabel(target if feat == top_features[0] else '', fontsize=9)
    ax.set_title(f'r={r:.2f}, p={p:.3f}', fontsize=9)
plt.suptitle('Scatter: ключевые предикторы vs juvenile_crime_share')
plt.tight_layout()
plt.savefig(f'{OUT}/06_scatter_features.png', dpi=120)
plt.close()
print("Сохранён: 06_scatter_features.png")


region_avg = df.groupby('region')[target].mean().dropna().sort_values(ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
top10 = region_avg.head(10)
axes[0].barh(top10.index[::-1], top10.values[::-1], color='tomato')
axes[0].set_title('Топ-10 регионов (наибольшая доля)')
axes[0].set_xlabel('%')
bot10 = region_avg.tail(10)
axes[1].barh(bot10.index[::-1], bot10.values[::-1], color='steelblue')
axes[1].set_title('Антитоп-10 (наименьшая доля)')
axes[1].set_xlabel('%')
plt.suptitle('Средняя доля несовершеннолетних в преступлениях (2013–2024)')
plt.tight_layout()
plt.savefig(f'{OUT}/07_top_regions.png', dpi=120)
plt.close()

print("\n=== Топ-10 регионов ===")
print(top10.round(3).to_string())
print("\n=== Антитоп-10 ===")
print(bot10.round(3).to_string())
print("Сохранён: 07_top_regions.png")


pivot = df.pivot_table(index='region', columns='year', values=target)
pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(14, 22))
sns.heatmap(pivot, cmap='YlOrRd', ax=ax, linewidths=0.3,
            cbar_kws={'label': '%', 'shrink': 0.4})
ax.set_title('Доля несовершеннолетних в преступлениях: регион × год')
ax.set_xlabel('Год'); ax.set_ylabel('')
plt.tight_layout()
plt.savefig(f'{OUT}/08_heatmap_region_year.png', dpi=100)
plt.close()
print("Сохранён: 08_heatmap_region_year.png")

print("\n✓ Все графики сохранены в notebooks/eda_output/")
