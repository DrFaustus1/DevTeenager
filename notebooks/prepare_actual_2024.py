import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data', 'processed')

NAME_MAP = {
    'г. Москва':                                         'Москва',
    'г. Санкт-Петербург':                                'Санкт-Петербург',
    'Архангельская область (с а/о )':                    'Архангельская область (с автономным округом)',
    'Архангельская область (без а/о)':                   'Архангельская область (без автономного округа)',
    'Ханты-Мансийский автономный округ - Югра':          'Ханты-Мансийский автономный округ — Югра',
    'Тюменская область (с а/о)':                         'Тюменская область (с автономными округами)',
    'Тюменская область (без а/о)':                       'Тюменская область (без автономных округов)',
    'Кемеровская область - Кузбасс':                     'Кемеровская область',
    'Новгородская  область':                             'Новгородская область',
}

SKIP_KEYWORDS = [
    'федеральный округ', 'Российская Федерация', 'ГУ МВД',
]

def is_skip(name: str) -> bool:
    return any(kw in name for kw in SKIP_KEYWORDS)


df_xls = pd.read_excel(
    os.path.join(BASE, 'data', 'raw', 'juvenile_2024.xls'),
    sheet_name=0, header=None
)

raw = df_xls.iloc[5:, [0, 11]].copy()
raw.columns = ['region', 'actual_jcs_2024']
raw = raw.dropna(subset=['region'])
raw['region'] = raw['region'].astype(str).str.strip()

raw = raw[~raw['region'].apply(is_skip)].copy()
raw['region'] = raw['region'].replace(NAME_MAP)
raw['actual_jcs_2024'] = pd.to_numeric(raw['actual_jcs_2024'], errors='coerce')
raw = raw.dropna(subset=['actual_jcs_2024'])

template = pd.read_csv(os.path.join(DATA, 'actual_2024.csv'), sep=';', encoding='utf-8-sig')
result = template[['region']].merge(raw, on='region', how='left')

matched   = result['actual_jcs_2024'].notna().sum()
unmatched = result[result['actual_jcs_2024'].isna()]['region'].tolist()

print(f'Сопоставлено регионов: {matched} / {len(result)}')
if unmatched:
    print(f'Не найдены в Excel ({len(unmatched)}):')
    for r in unmatched:
        print(f'  - {r}')

result.to_csv(os.path.join(DATA, 'actual_2024.csv'), sep=';', index=False, encoding='utf-8-sig')
print(f'\nСохранено: data/processed/actual_2024.csv')
