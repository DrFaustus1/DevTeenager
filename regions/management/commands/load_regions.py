from django.core.management.base import BaseCommand
from regions.models import FederalDistrict, Region


DISTRICTS = [
    ("CFO",  "Центральный федеральный округ"),
    ("SZFO", "Северо-Западный федеральный округ"),
    ("YUFO", "Южный федеральный округ"),
    ("SKFO", "Северо-Кавказский федеральный округ"),
    ("PFO",  "Приволжский федеральный округ"),
    ("UFO",  "Уральский федеральный округ"),
    ("SFO",  "Сибирский федеральный округ"),
    ("DVFO", "Дальневосточный федеральный округ"),
]

REGION_DISTRICT: dict[str, tuple[str, bool]] = {
    "Белгородская область":                          ("CFO",  False),
    "Брянская область":                              ("CFO",  False),
    "Владимирская область":                          ("CFO",  False),
    "Воронежская область":                           ("CFO",  False),
    "Ивановская область":                            ("CFO",  False),
    "Калужская область":                             ("CFO",  False),
    "Костромская область":                           ("CFO",  False),
    "Курская область":                               ("CFO",  False),
    "Липецкая область":                              ("CFO",  False),
    "Москва":                                        ("CFO",  False),
    "Московская область":                            ("CFO",  False),
    "Орловская область":                             ("CFO",  False),
    "Рязанская область":                             ("CFO",  False),
    "Смоленская область":                            ("CFO",  False),
    "Тамбовская область":                            ("CFO",  False),
    "Тверская область":                              ("CFO",  False),
    "Тульская область":                              ("CFO",  False),
    "Ярославская область":                           ("CFO",  False),
    "Республика Карелия":                            ("SZFO", False),
    "Республика Коми":                               ("SZFO", False),
    "Архангельская область (с автономным округом)":  ("SZFO", False),
    "Архангельская область (без автономного округа)": ("SZFO", False),
    "Ненецкий автономный округ":                     ("SZFO", False),
    "Вологодская область":                           ("SZFO", False),
    "Калининградская область":                       ("SZFO", False),
    "Ленинградская область":                         ("SZFO", False),
    "Мурманская область":                            ("SZFO", False),
    "Новгородская область":                          ("SZFO", False),
    "Псковская область":                             ("SZFO", False),
    "Санкт-Петербург":                               ("SZFO", False),
    "Республика Адыгея":                             ("YUFO", False),
    "Республика Калмыкия":                           ("YUFO", False),
    "Краснодарский край":                            ("YUFO", False),
    "Астраханская область":                          ("YUFO", False),
    "Волгоградская область":                         ("YUFO", False),
    "Ростовская область":                            ("YUFO", False),
    "Республика Крым":                               ("YUFO", False),
    "Севастополь":                                   ("YUFO", False),
    "Ставропольский край":                           ("SKFO", False),
    "Республика Башкортостан":                       ("PFO",  False),
    "Республика Марий Эл":                           ("PFO",  False),
    "Республика Мордовия":                           ("PFO",  False),
    "Республика Татарстан":                          ("PFO",  False),
    "Удмуртская Республика":                         ("PFO",  False),
    "Чувашская Республика":                          ("PFO",  False),
    "Пермский край":                                 ("PFO",  False),
    "Кировская область":                             ("PFO",  False),
    "Нижегородская область":                         ("PFO",  False),
    "Оренбургская область":                          ("PFO",  False),
    "Пензенская область":                            ("PFO",  False),
    "Самарская область":                             ("PFO",  False),
    "Саратовская область":                           ("PFO",  False),
    "Ульяновская область":                           ("PFO",  False),
    "Курганская область":                            ("UFO",  False),
    "Свердловская область":                          ("UFO",  False),
    "Тюменская область (с автономными округами)":    ("UFO",  False),
    "Тюменская область (без автономных округов)":    ("UFO",  False),
    "Ханты-Мансийский автономный округ — Югра":      ("UFO",  False),
    "Ямало-Ненецкий автономный округ":               ("UFO",  False),
    "Челябинская область":                           ("UFO",  False),
    "Республика Алтай":                              ("SFO",  False),
    "Республика Тыва":                               ("SFO",  False),
    "Республика Хакасия":                            ("SFO",  False),
    "Алтайский край":                                ("SFO",  False),
    "Красноярский край":                             ("SFO",  False),
    "Иркутская область":                             ("SFO",  False),
    "Кемеровская область":                           ("SFO",  False),
    "Новосибирская область":                         ("SFO",  False),
    "Омская область":                                ("SFO",  False),
    "Томская область":                               ("SFO",  False),
    "Республика Бурятия":                            ("DVFO", False),
    "Республика Саха (Якутия)":                      ("DVFO", False),
    "Забайкальский край":                            ("DVFO", False),
    "Камчатский край":                               ("DVFO", False),
    "Приморский край":                               ("DVFO", False),
    "Хабаровский край":                              ("DVFO", False),
    "Амурская область":                              ("DVFO", False),
    "Магаданская область":                           ("DVFO", False),
    "Сахалинская область":                           ("DVFO", False),
    "Еврейская автономная область":                  ("DVFO", False),
    "Чукотский автономный округ":                    ("DVFO", False),
}


class Command(BaseCommand):
    help = "Загружает справочник федеральных округов и регионов"

    def add_arguments(self, parser):
        parser.add_argument(
            "--clear", action="store_true",
            help="Удалить существующие записи перед загрузкой"
        )

    def handle(self, *args, **options):
        if options["clear"]:
            Region.objects.all().delete()
            FederalDistrict.objects.all().delete()
            self.stdout.write("Старые записи удалены.")

        district_map: dict[str, FederalDistrict] = {}
        for code, name in DISTRICTS:
            obj, created = FederalDistrict.objects.get_or_create(
                code=code, defaults={"name": name}
            )
            district_map[code] = obj
            if created:
                self.stdout.write(f"  [+] Округ: {name}")

        created_count = 0
        for idx, (region_name, (dist_code, is_excl)) in enumerate(
            sorted(REGION_DISTRICT.items()), start=1
        ):
            region_code = f"R{idx:02d}"
            _, created = Region.objects.get_or_create(
                name=region_name,
                defaults={
                    "code":        region_code,
                    "district":    district_map[dist_code],
                    "is_excluded": is_excl,
                },
            )
            if created:
                created_count += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Готово: {FederalDistrict.objects.count()} округов, "
                f"{Region.objects.count()} регионов "
                f"({created_count} новых)"
            )
        )
