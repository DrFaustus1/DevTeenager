from django.db import models


class FederalDistrict(models.Model):
    code = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100)

    class Meta:
        verbose_name = "Федеральный округ"
        verbose_name_plural = "Федеральные округа"

    def __str__(self):
        return self.name


class Region(models.Model):
    code = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=200)
    district = models.ForeignKey(FederalDistrict, on_delete=models.PROTECT, related_name="regions")
    is_excluded = models.BooleanField(
        default=False,
        help_text="True для 6 регионов СКФО — исключаются из анализа"
    )

    class Meta:
        verbose_name = "Регион"
        verbose_name_plural = "Регионы"
        ordering = ["name"]

    def __str__(self):
        return self.name


class PanelObservation(models.Model):
    region = models.ForeignKey(Region, on_delete=models.CASCADE, related_name="observations")
    year = models.IntegerField()

    juvenile_crime_share = models.FloatField(null=True, blank=True)
    avg_income = models.FloatField(null=True, blank=True)
    real_income_index = models.FloatField(null=True, blank=True)
    poverty_rate = models.FloatField(null=True, blank=True)
    unemployment_rate = models.FloatField(null=True, blank=True)
    gini = models.FloatField(null=True, blank=True)
    decile_coef = models.FloatField(null=True, blank=True)
    wage_to_subsistence = models.FloatField(null=True, blank=True)
    urbanization = models.FloatField(null=True, blank=True)
    alcohol_per100k = models.FloatField(null=True, blank=True)
    narco_per100k = models.FloatField(null=True, blank=True)
    population_total = models.FloatField(null=True, blank=True)
    pop_working_age_pct = models.FloatField(null=True, blank=True)
    crime_murder = models.FloatField(null=True, blank=True)
    crime_hooliganism = models.FloatField(null=True, blank=True)
    crime_drugs = models.FloatField(null=True, blank=True)

    class Meta:
        unique_together = ("region", "year")
        ordering = ["region", "year"]
        verbose_name = "Панельное наблюдение"
        verbose_name_plural = "Панельные наблюдения"

    def __str__(self):
        return f"{self.region.name} — {self.year}"
