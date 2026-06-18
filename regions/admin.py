from django.contrib import admin
from .models import FederalDistrict, Region, PanelObservation


@admin.register(FederalDistrict)
class FederalDistrictAdmin(admin.ModelAdmin):
    list_display = ("code", "name")


@admin.register(Region)
class RegionAdmin(admin.ModelAdmin):
    list_display = ("code", "name", "district", "is_excluded")
    list_filter = ("district", "is_excluded")
    search_fields = ("name",)


@admin.register(PanelObservation)
class PanelObservationAdmin(admin.ModelAdmin):
    list_display = (
        "region", "year",
        "juvenile_crime_share",
        "poverty_rate", "unemployment_rate",
        "avg_income", "gini",
        "alcohol_per100k",
    )
    list_filter = ("year", "region__district", "region__is_excluded")
    search_fields = ("region__name",)
    ordering = ("region__name", "year")

    readonly_fields = (
        "region", "year",
        "juvenile_crime_share",
        "avg_income", "real_income_index",
        "poverty_rate", "unemployment_rate",
        "gini", "decile_coef", "wage_to_subsistence",
        "urbanization", "alcohol_per100k", "narco_per100k",
        "population_total", "pop_working_age_pct",
        "crime_murder", "crime_hooliganism", "crime_drugs",
    )
