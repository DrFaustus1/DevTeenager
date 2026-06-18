import json
import plotly.graph_objects as go
from django.db.models import Max
from django.shortcuts import get_object_or_404, render

from .models import FederalDistrict, PanelObservation, Region


def region_list(request):
    districts = FederalDistrict.objects.order_by("name")
    district_filter = request.GET.get("district", "")

    regions = Region.objects.filter(is_excluded=False).select_related("district").order_by("name")
    if district_filter:
        regions = regions.filter(district__code=district_filter)

    latest_year = PanelObservation.objects.aggregate(Max("year"))["year__max"] or 2024

    obs_map = {
        o.region_id: o
        for o in PanelObservation.objects.filter(year=latest_year).select_related("region")
    }

    regions_data = []
    for region in regions:
        obs = obs_map.get(region.pk)
        regions_data.append({
            "region": region,
            "juvenile_crime_share": obs.juvenile_crime_share if obs else None,
            "poverty_rate": obs.poverty_rate if obs else None,
            "unemployment_rate": obs.unemployment_rate if obs else None,
        })

    return render(request, "regions/list.html", {
        "regions_data": regions_data,
        "districts": districts,
        "selected_district": district_filter,
        "latest_year": latest_year,
    })


def region_detail(request, pk):
    region = get_object_or_404(Region, pk=pk)
    observations = list(region.observations.order_by("year"))

    crime_years = [o.year for o in observations if o.juvenile_crime_share is not None]
    crime_vals  = [o.juvenile_crime_share for o in observations if o.juvenile_crime_share is not None]

    fig_crime = go.Figure()
    if crime_years:
        fig_crime.add_trace(go.Scatter(
            x=crime_years, y=crime_vals,
            mode="lines+markers",
            name="Доля преступлений, %",
            line=dict(color="#dc3545", width=2),
            marker=dict(size=6),
        ))
    fig_crime.update_layout(
        title=dict(text="Доля несовершеннолетних среди преступников, %", font=dict(size=14)),
        xaxis_title="Год",
        yaxis_title="%",
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode="x unified",
        template="plotly_white",
    )
    chart_crime = fig_crime.to_html(full_html=False, include_plotlyjs="cdn")

    years_all = [o.year for o in observations]

    def _vals(attr):
        return [getattr(o, attr) for o in observations]

    fig_econ = go.Figure()
    fig_econ.add_trace(go.Scatter(
        x=years_all, y=_vals("poverty_rate"),
        mode="lines+markers", name="Бедность, %",
        line=dict(color="#fd7e14", width=2),
    ))
    fig_econ.add_trace(go.Scatter(
        x=years_all, y=_vals("unemployment_rate"),
        mode="lines+markers", name="Безработица, %",
        line=dict(color="#0d6efd", width=2),
    ))
    fig_econ.add_trace(go.Scatter(
        x=years_all, y=[v / 1000 if v else None for v in _vals("avg_income")],
        mode="lines+markers", name="Доход, тыс. руб.",
        line=dict(color="#198754", width=2),
        yaxis="y2",
    ))
    fig_econ.update_layout(
        title=dict(text="Экономические показатели", font=dict(size=14)),
        xaxis_title="Год",
        yaxis=dict(title="%"),
        yaxis2=dict(title="тыс. руб.", overlaying="y", side="right"),
        height=320,
        margin=dict(l=40, r=60, t=50, b=40),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.2),
    )
    chart_econ = fig_econ.to_html(full_html=False, include_plotlyjs=False)

    latest_obs = observations[-1] if observations else None

    return render(request, "regions/detail.html", {
        "region": region,
        "observations": observations,
        "latest_obs": latest_obs,
        "chart_crime": chart_crime,
        "chart_econ": chart_econ,
        "year_min": observations[0].year if observations else "—",
        "year_max": observations[-1].year if observations else "—",
    })
