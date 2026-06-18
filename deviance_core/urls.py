from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("regions.urls")),
    path("analytics/", include("analytics.urls")),
    path("llm/", include("llm_agent.urls")),
]
