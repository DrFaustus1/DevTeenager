from django.urls import path
from . import views

urlpatterns = [
    path("", views.region_list, name="region_list"),
    path("region/<int:pk>/", views.region_detail, name="region_detail"),
]
