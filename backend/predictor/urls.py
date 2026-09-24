"""Routes of the HTML pages (namespace "predictor")."""

from django.urls import path

from .views import pages

app_name = "predictor"

urlpatterns = [
    path("", pages.index, name="index"),
    path("architecture/", pages.architecture, name="architecture"),
    path("jobs/<uuid:job_id>/", pages.job_detail, name="job_detail"),
    path("jobs/<uuid:job_id>/attention/", pages.attention_create, name="attention_create"),
    path(
        "jobs/<uuid:job_id>/attention/<uuid:run_id>/",
        pages.attention_explorer,
        name="attention_explorer",
    ),
]
