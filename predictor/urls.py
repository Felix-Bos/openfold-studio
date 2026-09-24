from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("architecture/", views.architecture, name="architecture"),
    path("jobs/<uuid:job_id>/", views.job_detail, name="job_detail"),
    path("jobs/<uuid:job_id>/status/", views.job_status, name="job_status"),
    path(
        "jobs/<uuid:job_id>/sample/<int:sample_num>/cif/",
        views.job_sample_cif,
        name="job_sample_cif",
    ),
    path(
        "jobs/<uuid:job_id>/attention/start/",
        views.attention_start,
        name="attention_start",
    ),
    path(
        "jobs/<uuid:job_id>/attention/<uuid:run_id>/",
        views.attention_explorer,
        name="attention_explorer",
    ),
    path(
        "jobs/<uuid:job_id>/attention/<uuid:run_id>/status/",
        views.attention_status,
        name="attention_status",
    ),
    path(
        "jobs/<uuid:job_id>/attention/<uuid:run_id>/data/<str:family>/",
        views.attention_data,
        name="attention_data",
    ),
    path(
        "jobs/<uuid:job_id>/attention/<uuid:run_id>/data/<str:family>/<int:block>/",
        views.attention_block_data,
        name="attention_block_data",
    ),
    path(
        "jobs/<uuid:job_id>/attention/<uuid:run_id>/cube/<str:family>/<int:block>/",
        views.attention_cube_block_data,
        name="attention_cube_block_data",
    ),
]
