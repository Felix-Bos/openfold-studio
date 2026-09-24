"""Routes of the JSON API (namespace "api", mounted under /api/)."""

from django.urls import path

from .views import api

app_name = "api"

_job = "jobs/<uuid:job_id>/"
_run = _job + "attention/<uuid:run_id>/"
_family = _run + "families/<str:family>/"

urlpatterns = [
    path(_job + "status/", api.job_status, name="job_status"),
    path(_job + "samples/<int:sample>/structure/", api.sample_structure, name="sample_structure"),
    path(_job + "samples/<int:sample>/confidence/", api.sample_confidence, name="sample_confidence"),
    path(_run + "status/", api.attention_status, name="attention_status"),
    path(_run + "families/", api.attention_families, name="attention_families"),
    path(_family, api.attention_family, name="attention_family"),
    path(_family + "blocks/<int:block>/", api.attention_block, name="attention_block"),
    path(_family + "blocks/<int:block>/cube/", api.attention_cube, name="attention_cube"),
]
