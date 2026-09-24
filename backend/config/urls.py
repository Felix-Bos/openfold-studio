"""Root URL configuration: HTML pages at /, JSON API at /api/, Django admin at /admin/."""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("predictor.api_urls")),
    path("", include("predictor.urls")),
]
