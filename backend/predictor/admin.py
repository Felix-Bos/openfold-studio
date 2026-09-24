from django.contrib import admin

from .models import AttentionRun, PredictionJob


@admin.register(PredictionJob)
class PredictionJobAdmin(admin.ModelAdmin):
    list_display = ("id", "status", "best_plddt", "created_at", "finished_at")
    list_filter = ("status",)
    search_fields = ("id", "sequence")


@admin.register(AttentionRun)
class AttentionRunAdmin(admin.ModelAdmin):
    list_display = ("id", "job", "status", "created_at", "finished_at")
    list_filter = ("status",)
