import uuid

from django.db import models


class PredictionJob(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "En attente"
        RUNNING_MSA = "running_msa", "Récupération MSA"
        RUNNING_INFERENCE = "running_inference", "Inférence du modèle"
        COMPLETED = "completed", "Terminé"
        FAILED = "failed", "Échec"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    sequence = models.TextField()
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING
    )

    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    output_dir = models.CharField(max_length=500)
    query_json_path = models.CharField(max_length=500)
    log_path = models.CharField(max_length=500)
    pid = models.IntegerField(null=True, blank=True)

    current_step = models.CharField(max_length=200, blank=True, default="")
    progress_percent = models.FloatField(null=True, blank=True)
    eta_seconds = models.FloatField(null=True, blank=True)
    elapsed_seconds = models.FloatField(null=True, blank=True)

    error_message = models.TextField(blank=True, default="")
    best_plddt = models.FloatField(null=True, blank=True)
    result_summary = models.JSONField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.id} ({self.status})"

    @property
    def is_active(self):
        return self.status in (
            self.Status.PENDING,
            self.Status.RUNNING_MSA,
            self.Status.RUNNING_INFERENCE,
        )


class AttentionRun(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "En attente"
        RUNNING = "running", "Extraction en cours"
        COMPLETED = "completed", "Terminé"
        FAILED = "failed", "Échec"

    FAMILY_CHOICES = [
        ("pairformer_self_attn", "Self-attention Pairformer"),
        ("triangle_start", "Triangle attention (starting node)"),
        ("triangle_end", "Triangle attention (ending node)"),
        ("diffusion_token", "Diffusion Transformer (token)"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey(
        PredictionJob, on_delete=models.CASCADE, related_name="attention_runs"
    )
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING
    )
    layer_families = models.JSONField(default=list)

    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    output_dir = models.CharField(max_length=500)
    log_path = models.CharField(max_length=500)
    pid = models.IntegerField(null=True, blank=True)

    error_message = models.TextField(blank=True, default="")
    meta = models.JSONField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"AttentionRun({self.id}) for {self.job_id} ({self.status})"

    @property
    def is_active(self):
        return self.status in (self.Status.PENDING, self.Status.RUNNING)
