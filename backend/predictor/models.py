"""Persistent entities: a prediction job and the attention runs made on it.

Each model owns its state transitions (mark_running, mark_completed, ...), so
the rules for moving between statuses live in one place. File locations are
not stored: `workspace` derives them from the id (see openfold/workspace.py).
"""

import uuid

from django.db import models
from django.utils import timezone

from .domain.attention import LAYER_FAMILY_CHOICES
from .openfold.log_parser import Phase, ProgressEvent
from .openfold.workspace import AttentionWorkspace, JobWorkspace

# How many trailing error lines of a failed process are kept for display.
MAX_ERROR_LINES = 40


class ActiveQuerySet(models.QuerySet):
    """Adds `.active()`: rows whose status is in the model's ACTIVE_STATUSES."""

    def active(self):
        return self.filter(status__in=self.model.ACTIVE_STATUSES)


def _failure_message(error_lines: list[str], return_code: int | None) -> str:
    if error_lines:
        return "\n".join(error_lines[-MAX_ERROR_LINES:])
    return f"The process exited with code {return_code}."


class PredictionJob(models.Model):
    """One structure prediction of one protein sequence."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING_MSA = "running_msa", "Fetching MSA"
        RUNNING_INFERENCE = "running_inference", "Running inference"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    ACTIVE_STATUSES = (Status.PENDING, Status.RUNNING_MSA, Status.RUNNING_INFERENCE)

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    sequence = models.TextField()
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)

    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    pid = models.IntegerField(null=True, blank=True)

    # Live progress, updated from the OpenFold log while the job runs.
    current_step = models.CharField(max_length=200, blank=True, default="")
    progress_percent = models.FloatField(null=True, blank=True)
    eta_seconds = models.FloatField(null=True, blank=True)
    elapsed_seconds = models.FloatField(null=True, blank=True)

    # Outcome.
    error_message = models.TextField(blank=True, default="")
    best_plddt = models.FloatField(null=True, blank=True)
    # List of SampleScores.as_dict(), best ranked first.
    result_summary = models.JSONField(null=True, blank=True)

    objects = ActiveQuerySet.as_manager()

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.id} ({self.status})"

    @property
    def workspace(self) -> JobWorkspace:
        return JobWorkspace.for_job(self.id)

    @property
    def is_active(self) -> bool:
        return self.status in self.ACTIVE_STATUSES

    @property
    def is_completed(self) -> bool:
        return self.status == self.Status.COMPLETED

    @property
    def best_sample(self) -> dict | None:
        return self.result_summary[0] if self.result_summary else None

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def duration(self):
        """Wall-clock time of the prediction (timedelta), None while unknown."""
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return None

    def mark_started(self, pid: int) -> None:
        self.status = self.Status.RUNNING_MSA
        self.started_at = timezone.now()
        self.current_step = "Starting"
        self.pid = pid
        self.save()

    def apply_progress(self, event: ProgressEvent) -> None:
        """Copies a parsed log event onto the job (not saved)."""
        self.current_step = event.step
        if event.progress_percent is not None:
            self.progress_percent = event.progress_percent
        if event.elapsed_seconds is not None:
            self.elapsed_seconds = event.elapsed_seconds
        self.eta_seconds = event.eta_seconds
        if event.phase is Phase.INFERENCE:
            self.status = self.Status.RUNNING_INFERENCE

    def mark_completed(self, samples: list[dict]) -> None:
        self.status = self.Status.COMPLETED
        self.finished_at = timezone.now()
        self.progress_percent = 100.0
        self.eta_seconds = 0.0
        self.current_step = "Done"
        self.result_summary = samples
        self.best_plddt = samples[0]["avg_plddt"] if samples else None
        self.save()

    def mark_failed(self, error_lines: list[str], return_code: int | None = None) -> None:
        self.status = self.Status.FAILED
        self.finished_at = timezone.now()
        self.error_message = _failure_message(error_lines, return_code)
        self.save()


class AttentionRun(models.Model):
    """Extraction of the attention maps of a completed prediction job."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Extracting"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    ACTIVE_STATUSES = (Status.PENDING, Status.RUNNING)
    FAMILY_CHOICES = LAYER_FAMILY_CHOICES

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey(
        PredictionJob, on_delete=models.CASCADE, related_name="attention_runs"
    )
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    # Keys of the layer families to extract (see domain/attention.py).
    layer_families = models.JSONField(default=list)

    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    pid = models.IntegerField(null=True, blank=True)

    error_message = models.TextField(blank=True, default="")
    # Content of meta.json: residue count, per-family entropy per block, ...
    meta = models.JSONField(null=True, blank=True)

    objects = ActiveQuerySet.as_manager()

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"AttentionRun({self.id}) for {self.job_id} ({self.status})"

    @property
    def workspace(self) -> AttentionWorkspace:
        return JobWorkspace.for_job(self.job_id).attention(self.id)

    @property
    def is_active(self) -> bool:
        return self.status in self.ACTIVE_STATUSES

    @property
    def is_completed(self) -> bool:
        return self.status == self.Status.COMPLETED

    @property
    def duration(self):
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return None

    def mark_started(self, pid: int) -> None:
        self.status = self.Status.RUNNING
        self.started_at = timezone.now()
        self.pid = pid
        self.save()

    def mark_completed(self, meta: dict | None) -> None:
        self.status = self.Status.COMPLETED
        self.finished_at = timezone.now()
        self.meta = meta
        self.save()

    def mark_failed(self, error_lines: list[str], return_code: int | None = None) -> None:
        self.status = self.Status.FAILED
        self.finished_at = timezone.now()
        self.error_message = _failure_message(error_lines, return_code)
        self.save()
