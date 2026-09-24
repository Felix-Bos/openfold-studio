import sys
from unittest import mock

from django.test import override_settings

from predictor import tasks
from predictor.models import PredictionJob

from .base import JobsDirTestCase
from .factories import SEQUENCE


def fake_openfold(script: str):
    """Replaces the OpenFold command by a Python one-liner."""
    return mock.patch(
        "predictor.openfold.commands.predict_command",
        return_value=[sys.executable, "-c", script],
    )


class PredictionTaskTests(JobsDirTestCase):
    def setUp(self):
        super().setUp()
        override = override_settings(OPENFOLD_PROJECT_DIR=self.jobs_dir)
        override.enable()
        self.addCleanup(override.disable)
        self.job = PredictionJob.objects.create(sequence=SEQUENCE)

    def test_successful_run_is_completed_and_logged(self):
        with fake_openfold("print('COMPLETE: 100%|#| 1/1 [elapsed: 00:01 remaining: 00:00]')"):
            tasks._run_prediction(self.job.id)
        self.job.refresh_from_db()
        self.assertEqual(self.job.status, PredictionJob.Status.COMPLETED)
        self.assertTrue(self.job.workspace.query_file.exists())
        self.assertIn("COMPLETE", self.job.workspace.log_file.read_text())

    def test_failed_run_keeps_the_error_lines(self):
        with fake_openfold("import sys; print('Traceback (most recent call last):'); print('ValueError: GPU on fire'); sys.exit(1)"):
            tasks._run_prediction(self.job.id)
        self.job.refresh_from_db()
        self.assertEqual(self.job.status, PredictionJob.Status.FAILED)
        self.assertIn("GPU on fire", self.job.error_message)
