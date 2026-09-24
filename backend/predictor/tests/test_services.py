from unittest import mock

from django.utils import timezone

from predictor.errors import ComputeBusyError, JobNotReadyError, ResourceNotFoundError
from predictor.models import AttentionRun, PredictionJob
from predictor.openfold.log_parser import Phase, ProgressEvent
from predictor.openfold.results import read_sample_scores
from predictor.services import attention as attention_service
from predictor.services import predictions as prediction_service

from .base import JobsDirTestCase
from .factories import CIF, SEQUENCE, completed_attention_run, completed_job


@mock.patch("predictor.tasks.start_prediction")
class SubmitPredictionTests(JobsDirTestCase):
    def test_creates_job_directory_and_starts_task(self, start_prediction):
        job = prediction_service.submit_prediction(SEQUENCE)
        self.assertEqual(job.status, PredictionJob.Status.PENDING)
        self.assertTrue(job.workspace.root.is_dir())
        start_prediction.assert_called_once_with(job.id)

    def test_refuses_while_another_prediction_runs(self, start_prediction):
        running = PredictionJob.objects.create(sequence=SEQUENCE, status=PredictionJob.Status.RUNNING_MSA)
        with self.assertRaises(ComputeBusyError) as caught:
            prediction_service.submit_prediction(SEQUENCE)
        self.assertEqual(caught.exception.blocking, running)
        start_prediction.assert_not_called()

    def test_refuses_while_an_attention_run_is_active(self, start_prediction):
        job = completed_job()
        AttentionRun.objects.create(job=job, status=AttentionRun.Status.RUNNING)
        with self.assertRaises(ComputeBusyError):
            prediction_service.submit_prediction(SEQUENCE)


class ProgressTests(JobsDirTestCase):
    def test_inference_progress_is_indeterminate(self):
        job = PredictionJob.objects.create(
            sequence=SEQUENCE, status=PredictionJob.Status.RUNNING_INFERENCE,
            progress_percent=0.0, eta_seconds=12.0, started_at=timezone.now(),
        )
        progress = prediction_service.prediction_progress(job)
        self.assertIsNone(progress["progress_percent"])
        self.assertIsNone(progress["eta_seconds"])
        self.assertTrue(progress["is_active"])

    def test_apply_progress_switches_to_inference_phase(self):
        job = PredictionJob(sequence=SEQUENCE, status=PredictionJob.Status.RUNNING_MSA, progress_percent=50)
        job.apply_progress(ProgressEvent(step="Inference", phase=Phase.INFERENCE))
        self.assertEqual(job.status, PredictionJob.Status.RUNNING_INFERENCE)
        self.assertEqual(job.progress_percent, 50)  # unknown progress keeps the previous value


class ResultsTests(JobsDirTestCase):
    def test_samples_are_ranked_best_first(self):
        job = completed_job()
        self.assertEqual([s.sample for s in read_sample_scores(job.workspace)], [2, 1])

    def test_structures(self):
        job = completed_job()
        self.assertEqual(prediction_service.best_structure(job), CIF.replace("data_test", "data_sample_2"))
        self.assertEqual(prediction_service.sample_structure(job, 1), CIF.replace("data_test", "data_sample_1"))
        with self.assertRaises(ResourceNotFoundError):
            prediction_service.sample_structure(job, 9)


@mock.patch("predictor.tasks.start_attention_run")
class AttentionServiceTests(JobsDirTestCase):
    def test_requires_a_completed_job(self, start_attention_run):
        job = PredictionJob.objects.create(sequence=SEQUENCE, status=PredictionJob.Status.FAILED)
        with self.assertRaises(JobNotReadyError):
            attention_service.start_attention_run(job, [])

    def test_empty_selection_means_all_families(self, start_attention_run):
        run = attention_service.start_attention_run(completed_job(), [])
        self.assertEqual(len(run.layer_families), 4)
        start_attention_run.assert_called_once_with(run.id)

    def test_reads_blocks_and_cubes(self, start_attention_run):
        run = completed_attention_run(completed_job())
        self.assertEqual(attention_service.family_shape(run, "pairformer_self_attn"), [3, 4, 4])
        self.assertEqual(len(attention_service.block_weights(run, "pairformer_self_attn", 2)), 4)
        cube = attention_service.cube_points(run, "triangle_start", 0, threshold=0.0)
        self.assertEqual(len(cube["points"]), 4 ** 3)
        with self.assertRaises(ResourceNotFoundError):
            attention_service.cube_points(run, "pairformer_self_attn", 0, threshold=0.5)
        with self.assertRaises(ResourceNotFoundError):
            attention_service.block_weights(run, "pairformer_self_attn", 3)
