from unittest import mock

from django.urls import reverse

from predictor.models import PredictionJob

from .base import JobsDirTestCase
from .factories import CIF, SEQUENCE, completed_attention_run, completed_job


class PageTests(JobsDirTestCase):
    def test_index_lists_jobs(self):
        completed_job()
        response = self.client.get(reverse("predictor:index"))
        self.assertContains(response, SEQUENCE)

    def test_invalid_sequence_shows_form_error(self):
        response = self.client.post(reverse("predictor:index"), {"sequence": "MK1"})
        self.assertContains(response, "rejected characters")
        self.assertEqual(PredictionJob.objects.count(), 0)

    @mock.patch("predictor.tasks.start_prediction")
    def test_valid_sequence_redirects_to_job(self, start_prediction):
        response = self.client.post(reverse("predictor:index"), {"sequence": "mkt ayi"})
        job = PredictionJob.objects.get()
        self.assertEqual(job.sequence, "MKTAYI")
        self.assertRedirects(response, reverse("predictor:job_detail", args=[job.id]))

    def test_job_and_explorer_pages_render(self):
        job = completed_job()
        run = completed_attention_run(job)
        self.assertContains(self.client.get(reverse("predictor:job_detail", args=[job.id])), "Diffusion samples")
        explorer = self.client.get(reverse("predictor:attention_explorer", args=[job.id, run.id]))
        self.assertContains(explorer, "data-attention-explorer")

    def test_architecture_page_renders(self):
        self.assertEqual(self.client.get(reverse("predictor:architecture")).status_code, 200)


class ApiTests(JobsDirTestCase):
    def setUp(self):
        super().setUp()
        self.job = completed_job()
        self.run = completed_attention_run(self.job)

    def url(self, name, *args):
        return reverse(f"api:{name}", args=[self.job.id, self.run.id, *args])

    def test_job_status(self):
        data = self.client.get(reverse("api:job_status", args=[self.job.id])).json()
        self.assertEqual(data["status"], "completed")
        self.assertFalse(data["is_active"])

    def test_sample_structure(self):
        response = self.client.get(reverse("api:sample_structure", args=[self.job.id, 2]))
        self.assertEqual(response.content, CIF.replace("data_test", "data_sample_2").encode())

    def test_families_and_blocks(self):
        families = self.client.get(self.url("attention_families")).json()["families"]
        self.assertEqual(families["pairformer_self_attn"]["shape"], [3, 4, 4])
        self.assertEqual(families["pairformer_self_attn"]["summary"]["blocks"], 3)
        block = self.client.get(self.url("attention_block", "pairformer_self_attn", 1)).json()
        self.assertEqual(len(block["weights"]), 4)
        self.assertIn("local_mass", block["stats"])

    def test_sample_confidence(self):
        data = self.client.get(reverse("api:sample_confidence", args=[self.job.id, 2])).json()
        self.assertEqual([r["index"] for r in data["residues"]], [1, 2, 3, 4])
        self.assertEqual(data["low_confidence_regions"], [])
        self.assertEqual(len(data["pde"]["values"]), 4)

    def test_cube_threshold_is_parsed(self):
        data = self.client.get(self.url("attention_cube", "triangle_start", 0) + "?threshold=2").json()
        self.assertEqual(data["threshold"], 1.0)

    def test_errors_are_json(self):
        response = self.client.get(self.url("attention_block", "unknown", 0))
        self.assertEqual(response.status_code, 404)
        self.assertIn("error", response.json())
