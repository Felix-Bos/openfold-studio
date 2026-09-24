import shutil
import tempfile
from pathlib import Path

from django.test import TestCase, override_settings


class JobsDirTestCase(TestCase):
    """Runs each test with OPENFOLD_JOBS_DIR pointing to a fresh temporary directory."""

    def setUp(self):
        super().setUp()
        self.jobs_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.jobs_dir, ignore_errors=True)
        override = override_settings(OPENFOLD_JOBS_DIR=self.jobs_dir)
        override.enable()
        self.addCleanup(override.disable)
