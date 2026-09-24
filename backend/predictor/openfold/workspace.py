"""On-disk layout of a prediction job: the single source of truth for every path.

Paths are never stored in the database: they are derived from the job/run id
and the configured jobs directory, so the project can be moved or cloned
without breaking existing jobs.

    <OPENFOLD_JOBS_DIR>/<job_id>/
    ├── query.json                         input given to OpenFold
    ├── run.log                            full prediction log
    ├── query_<job_id>/seed_<seed>/        OpenFold outputs
    │   ├── query_<job_id>_seed_<seed>_sample_<n>_confidences_aggregated.json
    │   ├── query_<job_id>_seed_<seed>_sample_<n>_confidences.json
    │   └── query_<job_id>_seed_<seed>_sample_<n>_model.cif
    └── attention/<run_id>/                one directory per attention run
        ├── run.log
        ├── meta.json
        ├── <family>.npz
        └── <family>_cube.npz              triangle families only
"""

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from django.conf import settings


@dataclass(frozen=True)
class AttentionWorkspace:
    root: Path

    @property
    def log_file(self) -> Path:
        return self.root / "run.log"

    @property
    def meta_file(self) -> Path:
        return self.root / "meta.json"

    def weights_file(self, family: str) -> Path:
        return self.root / f"{family}.npz"

    def cube_file(self, family: str) -> Path:
        return self.root / f"{family}_cube.npz"


@dataclass(frozen=True)
class JobWorkspace:
    job_id: UUID
    root: Path

    @classmethod
    def for_job(cls, job_id: UUID) -> "JobWorkspace":
        return cls(job_id=job_id, root=Path(settings.OPENFOLD_JOBS_DIR) / job_id.hex)

    @property
    def query_name(self) -> str:
        return f"query_{self.job_id.hex}"

    @property
    def query_file(self) -> Path:
        return self.root / "query.json"

    @property
    def log_file(self) -> Path:
        return self.root / "run.log"

    @property
    def seed(self) -> int:
        return settings.OPENFOLD_SEED

    @property
    def seed_dir(self) -> Path:
        return self.root / self.query_name / f"seed_{self.seed}"

    def _sample_prefix(self, sample: int) -> str:
        return f"{self.query_name}_seed_{self.seed}_sample_{sample}"

    def confidence_files(self) -> list[Path]:
        return sorted(self.seed_dir.glob("*_confidences_aggregated.json"))

    def structure_file(self, sample: int) -> Path:
        return self.seed_dir / f"{self._sample_prefix(sample)}_model.cif"

    def scores_file(self, sample: int) -> Path:
        """Aggregated scores: avg pLDDT, pTM, gPDE, disorder, clash, ranking."""
        return self.seed_dir / f"{self._sample_prefix(sample)}_confidences_aggregated.json"

    def confidence_details_file(self, sample: int) -> Path:
        """Per-atom pLDDT and the residue x residue PDE matrix."""
        return self.seed_dir / f"{self._sample_prefix(sample)}_confidences.json"

    def attention(self, run_id: UUID) -> AttentionWorkspace:
        return AttentionWorkspace(root=self.root / "attention" / run_id.hex)

    def create(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
