"""Django settings for OpenFold Studio.

Every machine-specific value comes from an environment variable with a sane
local default (see .env.example at the repository root).
"""

import os
from pathlib import Path

# backend/ (Django project) and the repository root that contains it.
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
VAR_DIR = PROJECT_ROOT / "var"  # runtime data, ignored by git


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default))


# --- Security -----------------------------------------------------------------

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "django-insecure-dev-only-change-me")
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")


# --- Application ----------------------------------------------------------------

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "predictor",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [FRONTEND_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

STATIC_URL = "static/"
STATICFILES_DIRS = [FRONTEND_DIR / "static"]


# --- Database -------------------------------------------------------------------

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": _env_path("DJANGO_DB_PATH", VAR_DIR / "db.sqlite3"),
    }
}
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]


# --- Internationalisation -------------------------------------------------------

LANGUAGE_CODE = "en-us"
TIME_ZONE = os.environ.get("DJANGO_TIME_ZONE", "Europe/Paris")
USE_I18N = True
USE_TZ = True


# --- OpenFold3-MLX integration --------------------------------------------------
# openfold-3-mlx is an external checkout (see scripts/setup_openfold.sh), run
# with its own interpreter as a subprocess.

OPENFOLD_PROJECT_DIR = _env_path("OPENFOLD_PROJECT_DIR", PROJECT_ROOT / "openfold-3-mlx")
OPENFOLD_PYTHON = _env_path("OPENFOLD_PYTHON", OPENFOLD_PROJECT_DIR / ".venv" / "bin" / "python3")
OPENFOLD_RUNNER_YAML = _env_path(
    "OPENFOLD_RUNNER_YAML",
    OPENFOLD_PROJECT_DIR / "examples" / "example_runner_yamls" / "mlx_runner.yml",
)
OPENFOLD_RUN_SCRIPT = OPENFOLD_PROJECT_DIR / "openfold3" / "run_openfold.py"
# Scripts of this repository executed with the OpenFold interpreter.
OPENFOLD_WORKER_DIR = PROJECT_ROOT / "openfold_worker"
# One sub-directory per prediction job (see predictor/openfold/workspace.py).
OPENFOLD_JOBS_DIR = _env_path("OPENFOLD_JOBS_DIR", VAR_DIR / "jobs")

OPENFOLD_SEED = 42
OPENFOLD_NUM_DIFFUSION_SAMPLES = 8
