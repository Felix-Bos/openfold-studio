"""Stop storing absolute file paths: they are now derived from the job/run id.

Stored paths broke every existing job as soon as the project directory was
moved. The job directory layout lives in predictor/openfold/workspace.py.
"""

from django.db import migrations


def strip_cif_paths(apps, schema_editor):
    PredictionJob = apps.get_model("predictor", "PredictionJob")
    for job in PredictionJob.objects.exclude(result_summary=None):
        for sample in job.result_summary:
            sample.pop("cif_path", None)
        job.save(update_fields=["result_summary"])


class Migration(migrations.Migration):
    dependencies = [
        ("predictor", "0002_attentionrun"),
    ]

    operations = [
        migrations.RunPython(strip_cif_paths, migrations.RunPython.noop),
        migrations.RemoveField(model_name="attentionrun", name="log_path"),
        migrations.RemoveField(model_name="attentionrun", name="output_dir"),
        migrations.RemoveField(model_name="predictionjob", name="log_path"),
        migrations.RemoveField(model_name="predictionjob", name="output_dir"),
        migrations.RemoveField(model_name="predictionjob", name="query_json_path"),
    ]
