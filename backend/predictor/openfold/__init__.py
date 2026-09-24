"""Adapter between the Django app and the external OpenFold3-MLX installation.

This package is the only place that knows how OpenFold3-MLX is invoked
(command lines, environment), what it prints (log parsing) and what it writes
on disk (job directory layout, result files). The rest of the app talks to it
through plain Python values, so swapping the engine only touches this package.
"""
