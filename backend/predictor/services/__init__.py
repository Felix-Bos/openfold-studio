"""Use cases of the application, called by the views.

Services enforce the business rules (one GPU computation at a time, a job
must be completed before its attention is analysed, ...), coordinate models,
the OpenFold adapter and background tasks, and raise `predictor.errors`
exceptions. They know nothing about HTTP: no request, no response, no template.
"""
