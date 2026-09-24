"""HTTP layer: `pages` renders HTML, `api` returns JSON for the front-end scripts.

Views stay thin: parse the request, call a service, turn the result or the
domain error into a response.
"""
