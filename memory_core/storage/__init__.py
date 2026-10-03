"""Persistence layer.

Hard rule (architecture red line 2): nothing in this package may import a model
SDK or an HTTP client. Storage must work with no model and no network.
"""
