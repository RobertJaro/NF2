.PHONY: docs docs-serve

docs:
	mkdocs build --strict

docs-serve:
	mkdocs serve
