# Documentation Deployment

NF2 documentation is set up for Read the Docs using MkDocs.

## Files Used

- repository-level RTD config: `.readthedocs.yml`
- MkDocs config: `mkdocs.yml`
- docs dependency list: `docs/requirements.txt`

## Read the Docs Setup

1. Connect the repository in Read the Docs.
2. Point RTD at the repository root.
3. Keep `.readthedocs.yml` enabled.
4. Use the default MkDocs build flow declared in that file.

## Local Verification

Before enabling or updating RTD, run:

```bash
pip install -r docs/requirements.txt
make docs
```

For local preview:

```bash
make docs-serve
```

## Recommended Next Step

Once RTD is connected, add the public docs URL to the README and package metadata.
