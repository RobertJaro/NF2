"""Generate compact Markdown reference pages from NF2 reference metadata."""

from __future__ import annotations

from pathlib import Path

from nf2.reference import CLI_COMMANDS, CONFIG_OPTIONS, DATASET_TYPES, EXPORT_METRICS, QUALITY_METRICS


def _table(headers, rows):
    def _escape(value):
        return str(value).replace("|", "\\|")

    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(_escape(cell) for cell in row) + " |" for row in rows)
    return "\n".join(out)


def generate(root: str | Path):
    root = Path(root)
    target = root / "generated"
    target.mkdir(exist_ok=True)

    (target / "configuration_reference.md").write_text(
        "# Full YAML Reference\n\n"
        "This page is generated from `nf2.reference` and mirrors the public v0.4 YAML schema.\n\n"
        + _table(["Key", "Type", "Default", "Description"], CONFIG_OPTIONS)
        + "\n",
        encoding="utf-8",
    )

    (target / "datasets_reference.md").write_text(
        "# Dataset And Sampler Reference\n\n"
        "Dataset entries are used under `data.boundaries`, `data.validation`, `data.sampler`, and `data.samplers`.\n\n"
        + _table(["Type", "Role", "Description"], DATASET_TYPES)
        + "\n",
        encoding="utf-8",
    )

    cli_sections = ["# CLI Reference\n"]
    for command, summary, options in CLI_COMMANDS:
        cli_sections.append(f"## `{command}`\n\n{summary}\n\n")
        cli_sections.append(_table(["Option", "Default", "Description"], options))
        cli_sections.append("\n")
    (target / "cli_reference.md").write_text("\n".join(cli_sections), encoding="utf-8")

    (target / "export_metrics_reference.md").write_text(
        "# Export And Metrics Reference\n\n"
        "## Export Metrics\n\n"
        + _table(["Metric", "Description"], EXPORT_METRICS)
        + "\n\n## Quality Metrics\n\n"
        + _table(["Metric", "Description"], QUALITY_METRICS)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    generate(Path(__file__).parent)
