"""Render a separate S3 preview without changing the manuscript figure."""

from scripts.make_supplemental_duration_resolution import ROOT, render

if __name__ == "__main__":
    render(
        ROOT / "figures/paper/previews/duration_resolution_split",
        "s03_duration_resolution_preview",
        preview=True,
    )
