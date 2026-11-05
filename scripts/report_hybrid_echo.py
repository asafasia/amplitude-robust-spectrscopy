"""Validate hybrid-search candidates with exact reuse of prior report caches."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import report_flexible_50mhz_echo as report  # noqa: E402

SOURCE = ROOT / "outputs/flexible_echo_50mhz_20261101"
OUT = ROOT / "outputs/hybrid_echo_50mhz_20261101"


class HybridCache(report.Cache):
    def locate(self, kind, identity, allow_old=False):
        encoded = json.dumps(identity, sort_keys=True)
        name = f"validate_{kind}_{hashlib.sha256(encoded.encode()).hexdigest()}.npz"
        target = self.directory / name
        prior = SOURCE / "evaluations" / name
        if not target.exists() and prior.exists():
            with np.load(prior) as saved:
                if json.loads(str(saved["identity"])) != json.loads(encoded):
                    raise RuntimeError(f"Previous flexible cache identity mismatch: {prior}")
            shutil.copy2(prior, target)
            self.imports.append(dict(source=str(prior), destination=str(target),
                                     sha256=report.sha(target), exact_identity_match=True))
            report.save_json(self.ledger_path, self.imports)
        return super().locate(kind, identity, allow_old=allow_old)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=OUT / "selected_candidates.json")
    parser.add_argument("--stage", choices=("validate", "render", "all"), default="all")
    args = parser.parse_args()
    report.Cache = HybridCache
    if args.stage != "render":
        report.validate(argparse.Namespace(candidates=args.candidates, output=OUT,
                                           amplitude_step_mhz=.5, skip_bandwidth=False))
        results = json.loads((OUT / "results.json").read_text())
        for case in results["cases"].values():
            if (case.get("selection_record") or {}).get("track") == "local":
                local = case["validation"]["fine_grid_50mhz"]["local_shoulders"]
                passed = local["accepted"] and local["contrast"] >= .4
                case["constraints"]["extra_local_contrast_pass"] = bool(passed)
                case["constraints"]["accepted"] &= bool(passed)
        results["metadata"]["accepted_new_candidates"] = [n for n, c in results["cases"].items()
                                                         if not c["control"] and c["constraints"]["accepted"]]
        for path in (Path(__file__), ROOT / "scripts/hybrid_echo_search.py",
                     ROOT / "echospec/analysis/notch_constraints.py"):
            results["metadata"]["source_sha256"][str(path.relative_to(ROOT))] = report.sha(path)
        report.save_json(OUT / "results.json", results)
    if args.stage != "validate":
        results = json.loads((OUT / "results.json").read_text())
        dense_rows, audit_inputs = [], {}
        for audit_path in sorted([*OUT.glob("dense_band*.json"), *OUT.glob("critical_*.json")]):
            audit = json.loads(audit_path.read_text())
            for name, checked in audit.get("cases", {}).items():
                case = results["cases"].get(name)
                if case is None or checked["pulse"] != case["pulse"]:
                    continue
                audit_inputs[audit_path.name] = report.sha(audit_path)
                passed = checked["constraints"]["accepted"]
                case["constraints"][f"{audit_path.stem}_pass"] = bool(passed)
                case["constraints"]["accepted"] &= bool(passed)
                a, m = audit["amplitudes_mhz"], checked["margins"]
                dense_rows.append(f'<tr><td>{html.escape(case["label"])}</td>'
                                  f'<td><a href="{audit_path.name}">{a[0]:g}–{a[-1]:g} MHz, {len(a)} points</a></td>'
                                  f'<td>{m["width_gain_khz"]:.6f} kHz at {m["width_gain_at_mhz"]:g} MHz</td>'
                                  f'<td>{m["contrast_gain"]:.7f} at {m["contrast_gain_at_mhz"]:g} MHz</td>'
                                  f'<td>{"Pass" if passed else "Fail"}</td></tr>')
        results["metadata"]["accepted_new_candidates"] = [n for n, c in results["cases"].items()
                                                         if not c["control"] and c["constraints"]["accepted"]]
        results["metadata"]["dense_audit_inputs_sha256"] = audit_inputs
        report.save_json(OUT / "results.json", results)
        best = {name: dict(pulse=c["pulse"], actual_peak_mhz=50.,
                           unit_waveform_peak=c["unit_waveform_peak"],
                           multiplier_for_50mhz=c["scale_for_actual_50mhz"],
                           metrics_50mhz=c["validation"]["fine_grid_50mhz"],
                           constraints=c["constraints"])
                for name, c in results["cases"].items()
                if name.startswith("hybrid_") and c["constraints"]["accepted"]}
        report.save_json(OUT / "best_pulses.json", dict(
            model=dict(t1_us=51.24, t2_us=7.31, anharmonicity_mhz=-216.),
            waveform_convention="u = I - i Q; multiply pulse.drive(t_us) by multiplier_for_50mhz. Scale the same waveform linearly for other peaks.",
            pulse_class="echospec.simulation.flexible_echo.FlexibleEchoPulse", candidates=best))
        report.render(OUT)
        path = OUT / "report.html"
        page = path.read_text()
        paragraph = """<div class="callout"><b>Hybrid search:</b> differential evolution explores a population of smooth single-echo controls, followed by constrained SLSQP refinement.
The local objective uses a variable target width and fixed-center half-depth inequalities; actual extracted FWHM and contrast remain the final acceptance metrics.
Gradients are finite differences through the deterministic Lindblad simulation, not adjoint/GOAT derivatives.
The family has 26 adjustable parameters, including a second Gaussian phase term and DRAG beta.
The primary track retains the original contrast definition; the local-contrast track additionally requires nearby-shoulder depth at least 0.4 at 50 MHz.
Final no-regression tolerances are zero on the sampled amplitude grid. This is a finite-family search, not a global optimality certificate.</div>"""
        paragraph += '<p><a href="best_pulses.json">Exact selected pulse parameters and normalization</a></p>'
        summary_path = OUT / "search_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            paragraph += (f'<p>The search used {summary["total_evaluations"]:,} optimization evaluations '
                          f'across {len(summary["campaigns"])} campaigns. '
                          '<a href="search_summary.json">Search settings, outcomes, and source snapshots</a>.</p>')
        if dense_rows:
            paragraph += """<h2>Checks between plotted amplitude samples</h2>
<p>Each row is a direct simulation against the original polynomial on the same grid. Positive margins mean improvement; no regression tolerance is allowed. These sampled checks are not a mathematical guarantee at every real-valued amplitude or under model uncertainty.</p>
<table><tr><th>Pulse</th><th>Amplitude grid</th><th>Smallest linewidth improvement</th><th>Smallest contrast improvement</th><th>Result</th></tr>""" + "".join(dense_rows) + "</table>"
        page = page.replace("<title>Flexible echo at 50 MHz</title>", "<title>Hybrid echo search at 50 MHz</title>")
        page = page.replace("<h1>Flexible pulse and phase shaping at 50 MHz</h1>", "<h1>Hybrid pulse and phase search at 50 MHz</h1>")
        first_heading_end = page.find("</h1>") + len("</h1>")
        page = page[:first_heading_end] + paragraph + page[first_heading_end:]
        path.write_text(page)
        for link in re.findall(r'(?:href|src)="([^"]+)"', page):
            if not (OUT / link).is_file():
                raise RuntimeError(f"Missing hybrid report link: {link}")
        report.save_json(OUT / "hybrid_render_provenance.json", {
            "generator": str(Path(__file__).relative_to(ROOT)), "generator_sha256": report.sha(Path(__file__)),
            "base_generator_sha256": report.sha(ROOT / "scripts/report_flexible_50mhz_echo.py"),
            "inputs": {**{n: report.sha(OUT / n) for n in ("results.json", "validated_spectra.npz")}, **audit_inputs},
            "outputs": {n: report.sha(OUT / n) for n in ("report.html", "maps_preferred.png", "maps_all.png")},
        })


if __name__ == "__main__":
    main()
