"""Render an offline interactive four-panel spectroscopy comparison."""

from __future__ import annotations

import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots

T2_US = 7.31
T2_FWHM_KHZ = 1000 / (math.pi * T2_US)


def write_interactive_sweeps(path, amplitudes, detuning_mhz, populations, rows, cases):
    """Keep all heatmaps on one color axis; slider changes display only."""
    titles = ["Hard cutoff · β = 0", "Hard cutoff · β = −0.5",
              "Polynomial p = 5 + cutoff subtraction · β = −0.5",
              "C² subtraction · β = −0.5"]
    figure = make_subplots(rows=2, cols=2, subplot_titles=titles,
                          horizontal_spacing=.10, vertical_spacing=.16)
    for index, (population, case) in enumerate(zip(populations, cases, strict=True)):
        row, col = index // 2 + 1, index % 2 + 1
        figure.add_trace(go.Heatmap(
            x=detuning_mhz * 1000, y=amplitudes, z=population[1], coloraxis="coloraxis",
            hovertemplate="Detuning: %{x:.1f} kHz<br>A: %{y:.2f} MHz"
                          "<br>Pₑ: %{z:.5f}<extra></extra>"), row=row, col=col)
        valid = [r for r in rows if r["case"] == case and r["accepted"]]
        for side in ["left", "right"]:
            figure.add_trace(go.Scatter(
                x=[r[f"{side}_half_depth_khz"] for r in valid],
                y=[r["amplitude_mhz"] for r in valid], mode="markers",
                customdata=[r["fwhm_hz"] / 1000 for r in valid],
                marker=dict(size=4, color="white", line=dict(color="#333", width=.5)),
                name="Half-depth crossings", legendgroup="fwhm",
                showlegend=index == 0 and side == "left",
                hovertemplate="Half-depth: %{x:.2f} kHz<br>A: %{y:.2f} MHz"
                              "<br>FWHM: %{customdata:.2f} kHz<extra></extra>"),
                row=row, col=col)
        figure.update_xaxes(title_text="Drive detuning (kHz)", range=[-500, 500],
                            tickvals=[-500, -250, 0, 250, 500], row=row, col=col)
        figure.update_yaxes(title_text="Rabi amplitude A (MHz)", range=[0, 100],
                            row=row, col=col)
        for direction in [-1, 1]:
            figure.add_vline(
                x=direction * T2_FWHM_KHZ / 2, row=row, col=col,
                line_color="#ff6b9d", line_width=1.5, line_dash="dash",
                name=f"T₂ reference FWHM = {T2_FWHM_KHZ:.2f} kHz",
                legendgroup="t2", showlegend=index == 0 and direction == -1)
    figure.update_layout(
        template="plotly_white", height=920, margin=dict(l=75, r=85, t=50, b=80),
        font=dict(family="Arial, sans-serif", size=12),
        coloraxis=dict(colorscale="Viridis", cmin=0, cmax=1, cauto=False,
                       colorbar=dict(title="Pₑ", thickness=18, len=.94)),
        legend=dict(orientation="h", x=.5, xanchor="center", y=-.09),
        uirevision="fixed-sweeps")
    plot = figure.to_html(full_html=False, include_plotlyjs=True,
                          div_id="sweep-plots", config=dict(responsive=True,
                          displaylogo=False, toImageButtonOptions=dict(
                              format="png", filename="echo_sweeps", scale=2)))
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Interactive echo spectroscopy comparison</title>
<style>
body{margin:0;background:#f5f7fa;color:#172333;font:15px/1.5 Arial,sans-serif}
main{max-width:1350px;margin:auto;padding:22px}h1{font-size:25px;margin:0 0 6px}
p{margin:6px 0 16px}.controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap;
position:sticky;top:0;z-index:10;background:#fff;padding:14px 18px;
border:1px solid #d9dfe8;border-radius:9px;box-shadow:0 3px 12px #00000008}
input[type=range]{flex:1;min-width:180px;accent-color:#256e9d}
input[type=number]{width:70px;padding:6px;font:inherit;border:1px solid #aeb9c7;
border-radius:5px}button{padding:7px 12px;background:#edf3f9;border:1px solid #b8c7d8;
border-radius:5px;cursor:pointer;font:inherit}.hint{font-size:13px;color:#536174}
.plot{margin-top:16px;background:white;border-radius:9px;overflow:hidden}
a{color:#176eaa}footer{font-size:13px;color:#536174;margin-top:15px}
</style></head><body><main>
<h1>Echo spectroscopy — four pulse comparisons</h1>
<p>30 µs · cutoff 0.001 · 16 ns echo transition · 100 amplitude points · ±500 kHz</p>
<div class="controls">
<label for="vmax"><b>Shared color-bar maximum</b></label>
<input id="vmax" type="range" min="0.05" max="1" step="0.01" value="1"
 aria-label="Shared color-bar maximum">
<input id="vmax-number" type="number" min="0.05" max="1" step="0.01" value="1.00"
 aria-label="Color-bar maximum value">
<button id="reset" type="button">Reset to 1</button>
<span id="scale-status" class="hint" aria-live="polite">Pₑ scale: 0–1.00</span>
</div>
<p class="hint">The slider updates all four plots together. Values above vmax use the
 brightest color. Populations and FWHM markers do not change. Hover to read values;
 drag to zoom and double-click to reset the axes.</p>
<div class="plot">__PLOT__</div>
<footer>White dots are the two half-depth crossings; their separation is the FWHM.
Pink dashed vertical lines are at ±21.77 kHz around zero detuning. Their separation
is the weak-drive T₂ reference FWHM, 1/(πT₂) = 43.54 kHz. They stay fixed as vmax changes.
No dip is defined at zero amplitude. Background: |detuning| = 400–500 kHz.
All four panels use the same color scale without row normalization.
T₁ = 51.24 µs, T₂ = 7.31 µs, α = −216 MHz.
<p><a href="sweeps_2d.pdf">Static PDF</a> ·
<a href="sweeps_2d_fwhm_points.csv">FWHM data</a> ·
<a href="sweeps_2d_provenance.json">Parameters and provenance</a></p></footer>
</main><script>
const slider = document.getElementById('vmax');
const numeric = document.getElementById('vmax-number');
const graph = document.getElementById('sweep-plots');
function setVmax(raw) {
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) return;
  const value = Math.max(0.05, Math.min(1, parsed));
  slider.value = value;
  numeric.value = value.toFixed(2);
  document.getElementById('scale-status').textContent = 'Pₑ scale: 0–' + value.toFixed(2);
  return Plotly.relayout(graph, {'coloraxis.cmin': 0, 'coloraxis.cmax': value,
                                'coloraxis.cauto': false});
}
slider.addEventListener('input', () => setVmax(slider.value));
numeric.addEventListener('change', () => setVmax(numeric.value));
document.getElementById('reset').addEventListener('click', () => setVmax(1));
</script></body></html>"""
    path.write_text(html.replace("__PLOT__", plot))
