"""Live Jupyter controls for cutoff exploration."""

from __future__ import annotations

from pathlib import Path

import ipywidgets as widgets
from IPython.display import Markdown, clear_output, display

from .aggregate import AggregateCampaign
from .analysis import analyze_run, best_amplitude_index
from .data import Campaign
from .plotting import plot_aggregate_cutoff, plot_cutoff_dashboard


class CutoffExplorer:
    """Campaign/cutoff/amplitude explorer for raw numerical campaigns."""

    def __init__(self, campaigns: list[Campaign]):
        if not campaigns:
            raise ValueError("At least one raw-backed campaign is required.")
        self.campaigns = campaigns
        self._updating = False
        self.campaign = widgets.Dropdown(
            options=[(item.label, index) for index, item in enumerate(campaigns)],
            value=0,
            description="Campaign:",
            layout=widgets.Layout(width="960px"),
            style={"description_width": "90px"},
        )
        self.cutoff = widgets.SelectionSlider(
            options=[("loading", 0.0)],
            description="Cutoff:",
            continuous_update=False,
            layout=widgets.Layout(width="960px"),
            style={"description_width": "90px"},
        )
        self.amplitude = widgets.SelectionSlider(
            options=[("loading", 0)],
            description="Rabi (MHz):",
            continuous_update=False,
            layout=widgets.Layout(width="960px"),
            style={"description_width": "90px"},
        )
        self.output = widgets.Output()
        self.controls = widgets.VBox([self.campaign, self.cutoff, self.amplitude])

        self.campaign.observe(self._campaign_changed, names="value")
        self.cutoff.observe(self._cutoff_changed, names="value")
        self.amplitude.observe(self._render, names="value")
        self._updating = True
        self._update_cutoffs()
        self._update_amplitudes()
        self._updating = False

    def _selected_campaign(self) -> Campaign:
        return self.campaigns[int(self.campaign.value)]

    def _selected_run(self) -> Path:
        return self._selected_campaign().runs[float(self.cutoff.value)]

    def _update_cutoffs(self) -> None:
        campaign = self._selected_campaign()
        self.cutoff.options = [(f"{value:.4g}", value) for value in campaign.cutoffs]
        self.cutoff.value = campaign.cutoffs[len(campaign.cutoffs) // 2]

    def _update_amplitudes(self) -> None:
        analysis = analyze_run(str(self._selected_run()))
        self.amplitude.options = [
            (f"{rabi:.4g}", index) for index, rabi in enumerate(analysis.raw.rabi_mhz)
        ]
        self.amplitude.value = best_amplitude_index(analysis)

    def _campaign_changed(self, change) -> None:
        if change.get("name") != "value":
            return
        self._updating = True
        self._update_cutoffs()
        self._update_amplitudes()
        self._updating = False
        self._render()

    def _cutoff_changed(self, change) -> None:
        if self._updating or change.get("name") != "value":
            return
        self._updating = True
        self._update_amplitudes()
        self._updating = False
        self._render()

    def _render(self, change=None) -> None:
        if self._updating:
            return
        with self.output:
            clear_output(wait=True)
            analysis = analyze_run(str(self._selected_run()))
            figure, _ = plot_cutoff_dashboard(analysis, int(self.amplitude.value))
            display(figure)
            display(
                Markdown(
                    f"**Accepted OPX fits:** {analysis.valid.sum()}/{len(analysis.valid)}  \\n"
                    f"**Raw source:** `{analysis.raw.run_dir}`"
                )
            )

    def display(self) -> None:
        """Display raw-campaign controls and the current selection."""
        display(self.controls, self.output)
        self._render()


class AggregateCutoffExplorer:
    """Cutoff-only explorer for the retained July 12 aggregate campaign."""

    def __init__(self, campaign: AggregateCampaign):
        self.campaign = campaign
        self.cutoff = widgets.SelectionSlider(
            options=[(f"{value:.4g}", value) for value in campaign.cutoffs],
            value=campaign.cutoffs[len(campaign.cutoffs) // 2],
            description="Cutoff:",
            continuous_update=False,
            layout=widgets.Layout(width="960px"),
            style={"description_width": "90px"},
        )
        self.output = widgets.Output()
        self.cutoff.observe(self._render, names="value")

    def _render(self, change=None) -> None:
        with self.output:
            clear_output(wait=True)
            figure, _ = plot_aggregate_cutoff(
                self.campaign,
                float(self.cutoff.value),
                resolution_vmin=0.1,
                resolution_vmax=1.0,
            )
            display(figure)
            display(Markdown(f"**Campaign source:** `{self.campaign.directory}`"))

    def display(self) -> None:
        """Display the 20-point cutoff slider and current campaign view."""
        display(self.cutoff, self.output)
        self._render()
