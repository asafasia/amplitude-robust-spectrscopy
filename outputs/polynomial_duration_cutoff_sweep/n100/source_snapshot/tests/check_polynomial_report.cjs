// Code-level test of our report's controls and data wiring. This does not
// launch a browser or substitute for a visual browser rendering check.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(process.argv[2], 'utf8');
const elementIds = new Set([...source.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]));
const payload = source.match(/<script id="studyData" type="application\/json">([\s\S]*?)<\/script>/)[1];
const program = [...source.matchAll(/<script>([\s\S]*?)<\/script>/g)].at(-1)[1];
assert.ok(!source.includes('id="templateNotice"'));
assert.ok(source.includes('<main id="reportContent">'));
// Opening the source template should show a report link, not attempt to parse
// placeholder data or leave the user with an empty set of charts.
const template = fs.readFileSync(path.join(__dirname, '../scripts/templates/polynomial_duration_report.html'), 'utf8');
assert.ok(template.includes('id="templateNotice"'));
assert.ok(template.includes('<main id="reportContent" hidden>'));
assert.ok(template.includes('href="../../outputs/polynomial_duration_cutoff_sweep/n100/report.html"'));
const templateProgram = [...template.matchAll(/<script>([\s\S]*?)<\/script>/g)].at(-1)[1];
assert.doesNotThrow(() => vm.runInNewContext(templateProgram, {
  document: {getElementById() { return {textContent: '__PAYLOAD__'}; }}
}));
const elements = new Map(), plots = new Map(), schemaChecks = [];
function element(id) {
  assert.ok(elementIds.has(id), `Missing HTML element: ${id}`);
  if (!elements.has(id)) elements.set(id, {
    value: id === 'mapMetric' ? 'fwhm_hz' : '0', textContent: '', innerHTML: '',
    options: [], listeners: {}, classList: { add() {} },
    add(option) { this.options.push(option); },
    addEventListener(event, callback) { this.listeners[event] = callback; }
  });
  return elements.get(id);
}
element('studyData').textContent = payload;
const sandbox = {
  document: { getElementById: element }, window: {},
  Plotly: { react(id, data, layout) { plots.set(id, {data, layout}); } },
  Option: function(text, value) { this.text = text; this.value = value; },
  atob: text => Buffer.from(text, 'base64').toString('binary'), console,
};
vm.createContext(sandbox);
vm.runInContext(program, sandbox);
const data = JSON.parse(payload);
const buffers = data.cases.map(c => Buffer.from(c.population_base64, 'base64'));
const nd = data.detuning.length, na = data.amps.length, nc = data.cutoffs.length;
const referenceHz = 1 / (Math.PI * data.configuration.t2_us * 1e-6);
assert.equal(sandbox.window.reportReady, true);
assert.equal(plots.size, 7 + data.cases.length);
assert.equal(element('widthUnit').value, 't2');
assert.equal(element('amplitude').options.length, na);
assert.equal(plots.get('spectrum').data.length, data.cases.length);
const capture = () => schemaChecks.push(...plots.values());
capture();
function near(actual, expected) {
  assert.ok(Math.abs(actual - expected) < 1e-10 * Math.max(1, Math.abs(expected)),
    `${actual} != ${expected}`);
}
function verifyPopulations(ci, ai) {
  assert.equal(+element('populationCutoff').value, ci);
  assert.equal(element('populationCutoffValue').textContent, element('cutoffValue').textContent);
  data.cases.forEach((c, index) => {
    const bytes = buffers[index];
    assert.equal(bytes.length, nc * na * nd * 4);
    const points = plots.get('spectrum').data[index].y;
    assert.equal(points.length, nd);
    const offset = (ci * na + ai) * nd;
    for (let k = 0; k < nd; k++)
      assert.equal(points[k], bytes.readFloatLE(4 * (offset + k)));
    const map = plots.get(`populationMap${index}`);
    const heat = map.data[0];
    assert.equal(heat.type, 'heatmap');
    assert.equal(heat.zmin, 0);
    assert.equal(heat.zmax, 1);
    assert.equal(heat.zsmooth, false);
    assert.equal(JSON.stringify(heat.x), JSON.stringify(data.detuning.map(v => v * 1000)));
    assert.equal(JSON.stringify(heat.y), JSON.stringify(data.amps));
    assert.equal(heat.z.length, na);
    assert.equal(map.layout.shapes[0].y0, data.amps[ai]);
    const markers = map.layout.shapes.slice(1);
    assert.equal(markers.length, 2);
    markers.forEach((marker, j) => {
      assert.equal(marker.type, 'line');
      assert.equal(marker.xref, 'x');
      assert.equal(marker.yref, 'paper');
      assert.equal(marker.y0, 0);
      assert.equal(marker.y1, 1);
      near(marker.x0, (j === 0 ? -1 : 1) * referenceHz / 2000);
      assert.equal(marker.x0, marker.x1);
    });
    near(markers[1].x0 - markers[0].x0, referenceHz / 1000);
    for (let j = 0; j < na; j++) {
      assert.equal(heat.z[j].length, nd);
      for (let k = 0; k < nd; k++)
        assert.equal(heat.z[j][k], bytes.readFloatLE(4 * ((ci * na + j) * nd + k)));
    }
  });
}
function verifyWidths(unit) {
  const factor = unit === 't2' ? 1 / referenceHz : .001;
  const label = unit === 't2' ? 'FWHM / Γ₂' : 'FWHM (kHz)';
  const ci = +element('cutoff').value, ai = +element('amplitude').value;
  assert.equal(element('pointWidthHeading').textContent, label);
  assert.equal(element('bestWidthHeading').textContent, label);
  assert.equal(element('robustWidthHeading').textContent, `Worst ${label}`);
  assert.equal(element('mapWidthOption').textContent, label);
  assert.equal(plots.get('widthCutoff').layout.yaxis.title.text, label);
  assert.equal(plots.get('tradeoff').layout.xaxis.title.text, label);
  assert.equal(plots.get('robustWidth').layout.yaxis.title.text, label);
  near(plots.get('widthCutoff').layout.shapes[0].y0, referenceHz * factor);
  near(plots.get('robustWidth').layout.shapes[0].y0, referenceHz * factor);
  assert.equal(element('overviewT2').hidden, unit !== 't2');
  assert.equal(element('overviewKhz').hidden, unit === 't2');
  assert.equal(element('overviewPng').href,
    unit === 't2' ? 'duration_comparison.png' : 'duration_comparison_khz.png');
  assert.equal(element('overviewLink').href, element('overviewPng').href);
  assert.equal(element('overviewPdf').href, element('overviewPng').href.replace('.png', '.pdf'));
  data.cases.forEach((c, i) => {
    const metric = c.metrics;
    for (let j = 0; j < nc; j++) {
      const actual = plots.get('widthCutoff').data[i].y[j];
      if (metric.accepted[j][ai]) near(actual, metric.fwhm_hz[j][ai] * factor);
      else assert.equal(actual, null);
    }
    if (metric.accepted[ci][ai]) assert.ok(element('pointRows').innerHTML.includes(
      (metric.fwhm_hz[ci][ai] * factor).toFixed(2)));
    if (c.robust) assert.ok(element('robustRows').innerHTML.includes(
      (c.robust.worst_fwhm_khz * 1000 * factor).toFixed(2)));
  });
  const trade = plots.get('tradeoff').data.filter(t => t.showlegend !== false);
  trade.forEach((t, i) => {
    const c = data.cases[i];
    const expected = data.cutoffs.flatMap((_, j) => c.metrics.accepted[j][ai] ?
      [c.metrics.fwhm_hz[j][ai] * factor] : []);
    assert.equal(JSON.stringify(t.x), JSON.stringify(expected));
  });
}
verifyPopulations(+element('cutoff').value, +element('amplitude').value);
verifyWidths('t2');
// Unit changes affect width displays only; population and SNR values remain unchanged.
const snrBefore = JSON.stringify(plots.get('snrCutoff').data);
const populationsBefore = JSON.stringify(plots.get('populationMap0').data);
for (const unit of ['khz', 't2']) {
  element('widthUnit').value = unit;
  element('widthUnit').listeners.change();
  verifyWidths(unit);
  capture();
  assert.equal(JSON.stringify(plots.get('snrCutoff').data), snrBefore);
  assert.equal(JSON.stringify(plots.get('populationMap0').data), populationsBefore);
  for (let i = 0; i < data.cases.length; i++) {
    element('mapLength').value = String(i);
    for (const key of ['fwhm_hz', 'contrast_snr', 'snr_per_mhz', 'max_final_pf']) {
      element('mapMetric').value = key;
      element('mapMetric').listeners.change();
      const plot = plots.get('metricMap');
      schemaChecks.push(plot);
      const z = plot.data[0].z, c = data.cases[i];
      assert.equal(z.length, nc);
      assert.equal(z[0].length, na);
      for (const ci of [0, Math.floor(nc / 2), nc - 1]) {
        for (const ai of [0, Math.floor(na / 2), na - 1]) {
          const raw = c.metrics[key][ci][ai];
          const factor = key === 'fwhm_hz' ? (unit === 't2' ? 1 / referenceHz : .001) : 1;
          const expected = key === 'max_final_pf' ? raw : c.metrics.accepted[ci][ai] ? raw * factor : null;
          if (expected === null) assert.equal(z[ci][ai], null);
          else near(z[ci][ai], expected);
        }
      }
    }
  }
}
// Both cutoff controls update every population map and the corresponding spectra.
for (const ci of [0, Math.floor(nc / 2), nc - 1]) {
  element('amplitude').value = String(Math.floor(na * .37));
  element('populationCutoff').value = String(ci);
  element('populationCutoff').listeners.input();
  assert.equal(+element('cutoff').value, ci);
  verifyPopulations(ci, +element('amplitude').value);
}
element('amplitude').value = '0';
element('amplitude').listeners.change();
assert.match(element('selectionFinding').textContent, /No duration has an accepted dip/);
for (const t of plots.get('widthCutoff').data) assert.ok(t.y.every(v => v === null));
verifyPopulations(nc - 1, 0);
element('amplitude').value = String(na - 1);
element('cutoff').value = '0';
element('cutoff').listeners.input();
verifyPopulations(0, na - 1);
if (process.argv[3]) fs.writeFileSync(process.argv[3], JSON.stringify(schemaChecks));
console.log(`Report checks passed: ${data.cases.length} full population maps, ${na} amplitudes, synchronized cutoff controls, T2/kHz widths, four metric maps, ${schemaChecks.length} chart specifications.`);
