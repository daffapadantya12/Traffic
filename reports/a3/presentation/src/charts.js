// Charts are drawn as inline SVG from the numbers build_deck.py read out of the repository's result files.
// Colours come from CSS variables, so they follow the light and dark themes.
const Charts = (() => {
  const D = JSON.parse(document.getElementById('deck-data').textContent);
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const fmt = (v, d = 0) => Number(v).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
  const low = k => D.low.find(m => m.key === k);
  const head = (t, s, legend) => `<p class="ct">${t}</p>${s ? `<p class="cs">${s}</p>` : ''}` +
    (legend ? `<div class="legend">${legend.map(([c, l, cls]) => `<span><i class="${cls || ''}" style="background:${c}"></i>${l}</span>`).join('')}</div>` : '');
  // a bar that is square at the baseline and rounded (4 px) at the data end
  const bar = (x, y, w, h, fill, extra = '') => { w = Math.max(w, 1.5); const r = Math.min(4, w, h / 2);
    return `<path class="mark" d="M${x} ${y}h${w - r}a${r} ${r} 0 0 1 ${r} ${r}v${h - 2 * r}a${r} ${r} 0 0 1 -${r} ${r}h-${w - r}z" fill="${fill}" ${extra}/>`; };

  // horizontal bars: rows = [{name, sub, bars:[{v, color, label, tip}], note}]
  function hbars({ W = 620, labelW = 170, rightW = 0, rows, max, ticks = [], tickFmt = v => fmt(v), barH = 16, rowGap = 14, refs = [], top = 6 }) {
    const x0 = labelW, x1 = W - rightW - 54, sx = v => x0 + (x1 - x0) * v / max;
    let y = top + (refs.length ? 16 : 0), out = '';
    const body = [], y0 = y;
    for (const r of rows) {
      const h = r.bars.length * barH + (r.bars.length - 1) * 2;
      body.push(`<text class="name ${r.em ? 'em' : ''}" x="${x0 - 10}" y="${y + h / 2 + 5}" text-anchor="end">${esc(r.name)}</text>`);
      r.bars.forEach((b, i) => {
        const by = y + i * (barH + 2), w = sx(Math.max(b.v, 0)) - x0;
        body.push(`<g data-tip="${esc(b.tip || `<b>${r.name}</b>${b.label}`)}">${bar(x0, by, w, barH, b.color)}
          <rect class="hit" x="${x0}" y="${by - 3}" width="${x1 - x0 + 50}" height="${barH + 6}"/>
          <text class="val" x="${x0 + Math.max(w, 1.5) + 7}" y="${by + barH / 2 + 4.5}">${esc(b.label)}</text></g>`);
      });
      if (r.note) body.push(`<text class="note" x="${W}" y="${y + h / 2 + 4}" text-anchor="end">${esc(r.note)}</text>`);
      y += h + rowGap;
    }
    const yb = y - rowGap + 6;
    for (const t of ticks) out += `<line class="grid" x1="${sx(t)}" x2="${sx(t)}" y1="${y0 - 4}" y2="${yb}"/><text class="ax" x="${sx(t)}" y="${yb + 15}" text-anchor="middle">${tickFmt(t)}</text>`;
    out += `<line class="axis" x1="${x0}" x2="${x0}" y1="${y0 - 4}" y2="${yb}"/>` + body.join('');
    for (const f of refs) out += `<line x1="${sx(f.v)}" x2="${sx(f.v)}" y1="${y0 - 6}" y2="${yb}" stroke="var(--ink)" stroke-width="1.5"/>
      <text class="note" x="${sx(f.v) + (f.left ? -6 : 6)}" y="${y0 - 9}" text-anchor="${f.left ? 'end' : 'start'}" style="fill:var(--ink)">${esc(f.label)}</text>`;
    return `<svg viewBox="0 0 ${W} ${yb + (ticks.length ? 22 : 4)}" role="img">${out}</svg>`;
  }

  const R = {
    calib(el) {
      const names = { N: 'North · RS Fatmawati', E: 'East · TB Simatupang', S: 'South · RS Fatmawati', W: 'West · RA Kartini' };
      el.innerHTML = head('Longest peak-hour queue per approach', 'Metres of standing traffic: published simulation of the junction against our twin',
        [['var(--ctx)', 'Published (target)'], ['var(--s1)', 'Our twin, mean of 4 seeds']]) +
        hbars({ W: 640, labelW: 178, rightW: 150, max: 400, ticks: [0, 100, 200, 300], tickFmt: v => v + ' m', barH: 18, rowGap: 20, rows: D.calib.map(c => ({
          name: names[c.leg], note: `${c.share}% of counted traffic`,
          bars: [{ v: c.target, color: 'var(--ctx)', label: fmt(c.target) + ' m', tip: `<b>${names[c.leg]}</b>Published longest queue: ${fmt(c.target, 1)} m` },
                 { v: c.twin, color: 'var(--s1)', label: `${fmt(c.twin)} m${c.limited ? ' · capacity-limited' : ` · ${c.err > 0 ? '+' : '−'}${Math.abs(c.err)}%`}`,
                   tip: `<b>${names[c.leg]}</b>Twin: ${fmt(c.twin, 1)} m (± ${fmt(c.sd, 1)} m over seeds)<br>${c.limited ? 'Capacity-limited upstream' : `${c.err > 0 ? '+' : '−'}${Math.abs(c.err)}% against the target`}` }] })) });
    },
    models(el) {
      const row = (m, k, pct) => ({ name: m.name, em: m.name === 'Gradient boosting', bars: [{ v: m[k], color: m.kind === 'model' ? 'var(--s1)' : 'var(--ctx)', label: pct ? Math.round(100 * m[k]) + '%' : m[k].toFixed(2),
        tip: `<b>${m.name}</b>Macro-F1 ${m.f1.toFixed(3)}<br>Accuracy with an ambulance present ${(100 * m.amb).toFixed(1)}%` }] });
      el.innerHTML = head('How well each model copies the teacher, on a seed it never saw', '', [['var(--s1)', 'Trained model'], ['var(--ctx)', 'Simple baseline']]) +
        `<div style="display:flex;gap:6px"><div style="flex:1.45"><p class="cs" style="margin:2px 0 4px">Macro-F1, all 25,920 test rows</p>${hbars({ W: 380, labelW: 150, max: 1, ticks: [0, 0.5, 1], tickFmt: v => v, rows: D.models.map(m => row(m, 'f1')), rowGap: 13 })}</div>
        <div style="flex:1"><p class="cs" style="margin:2px 0 4px">Accuracy, 1,986 rows with an ambulance</p>${hbars({ W: 262, labelW: 8, max: 1, ticks: [0, 0.5, 1], tickFmt: v => Math.round(v * 100) + '%', rows: D.models.map(m => ({ ...row(m, 'amb', true), name: '' })), rowGap: 13 })}</div></div>`;
    },
    mix(el) {
      const m = D.mix, W = 420, x = v => 4 + (W - 8) * v / 100;
      const segs = [['motorcycle', 'Motorcycles', 'var(--s1)'], ['car', 'Cars', 'var(--s2)'], ['bus', 'Buses and trucks', 'var(--s3)']];
      const val = k => k === 'bus' ? m.bus + m.truck : m[k];
      let acc = 0, bars = '';
      for (const [k, name, c] of segs) { const v = val(k), w = x(acc + v) - x(acc) - 2;
        bars += `<g data-tip="<b>${name}</b>${v.toFixed(1)}% of vehicles entering the junction"><rect class="mark" x="${x(acc)}" y="96" width="${Math.max(w, 2)}" height="40" fill="${c}" rx="3"/></g>`; acc += v; }
      el.innerHTML = head('Vehicle mix at the entry gates', `${fmt(D.mix_n)} vehicles counted by the twin's gates`, segs.map(([k, n, c]) => [c, `${n} ${val(k).toFixed(1)}%`])) +
        `<svg viewBox="0 0 ${W} 220" role="img" aria-label="Motorcycles are ${m.motorcycle}% of vehicles, inside the published range of 77 to 82 percent">
          <text x="4" y="62" style="font-size:60px;font-weight:750;letter-spacing:-.03em;fill:var(--ink)">${m.motorcycle.toFixed(1)}%</text>
          <text x="${W - 4}" y="44" text-anchor="end" class="name">of vehicles are</text><text x="${W - 4}" y="62" text-anchor="end" class="name">motorcycles</text>
          ${bars}
          <path d="M${x(77)} 146v8H${x(82)}v-8" fill="none" stroke="var(--ink)" stroke-width="1.5"/>
          <line x1="${x(79.5)}" x2="${x(79.5)}" y1="154" y2="164" stroke="var(--ink)" stroke-width="1.5"/>
          <text x="${x(79.5)}" y="180" text-anchor="middle" class="name">published for this road: 77–82%</text>
          <text x="4" y="212" class="ax">0%</text><text x="${W - 4}" y="212" text-anchor="end" class="ax">100%</text><line class="axis" x1="4" x2="${W - 4}" y1="196" y2="196"/>
        </svg>`;
    },
    queues(el) {
      const legs = ['N', 'E', 'S', 'W'], names = { N: 'North', E: 'East', S: 'South', W: 'West' }, Q = D.queues, n = Q.N.length;
      const W = 640, x0 = 62, x1 = W - 58, rowH = 66, gap = 9, top = 4, ymax = 130;
      const sx = i => x0 + (x1 - x0) * i / (n - 1), H = top + 4 * rowH + 3 * gap + 26;
      let out = '';
      for (let c = 231 / 5; c < n; c += 231 / 5) out += `<line class="grid" x1="${sx(c)}" x2="${sx(c)}" y1="${top}" y2="${H - 26}"/>`;
      legs.forEach((a, r) => {
        const y0 = top + r * (rowH + gap), sy = v => y0 + rowH - (rowH - 4) * Math.min(v, ymax) / ymax;
        const pts = Q[a].map((v, i) => `${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(' ');
        out += `<polygon points="${x0},${y0 + rowH} ${pts} ${x1},${y0 + rowH}" fill="var(--s1)" opacity=".1"/>
          <polyline points="${pts}" fill="none" stroke="var(--s1)" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>
          <line class="axis" x1="${x0}" x2="${x1}" y1="${y0 + rowH}" y2="${y0 + rowH}"/>
          <text class="name em" x="0" y="${y0 + rowH / 2 + 5}">${names[a]}</text>
          <text class="note" x="${x1 + 8}" y="${y0 + rowH / 2 + 4}">max ${Math.max(...Q[a])}</text>`;
      });
      for (const mnt of [0, 15, 30, 45, 60]) out += `<text class="ax" x="${sx(Math.min(mnt * 12, n - 1))}" y="${H - 10}" text-anchor="middle">${mnt} min</text>`;
      out += `<line id="q-cross" x1="0" x2="0" y1="${top}" y2="${H - 26}" stroke="var(--ink)" stroke-width="1" visibility="hidden"/><rect class="hit" id="q-hit" x="${x0}" y="${top}" width="${x1 - x0}" height="${H - 26 - top}"/>`;
      el.innerHTML = head('Standing vehicles on each approach through one recorded hour', 'Every 5 s; each vertical line is one 231 s signal cycle. The four panels share one scale.') +
        `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Queues on the four approaches rise and fall with the signal cycle">${out}</svg>`;
      const hit = el.querySelector('#q-hit'), cross = el.querySelector('#q-cross'), tip = document.getElementById('tip');
      hit.addEventListener('mousemove', e => {
        const b = hit.getBoundingClientRect(), i = Math.max(0, Math.min(n - 1, Math.round((e.clientX - b.left) / b.width * (n - 1))));
        cross.setAttribute('x1', sx(i)); cross.setAttribute('x2', sx(i)); cross.setAttribute('visibility', 'visible');
        const s = i * 5, t = `17:${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
        tip.innerHTML = `<b>${t}</b>` + legs.map(a => `${names[a]}: ${Q[a][i]} standing`).join('<br>'); tip.hidden = false;
        tip.style.left = Math.min(e.clientX + 14, innerWidth - 180) + 'px'; tip.style.top = (e.clientY + 14) + 'px';
      });
      hit.addEventListener('mouseleave', () => { cross.setAttribute('visibility', 'hidden'); tip.hidden = true; });
    },
    tradeoff(el) {
      const pts = D.low.filter(m => m.A < 400), W = 640, H = 400, x0 = 54, x1 = W - 18, y0 = 14, y1 = H - 44;
      const ax = [80, 190], ay = [0.5, 1.5], sx = v => x0 + (x1 - x0) * (v - ax[0]) / (ax[1] - ax[0]), sy = v => y1 - (y1 - y0) * (v - ay[0]) / (ay[1] - ay[0]);
      const cl = (v, a) => Math.max(a[0], Math.min(a[1], v)), fx = low('fixed');
      const off = { gradient_boosting: [4, 31, 'end'], longest_queue: [12, 5, 'start'], priority_mp: [-12, 5, 'end'], neural_network: [-10, -9, 'end'], max_pressure: [10, -9, 'start'],
        logistic_regression: [10, -9, 'start'], random_forest: [11, 5, 'start'], preempt_naive: [10, -10, 'start'], fixed: [10, -10, 'start'], decision_tree: [-10, -10, 'end'] };
      const short = { gradient_boosting: 'Gradient boosting', longest_queue: 'Longest queue', priority_mp: 'Priority teacher', neural_network: 'Neural network', max_pressure: 'Max-pressure',
        logistic_regression: 'Logistic regression', random_forest: 'Random forest', preempt_naive: 'Naive pre-emption', fixed: 'Fixed plan (today)', decision_tree: 'Decision tree' };
      let out = '';
      for (const t of [80, 100, 120, 140, 160, 180]) out += `<line class="grid" x1="${sx(t)}" x2="${sx(t)}" y1="${y0}" y2="${y1}"/><text class="ax" x="${sx(t)}" y="${y1 + 16}" text-anchor="middle">${t} s</text>`;
      for (const t of [0.5, 0.75, 1, 1.25, 1.5]) out += `<line class="grid" x1="${x0}" x2="${x1}" y1="${sy(t)}" y2="${sy(t)}"/><text class="ax" x="${x0 - 8}" y="${sy(t) + 4}" text-anchor="end">${t.toFixed(2)}</text>`;
      out += `<rect x="${x0}" y="${sy(D.cost_gate)}" width="${sx(fx.A) - x0}" height="${y1 - sy(D.cost_gate)}" fill="var(--accent)" opacity=".1"/>
        <line x1="${sx(fx.A)}" x2="${sx(fx.A)}" y1="${y0}" y2="${y1}" stroke="var(--axis)" stroke-width="1.5"/><line x1="${x0}" x2="${x1}" y1="${sy(1)}" y2="${sy(1)}" stroke="var(--axis)" stroke-width="1.5"/>
        <text class="note" x="${x0 + 8}" y="${y1 - 8}" style="fill:var(--accent-ink)">better than today for ambulances, within the cost gate</text>
        <text class="ax" x="${(x0 + x1) / 2}" y="${H - 6}" text-anchor="middle">Ambulance delay (s), averaged over directions</text>
        <text class="ax" transform="translate(13 ${(y0 + y1) / 2}) rotate(-90)" text-anchor="middle">Everyone else's delay (today = 1)</text>`;
      for (const m of pts) {
        const X = sx(m.A), Y = sy(m.R), win = m.key === 'gradient_boosting', fix = m.key === 'fixed', o = off[m.key] || [10, 4, 'start'];
        const tip = `<b>${m.name}</b>Ambulance delay ${m.A} s (95%: ${m.A_lo} to ${m.A_hi})<br>Others' delay ${m.R.toFixed(2)} × today (95%: ${m.R_lo.toFixed(2)} to ${m.R_hi.toFixed(2)})<br>Worst approach ${m.worst.toFixed(2)} × today (${m.worst_leg})<br>${m.passes ? 'Passes every gate on point estimates' : fix ? 'Reference' : 'Fails a gate'}`;
        out += `<g data-tip="${esc(tip)}">
          <line x1="${sx(cl(m.A_lo, ax))}" x2="${sx(cl(m.A_hi, ax))}" y1="${Y}" y2="${Y}" stroke="var(--ctx)" stroke-width="1" opacity=".55"/><line x1="${X}" x2="${X}" y1="${sy(cl(m.R_lo, ay))}" y2="${sy(cl(m.R_hi, ay))}" stroke="var(--ctx)" stroke-width="1" opacity=".55"/>
          ${win ? `<circle cx="${X}" cy="${Y}" r="12" fill="none" stroke="var(--s1)" stroke-width="2"/>` : ''}
          ${fix ? `<rect class="mark" x="${X - 6}" y="${Y - 6}" width="12" height="12" fill="var(--ink)" stroke="var(--card)" stroke-width="2"/>`
            : `<circle class="mark" cx="${X}" cy="${Y}" r="${m.passes ? 6.5 : 5.5}" fill="${m.passes ? 'var(--s1)' : 'var(--card)'}" stroke="${m.passes ? 'var(--card)' : 'var(--ctx)'}" stroke-width="${m.passes ? 2 : 2.5}"/>`}
          <text class="halo ${win ? 'em' : ''}" x="${X + o[0]}" y="${Y + o[1]}" text-anchor="${o[2]}" style="font-size:12.5px">${short[m.key]}</text>
          <circle class="hit" cx="${X}" cy="${Y}" r="16"/></g>`;
      }
      el.innerHTML = head('Ambulance delay against everyone else\'s delay, low demand', '',
        [['var(--s1)', 'Passes every gate (point estimates)'], ['var(--ctx)', 'Fails a gate', 'ring'], ['var(--ink)', 'Today\'s fixed plan']]) + `<svg viewBox="0 0 ${W} ${H}" role="img">${out}</svg>`;
    },
    win(el) {
      const rows = Object.entries(D.win).sort((a, b) => b[1] - a[1]).map(([k, v]) => ({ name: k.replace('Learned: g', 'G').replace('Learned: r', 'R'), em: k.includes('gradient'),
        bars: [{ v, color: k === 'No method' ? 'var(--ctx)' : 'var(--s1)', label: v.toFixed(1) + '%', tip: `<b>${k}</b>Wins ${v.toFixed(1)}% of 1,000 resamples of the four seeds` }] }));
      el.innerHTML = head('Who wins when the four seeds are resampled', 'Share of 1,000 bootstrap resamples in which each method is the Nash choice') +
        hbars({ W: 600, labelW: 170, max: 60, ticks: [0, 20, 40, 60], tickFmt: v => v + '%', rows, barH: 20, rowGap: 16 });
    },
    stress(el) {
      const order = ['fixed', 'priority_mp', 'max_pressure', 'longest_queue', 'preempt_naive'];
      const panel = (lvl, title) => { const fx = D.stress[lvl].find(m => m.key === 'fixed');
        return `<p class="cs" style="margin:6px 0 4px"><b style="color:var(--ink)">${title}</b></p>` + hbars({ W: 640, labelW: 168, rightW: 150, max: 2600, ticks: [0, 1000, 2000], tickFmt: v => fmt(v) + ' s', barH: 15, rowGap: 9,
          rows: order.map(k => { const m = D.stress[lvl].find(x => x.key === k);
            return { name: m.name, em: k === 'priority_mp', note: k === 'fixed' ? `${Math.round(100 * m.unf)}% stranded` : `${(100 * m.unf).toFixed(0)}% stranded · worst ${m.worst.toFixed(1)}×`,
              bars: [{ v: m.A, color: k === 'fixed' ? 'var(--ctx)' : 'var(--s1)', label: fmt(m.A) + ' s', tip: `<b>${m.name}</b>Ambulance delay ${fmt(m.A)} s (${k === 'fixed' ? 'reference' : Math.round(100 * (m.A / fx.A - 1)) + '% against today'})<br>${(100 * m.unf).toFixed(1)}% of ambulances still stuck at the end<br>Worst approach ${m.worst.toFixed(2)} × today` }] }; }) }); };
      el.innerHTML = head('Ambulance delay when the peak is held for 4.4 hours', 'Seconds, averaged over directions; right: ambulances stranded, and the worst approach against today',
        [['var(--ctx)', 'Today\'s fixed plan'], ['var(--s1)', 'Adaptive and priority methods']]) + panel('design', 'Design demand (the calibrated peak)') + panel('high', 'High demand (1.15 × the peak)');
    },
    hind(el) {
      const keys = ['max_pressure', 'longest_queue', 'gradient_boosting', 'priority_mp', 'fixed', 'neural_network', 'preempt_naive'];
      const rows = keys.map(k => { const m = low(k); return { name: m.name.replace('Learned: ', '').replace(/^./, c => c.toUpperCase()), bars: [{ v: m.rep, color: 'var(--s1)', label: fmt(m.rep) + ' s',
        tip: `<b>${m.name}</b>${m.rep} s on the replayed passages<br>captures ${Math.round(100 * m.gain)}% of the gain perfect timing offers over max-pressure` }] }; });
      el.innerHTML = head('Hindsight: how much better could the timing be?', 'Ambulance delay on the 24 replayed passages (s)') +
        hbars({ W: 520, labelW: 150, max: 190, ticks: [0, 50, 100, 150], tickFmt: v => v + ' s', rows, barH: 15, rowGap: 10, refs: [{ v: D.hind.ceiling, label: `perfect timing: ${fmt(D.hind.ceiling)} s`, left: true }] });
    },
    moto(el) {
      const rows = D.moto70.slice().sort((a, b) => a.A - b.A).map(m => { const b = low(m.key);
        return { name: m.name.replace('Learned: g', 'G'), bars: [{ v: b.A, color: 'var(--ctx)', label: fmt(b.A) + ' s', tip: `<b>${m.name}</b>80% motorcycles: ${b.A} s` },
          { v: m.A, color: 'var(--s2)', label: `${fmt(m.A)} s`, tip: `<b>${m.name}</b>70% motorcycles: ${m.A} s (× ${(m.A / b.A).toFixed(1)})<br>worst approach ${m.worst.toFixed(2)} × today` }] }; });
      el.innerHTML = head('What if only 70% were motorcycles?', 'Ambulance delay at low demand (s)', [['var(--ctx)', '80% motorcycles (the dataset)'], ['var(--s2)', '70% motorcycles']]) +
        hbars({ W: 520, labelW: 150, max: 400, ticks: [0, 100, 200, 300], tickFmt: v => v + ' s', rows, barH: 12, rowGap: 9 });
    },
    red(el) {
      const keys = ['fixed', 'max_pressure', 'gradient_boosting', 'longest_queue', 'priority_mp', 'neural_network', 'preempt_naive'];
      const rows = keys.map(k => { const m = low(k), over = m.red > 252; return { name: m.name.replace('Learned: ', '').replace(/^./, c => c.toUpperCase()), em: over,
        bars: [{ v: m.red, color: over ? 'var(--red)' : 'var(--s1)', label: `${fmt(m.red)} s${over ? '  ▲ above the cap' : ''}`, tip: `<b>${m.name}</b>Longest red on any approach: ${fmt(m.red)} s` }] }; });
      el.innerHTML = head('Longest red any approach had to wait', 'Low demand, seconds; the plant caps request-only controllers at 252 s') +
        hbars({ W: 620, labelW: 160, max: 500, ticks: [0, 100, 200, 300, 400], tickFmt: v => v + ' s', rows, barH: 18, rowGap: 13, refs: [{ v: 252, label: 'cap: 252 s', left: true }] });
    },
  };

  function fill() {
    document.querySelectorAll('[data-v]').forEach(td => {
      const [f, k] = td.dataset.v.split(':');
      if (f === 'lambda') { td.textContent = `${Math.round(D.lambda[0])} to ${Math.round(D.lambda[1])}`; return; }
      const m = low(k); if (!m) return;
      td.textContent = f === 'A' ? m.A.toFixed(1) + ' s' : f === 'R' ? m.R.toFixed(2) + ' ×' : m.nash == null ? '–' : m.nash.toFixed(4);
    });
  }
  function init() {
    document.querySelectorAll('[data-chart]').forEach(el => { try { R[el.dataset.chart](el); } catch (e) { el.textContent = 'Chart could not be drawn: ' + e.message; console.error(e); } });
    fill();
    const tip = document.getElementById('tip');
    document.addEventListener('mousemove', e => {
      const g = e.target.closest && e.target.closest('[data-tip]');
      if (!g) { if (!e.target.closest || !e.target.closest('#q-hit')) tip.hidden = true; return; }
      tip.innerHTML = g.dataset.tip; tip.hidden = false;
      tip.style.left = Math.min(e.clientX + 14, innerWidth - 310) + 'px'; tip.style.top = Math.min(e.clientY + 14, innerHeight - 120) + 'px';
    });
  }
  return { init, D };
})();
