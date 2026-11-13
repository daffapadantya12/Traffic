// Deck navigation: ten core slides left to right, deep-dive slides below each of them.
(() => {
  const $ = id => document.getElementById(id);
  const cols = [...document.querySelectorAll('.stage > .col')].map(c => [...c.querySelectorAll(':scope > .slide')]);
  const root = document.documentElement, stage = $('stage'), viewport = document.querySelector('.viewport');
  let ci = 0, ri = 0, notesOn = false, t0 = null;

  try { const t = localStorage.getItem('gfa-theme'); if (t) root.dataset.theme = t; } catch (e) { /* storage can be blocked */ }

  function fit() {
    const w = viewport.clientWidth, h = viewport.clientHeight;
    stage.style.setProperty('--s', Math.min(w / 1280, h / 720));
  }
  addEventListener('resize', fit);

  function label(c, r) { return r ? `${c + 1}.${r}` : `${c + 1}`; }
  function go(c, r, dir) {
    c = Math.max(0, Math.min(cols.length - 1, c)); r = Math.max(0, Math.min(cols[c].length - 1, r));
    const prev = cols[ci][ri], next = cols[c][r];
    if (!document.querySelector('.slide.active')) {              // first paint: no transition to wait for
      next.style.transition = 'none'; next.classList.add('active'); void next.offsetWidth; next.style.transition = '';
    } else if (prev !== next) {
      const d = dir || (c !== ci ? (c > ci ? 'r' : 'l') : (r > ri ? 'd' : 'u'));
      const off = { r: 'translateX(46px)', l: 'translateX(-46px)', d: 'translateY(40px)', u: 'translateY(-40px)' }[d];
      next.style.transition = 'none'; next.style.transform = off; void next.offsetWidth; next.style.transition = '';
      prev.classList.remove('active'); next.classList.add('active'); next.style.transform = '';
      if (t0 === null && (c || r)) t0 = Date.now();
    }
    ci = c; ri = r;
    $('progress').style.width = (100 * (ci + 1) / cols.length) + '%';
    const n = cols[ci].length;
    $('rail').innerHTML = n > 1 ? cols[ci].map((_, i) => `<i class="${i === ri ? 'on' : ''}"></i>`).join('') : '';
    const more = $('more'), left = n - 1 - ri;
    more.hidden = !(left > 0);
    more.textContent = ri === 0 ? `↓ ${left} deep dive${left > 1 ? 's' : ''}` : `↓ ${left} more`;
    document.querySelectorAll('.pad [data-go]').forEach(b => {
      const k = b.dataset.go;
      b.disabled = k === 'left' ? ci === 0 : k === 'right' ? ci === cols.length - 1 : k === 'up' ? ri === 0 : ri === n - 1;
    });
    const h = '#/' + (ci + 1) + (ri ? '/' + ri : '');
    if (location.hash !== h) history.replaceState(null, '', h);
    document.title = `${label(ci, ri)} · ${next.dataset.title || 'Green for the Ambulance'}`;
    const slot = next.querySelector('.twin-slot');
    if (slot) Twin.hideAll(Twin.show(slot)); else Twin.hideAll(null);
    renderNotes(); renderOverview();
  }
  const move = k => k === 'right' ? go(ci + 1, 0) : k === 'left' ? go(ci - 1, 0) : k === 'down' ? go(ci, ri + 1) : go(ci, ri - 1);

  function fromHash() {
    const m = location.hash.match(/^#\/(\d+)(?:\/(\d+))?/);
    go(m ? +m[1] - 1 : 0, m && m[2] ? +m[2] : 0);
  }
  addEventListener('hashchange', fromHash);

  // ---------------------------------------------------------------- notes, overview, theme
  function renderNotes() {
    if (!notesOn) return;
    const s = cols[ci][ri], n = s.querySelector('aside.notes');
    $('notes-where').textContent = `${label(ci, ri)} · ${s.dataset.title || ''}`;
    $('notes-body').innerHTML = n ? n.innerHTML : '<i>No notes for this slide.</i>';
    const nx = cols[ci + 1] && cols[ci + 1][0];
    $('notes-next').textContent = nx ? `Next: ${nx.dataset.title}` : 'Last slide';
  }
  function setNotes(on) {
    notesOn = on; $('notes').hidden = !on;
    root.style.setProperty('--notes-h', on ? Math.round(Math.min(260, innerHeight * 0.3)) + 'px' : '0px');
    fit(); renderNotes();
  }
  setInterval(() => {
    if (!notesOn) return;
    const s = t0 === null ? 0 : Math.floor((Date.now() - t0) / 1000);
    $('notes-timer').textContent = `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  }, 500);
  $('notes-timer').onclick = () => { t0 = Date.now(); };

  function renderOverview() {
    const o = $('overview'); if (o.hidden) return;
    o.innerHTML = cols.map((col, c) => `<div class="ocol">${col.map((s, r) =>
      `<button class="${r ? 'd' : ''} ${c === ci && r === ri ? 'on' : ''}" data-c="${c}" data-r="${r}"><b>${label(c, r)}${r ? ' · deep dive' : ''}</b>${s.dataset.title || ''}</button>`).join('')}</div>`).join('');
  }
  function setOverview(on) { $('overview').hidden = !on; renderOverview(); }
  $('overview').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { setOverview(false); go(+b.dataset.c, +b.dataset.r); } else setOverview(false); });
  function toggleTheme() {
    const dark = root.dataset.theme === 'dark'; root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('gfa-theme', root.dataset.theme); } catch (e) { /* ignore */ }
  }
  function toggleFull() { if (document.fullscreenElement) document.exitFullscreen(); else if (root.requestFullscreen) root.requestFullscreen().catch(() => {}); }
  const act = k => k === 'overview' ? setOverview($('overview').hidden) : k === 'notes' ? setNotes(!notesOn) : k === 'theme' ? toggleTheme() : toggleFull();

  // ---------------------------------------------------------------- input
  document.querySelector('.pad').addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    if (b.dataset.act === 'pad') document.querySelector('.pad').classList.toggle('open'); else if (b.dataset.go) move(b.dataset.go); else act(b.dataset.act);
    b.blur();
  });
  $('more').onclick = () => move('down');
  document.addEventListener('click', e => { const b = e.target.closest('.twin-bar button'); if (b) b.blur(); });
  addEventListener('keydown', e => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.target.matches && e.target.matches('input, select, textarea')) return;
    const k = e.key;
    if (!$('overview').hidden && k === 'Escape') return setOverview(false);
    if (k === 'ArrowRight' || k === 'PageDown' || (k === ' ' && !e.shiftKey)) { e.preventDefault(); move('right'); }
    else if (k === 'ArrowLeft' || k === 'PageUp' || (k === ' ' && e.shiftKey) || k === 'Backspace') { e.preventDefault(); move('left'); }
    else if (k === 'ArrowDown') { e.preventDefault(); move('down'); }
    else if (k === 'ArrowUp') { e.preventDefault(); move('up'); }
    else if (k === 'Home') go(0, 0); else if (k === 'End') go(cols.length - 1, 0);
    else if (k === 'o' || k === 'O' || k === 'Escape') act('overview');
    else if (k === 'n' || k === 'N') act('notes');
    else if (k === 't' || k === 'T') act('theme');
    else if (k === 'f' || k === 'F') act('full');
    else if (k === 'p' || k === 'P') { const v = Twin.current(); if (v && v.ready) v.setPlaying(!v.playing); }
    else if (/^[0-9]$/.test(k)) go(k === '0' ? 9 : +k - 1, 0);
  });
  let touch = null;
  stage.addEventListener('touchstart', e => { if (e.touches.length === 1 && !e.target.closest('.twin')) touch = [e.touches[0].clientX, e.touches[0].clientY]; else touch = null; }, { passive: true });
  stage.addEventListener('touchend', e => {
    if (!touch) return;
    const dx = e.changedTouches[0].clientX - touch[0], dy = e.changedTouches[0].clientY - touch[1]; touch = null;
    if (Math.max(Math.abs(dx), Math.abs(dy)) < 50) return;
    if (Math.abs(dx) > Math.abs(dy)) move(dx < 0 ? 'right' : 'left'); else move(dy < 0 ? 'down' : 'up');
  }, { passive: true });

  Charts.init(); fit(); fromHash();
})();
