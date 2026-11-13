// 3D replay of recorded twin frames.  The scene code is adapted from the live viewers
// (traffic_twin/fatmawati/static/index.html and scripts/bundaran_hi_3d.html); here the frames come from
// recordings baked into this file instead of a /stream, so it runs with no server.
const Twin = (() => {
  let T = null, loading = null;

  async function b64bytes(b64) {
    try { return new Uint8Array(await (await fetch('data:application/octet-stream;base64,' + b64)).arrayBuffer()); }
    catch (e) { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u; }
  }
  async function gunzip(b64) {
    const bytes = await b64bytes(b64);
    return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text();
  }
  const store = id => JSON.parse(document.getElementById(id).textContent);

  // Three.js is inlined (gzip + base64).  Its modules import each other by name, so each one is turned into a
  // blob URL and the import specifiers are pointed at those URLs.
  function loadThree() {
    if (loading) return loading;
    loading = (async () => {
      const V = store('vendor-data'), src = {};
      await Promise.all(Object.keys(V).map(async k => { src[k] = await gunzip(V[k]); }));
      const url = s => URL.createObjectURL(new Blob([s], { type: 'text/javascript' }));
      const uCore = url(src.core);
      const uThree = url(src.module.replaceAll('./three.core.js', uCore));
      const fix = s => s.replace(/from\s*['"]three['"]/g, `from '${uThree}'`);
      const uOrbit = url(fix(src.orbit));
      const uMap = url(fix(src.map).replace(/from\s*['"]\.\/OrbitControls\.js['"]/, `from '${uOrbit}'`));
      const uBgu = url(fix(src.bgu));
      const [THREE, m, b] = await Promise.all([import(uThree), import(uMap), import(uBgu)]);
      T = { THREE, MapControls: m.MapControls, mergeGeometries: b.mergeGeometries };
      return T;
    })();
    return loading;
  }

  const SITES = {
    fat: {
      title: 'Perempatan Fatmawati', legs: ['N', 'E', 'S', 'W'], live: 'http://127.0.0.1:8770/',
      liveCmd: 'python3 -m traffic_twin.fatmawati.webapp3d --port 8770 --level low --speed 5',
      liveNote: 'Run it from the repository root, wait for the 10-minute warm-up, then press Live again.',
      cams: [['junction', 'Junction'], ['north', 'North approach'], ['closeup', 'Motorcycle box'], ['flyover', 'Toll flyover'], ['hospital', 'Hospital'],
             ['overview', 'Overview'], ['follow', 'Follow ambulance']],
      fog: [420, 980], maxDist: 900,
    },
    bhi: {
      title: 'Bundaran HI', legs: ['N', 'E', 'S'], live: 'http://127.0.0.1:8792/',
      liveCmd: 'python3 -I scripts/bundaran_hi_3d.py --port 8792',
      liveNote: 'Needs the branch claude/bundaran-hi-demo checked out. Wait for the warm-up, then press Live again.',
      cams: [['junction', 'Ring'], ['monument', 'Monument'], ['thamrin', 'Thamrin entry'], ['overview', 'Overview']],
      fog: [420, 980], maxDist: 900,
    },
  };

  class Viewer {
    constructor(site) {
      this.site = site; this.cfg = SITES[site];
      this.root = document.createElement('div'); this.root.className = 'twin';
      this.root.innerHTML = `<div class="twin-canvas"></div><div class="twin-msg">Loading the 3D replay…</div>
        <div class="twin-huds"></div><div class="twin-divider"></div><iframe class="twin-live" title="Live twin" hidden></iframe>
        <div class="twin-pop" hidden></div><div class="twin-attr">Roads &amp; buildings © OpenStreetMap contributors (ODbL) · Eclipse SUMO · recorded replay</div>
        <div class="twin-bar"></div>`;
      this.q = s => this.root.querySelector(s);
      this.mode = 'a'; this.playing = true; this.speed = 4; this.tau = 0; this.cam = 'junction';
      this.active = false; this.ready = false; this.xray = false; this.live = false; this.sets = [];
    }

    mount(slot) {
      if (this.root.parentNode !== slot) slot.appendChild(this.root);
      const want = slot.dataset.mode || 'a', cam = slot.dataset.cam || 'junction';
      this.active = true;
      if (!this.ready) { this.pending = { want, cam, xray: slot.dataset.xray === '1' }; if (!this.booting) this.boot(); return; }
      this.setMode(want); this.setCam(cam); if (this.G.flyover) this.setXray(slot.dataset.xray === '1'); this.resize(); this.start();
    }
    unmount() { this.active = false; if (this.renderer) this.renderer.setAnimationLoop(null); if (this.live) this.setLive(false); }

    async boot() {
      this.booting = true;
      try {
        if (!('DecompressionStream' in window)) throw new Error('this browser is too old for the embedded replay (no DecompressionStream)');
        const S = store('twin-data')[this.site];
        const [three, geo, ...clips] = await Promise.all([loadThree(), gunzip(S.geo).then(JSON.parse), ...S.clips.map(c => gunzip(c).then(JSON.parse))]);
        this.G = geo; this.clips = clips;
        for (const c of clips) { c.n = c.V.length; c.idx = new Array(c.n); c.tlsCache = new Map(); }
        this.build();
        this.ready = true; this.q('.twin-msg').remove();
        const p = this.pending || { want: 'a', cam: 'junction' };
        this.setMode(p.want); this.setCam(p.cam); if (this.G.flyover) this.setXray(!!p.xray); this.resize();
        if (this.active) this.start();
      } catch (e) {
        this.q('.twin-msg').textContent = 'The 3D replay could not start: ' + (e.message || e);
        console.error(e);
      }
    }

    // ------------------------------------------------------------ scene
    build() {
      const { THREE, MapControls, mergeGeometries } = T, G = this.G, cfg = this.cfg;
      const V = (x, y, z = 0) => new THREE.Vector3(x, z, -y);       // simulation (x east, y north, z up) -> three (y up)
      this.V = V;
      const renderer = this.renderer = new THREE.WebGLRenderer({ antialias: true });
      this.q('.twin-canvas').appendChild(renderer.domElement);
      const scene = this.scene = new THREE.Scene();
      const sky = 0xbfd6ec;
      scene.background = new THREE.Color(sky);
      scene.fog = new THREE.Fog(sky, cfg.fog[0], cfg.fog[1]);
      const camera = this.camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.5, 6000);
      const controls = this.controls = new MapControls(camera, renderer.domElement);
      controls.enableDamping = true; controls.dampingFactor = 0.08;
      controls.maxPolarAngle = Math.PI * 0.47; controls.minDistance = 6; controls.maxDistance = cfg.maxDist;
      controls.addEventListener('start', () => { if (this.cam === 'follow') this.setCam(null); });
      scene.add(new THREE.HemisphereLight(0xeef4ff, 0x4a5340, 0.95));
      const sun = new THREE.DirectionalLight(0xffffff, 1.25); sun.position.set(400, 700, 250); scene.add(sun);
      const ground = new THREE.Mesh(new THREE.PlaneGeometry(9000, 9000), new THREE.MeshLambertMaterial({ color: 0x8d9a78 }));
      ground.rotation.x = -Math.PI / 2; ground.position.y = -0.25; scene.add(ground);

      const flyMats = this.flyMats = [], heads = this.heads = [];
      const laneStrips = (lanes, lift) => {
        const pos = [], col = [], idx = [];
        const cA = new THREE.Color(0x41454d), cT = new THREE.Color(0x50545d);
        for (const ln of lanes) {
          const p = ln.p; if (p.length < 2) continue;
          const hw = ln.w / 2, base = pos.length / 3, c = ln.k.startsWith('motorway') ? cT : cA;
          for (let i = 0; i < p.length; i++) {
            const a = p[Math.max(i - 1, 0)], b = p[Math.min(i + 1, p.length - 1)];
            let tx = b[0] - a[0], ty = b[1] - a[1]; const L = Math.hypot(tx, ty) || 1; tx /= L; ty /= L;
            const nx = -ty, ny = tx, z = (p[i][2] || 0) + lift;
            pos.push(p[i][0] + nx * hw, z, -(p[i][1] + ny * hw), p[i][0] - nx * hw, z, -(p[i][1] - ny * hw));
            col.push(c.r, c.g, c.b, c.r, c.g, c.b);
          }
          for (let i = 0; i < p.length - 1; i++) { const k = base + 2 * i; idx.push(k, k + 1, k + 2, k + 1, k + 3, k + 2); }
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
        g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
        g.setIndex(idx); g.computeVertexNormals();
        return new THREE.Mesh(g, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }));
      };
      const laneLines = lanes => {
        const byEdge = new Map();
        for (const ln of lanes) { const e = ln.id.slice(0, ln.id.lastIndexOf('_')) + '|' + (ln.run || 0); if (!byEdge.has(e)) byEdge.set(e, []); byEdge.get(e).push(ln); }
        const pos = [];
        for (const group of byEdge.values()) {
          group.sort((a, b) => a.idx - b.idx);
          for (let k = 1; k < group.length; k++) {
            const A = group[k - 1].p, B = group[k].p;
            if (A.length !== B.length) continue;
            for (let i = 0; i < A.length - 1; i++) {
              const z0 = ((A[i][2] || 0) + (B[i][2] || 0)) / 2 + 0.06, z1 = ((A[i + 1][2] || 0) + (B[i + 1][2] || 0)) / 2 + 0.06;
              pos.push((A[i][0] + B[i][0]) / 2, z0, -(A[i][1] + B[i][1]) / 2, (A[i + 1][0] + B[i + 1][0]) / 2, z1, -(A[i + 1][1] + B[i + 1][1]) / 2);
            }
          }
        }
        const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
        const lines = new THREE.LineSegments(g, new THREE.LineDashedMaterial({ color: 0xf2f2f2, dashSize: 3, gapSize: 5, transparent: true, opacity: 0.8 }));
        lines.computeLineDistances();
        return lines;
      };
      const junctionMeshes = js => {
        const geos = [];
        for (const j of js) {
          if (j.p.length < 3) continue;
          const g = new THREE.ShapeGeometry(new THREE.Shape(j.p.map(p => new THREE.Vector2(p[0], p[1]))));
          g.rotateX(-Math.PI / 2); g.translate(0, (j.z || 0) + 0.015, 0); geos.push(g);
        }
        return new THREE.Mesh(mergeGeometries(geos), new THREE.MeshLambertMaterial({ color: 0x41454d, side: THREE.DoubleSide }));
      };
      const pierMeshes = (piers, color) => {                  // [x, y, underside, ux, uy, cap half-length]; 0: a plain column
        const m = new THREE.Matrix4(), q = new THREE.Quaternion(), Y = new THREE.Vector3(0, 1, 0);
        const colGeo = new THREE.CylinderGeometry(1, 1, 1, 16).translate(0, 0.5, 0);
        const capGeo = new THREE.BoxGeometry(1, 1, 1).translate(0, -0.5, 0);
        const mat = new THREE.MeshLambertMaterial({ color });
        const cols = new THREE.InstancedMesh(colGeo, mat, piers.length), caps = new THREE.InstancedMesh(capGeo, mat, piers.length);
        piers.forEach(([x, y, under, ux, uy, half], i) => {
          q.setFromAxisAngle(Y, Math.atan2(ux, -uy));
          const capH = half > 0 ? Math.min(1.2, Math.max(0.01, under - 0.6)) : 0.01;
          const [rx, rz] = half > 0 ? [1.2, 0.8] : [0.5, 0.5];
          m.compose(new THREE.Vector3(x, under, -y), q, new THREE.Vector3(Math.max(2 * half, 0.01), capH, 1.8)); caps.setMatrixAt(i, m);
          m.compose(new THREE.Vector3(x, -0.25, -y), q, new THREE.Vector3(rx, under - capH + 0.25, rz)); cols.setMatrixAt(i, m);
        });
        return { mat, meshes: [cols, caps] };
      };
      const isRaised = ln => ln.k.startsWith('motorway') && ln.p.some(q => (q[2] || 0) > 0.3);
      const flyover = F => {                                  // the JORR toll flyover: deck, parapets, piers, lamp posts
        const grp = new THREE.Group();
        if (!F || !F.decks.length) return grp;
        const pos = [], col = [];
        const cTop = new THREE.Color(0x5a5e66), cWall = new THREE.Color(0xc9ccd0), cPar = new THREE.Color(0xdfe1e4), cUnder = new THREE.Color(0x8f9399);
        const P = (x, y, z) => [x, z, -y];
        const tri = (a, b, c, k) => { pos.push(...a, ...b, ...c); for (let j = 0; j < 3; j++) col.push(k.r, k.g, k.b); };
        const quad = (a, b, c, d, k) => { tri(a, b, c, k); tri(a, c, d, k); };
        for (const d of F.decks) {
          const c = d.p, n = c.length; if (n < 2) continue;
          const nrm = c.map((p, i) => { const a = c[Math.max(i - 1, 0)], b = c[Math.min(i + 1, n - 1)];
            const tx = b[0] - a[0], ty = b[1] - a[1], L = Math.hypot(tx, ty) || 1; return [-ty / L, tx / L]; });
          const at = (i, o, z) => P(c[i][0] + nrm[i][0] * o, c[i][1] + nrm[i][1] * o, z);
          const hw = d.hw, t = F.parapet_w, H = F.parapet, bot = z => Math.max(-0.25, z - F.depth);
          for (let i = 0; i < n - 1; i++) {
            const z0 = c[i][2], z1 = c[i + 1][2], j = i + 1;
            quad(at(i, -hw, z0 - 0.02), at(j, -hw, z1 - 0.02), at(j, hw, z1 - 0.02), at(i, hw, z0 - 0.02), cTop);
            quad(at(i, -hw, bot(z0)), at(i, hw, bot(z0)), at(j, hw, bot(z1)), at(j, -hw, bot(z1)), cUnder);
            for (const s of [-1, 1]) {
              quad(at(i, s * hw, bot(z0)), at(j, s * hw, bot(z1)), at(j, s * hw, z1 + H), at(i, s * hw, z0 + H), cWall);
              const o = s * (hw - t);
              quad(at(i, o, z0), at(j, o, z1), at(j, o, z1 + H), at(i, o, z0 + H), cPar);
              quad(at(i, o, z0 + H), at(j, o, z1 + H), at(j, s * hw, z1 + H), at(i, s * hw, z0 + H), cPar);
            }
          }
          for (const i of [0, n - 1]) { const z = c[i][2]; quad(at(i, -hw, bot(z)), at(i, hw, bot(z)), at(i, hw, z + H), at(i, -hw, z + H), cWall); }
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
        g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
        g.computeVertexNormals();
        const deckMat = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
        grp.add(new THREE.Mesh(g, deckMat));
        const piers = pierMeshes(F.piers, 0xc3c6ca); grp.add(...piers.meshes);
        const m = new THREE.Matrix4(), q = new THREE.Quaternion(), Y = new THREE.Vector3(0, 1, 0);
        const lampGeo = mergeGeometries([new THREE.CylinderGeometry(0.09, 0.14, 9, 6).translate(0, 4.5, 0),
          new THREE.BoxGeometry(2.4, 0.12, 0.12).translate(1.2, 9, 0), new THREE.BoxGeometry(0.8, 0.2, 0.35).translate(2.3, 8.9, 0)]);
        const lampMat = new THREE.MeshLambertMaterial({ color: 0x9a9fa6 });
        const lamps = new THREE.InstancedMesh(lampGeo, lampMat, F.lamps.length);
        F.lamps.forEach(([x, y, z, dx, dy], i) => { q.setFromAxisAngle(Y, Math.atan2(dy, dx)); m.compose(new THREE.Vector3(x, z, -y), q, new THREE.Vector3(1, 1, 1)); lamps.setMatrixAt(i, m); });
        grp.add(lamps);
        flyMats.push(deckMat, piers.mat, lampMat);
        return grp;
      };
      const metro = M => {                                    // the MRT viaduct: decks, piers, masts and wire, rails, stations
        const grp = new THREE.Group();
        if (!M) return grp;
        const pos = [], col = [];
        const cTop = new THREE.Color(0x9a9ea5), cWall = new THREE.Color(0xe3e5e8), cUnder = new THREE.Color(0xa3a7ad), cBed = new THREE.Color(0x3a3c41);
        const P = (x, y, z) => [x, z, -y];
        const tri = (a, b, c, k) => { pos.push(...a, ...b, ...c); for (let j = 0; j < 3; j++) col.push(k.r, k.g, k.b); };
        const quad = (a, b, c, d, k) => { tri(a, b, c, k); tri(a, c, d, k); };
        const rails = [], wire = [];
        for (const d of M.decks) {
          const c = d.p, n = c.length; if (n < 2) continue;
          const nrm = c.map((p, i) => { const a = c[Math.max(i - 1, 0)], b = c[Math.min(i + 1, n - 1)];
            const tx = b[0] - a[0], ty = b[1] - a[1], L = Math.hypot(tx, ty) || 1; return [-ty / L, tx / L]; });
          const at = (i, o, z) => P(c[i][0] + nrm[i][0] * o, c[i][1] + nrm[i][1] * o, z);
          const top = M.top, und = M.under, H = 1.1, t = 0.3;
          for (let i = 0; i < n - 1; i++) {
            const j = i + 1;
            const inA = d.side[i] * d.inner[i], inB = d.side[j] * d.inner[j], outA = -d.side[i] * M.half, outB = -d.side[j] * M.half;
            const pA = outA + d.side[i] * t, pB = outB + d.side[j] * t;
            quad(at(i, inA, top), at(j, inB, top), at(j, outB, top), at(i, outA, top), cTop);
            quad(at(i, inA, und), at(i, outA, und), at(j, outB, und), at(j, inB, und), cUnder);
            quad(at(i, outA, und), at(j, outB, und), at(j, outB, top + H), at(i, outA, top + H), cWall);
            quad(at(i, pA, top), at(j, pB, top), at(j, pB, top + H), at(i, pA, top + H), cWall);
            quad(at(i, pA, top + H), at(j, pB, top + H), at(j, outB, top + H), at(i, outA, top + H), cWall);
            if (d.inner[i] >= M.half - 0.01) quad(at(i, inA, und), at(j, inB, und), at(j, inB, top), at(i, inA, top), cWall);
            quad(at(i, -1.3, top + 0.12), at(j, -1.3, top + 0.12), at(j, 1.3, top + 0.12), at(i, 1.3, top + 0.12), cBed);
            for (const o of [-0.53, 0.53]) rails.push(...at(i, o, M.rail - 0.04), ...at(j, o, M.rail - 0.04));
            wire.push(...at(i, 0, M.rail + 5.1), ...at(j, 0, M.rail + 5.1));
          }
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
        g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
        g.computeVertexNormals();
        const deckMat = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
        grp.add(new THREE.Mesh(g, deckMat));
        const lines = (arr, color) => { const lg = new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.Float32BufferAttribute(arr, 3));
          return new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color })); };
        grp.add(lines(rails, 0xa4aab2), lines(wire, 0x2b2d31));
        const piers = pierMeshes(M.piers, 0xd0d3d7); grp.add(...piers.meshes);
        const m = new THREE.Matrix4(), q = new THREE.Quaternion(), Y = new THREE.Vector3(0, 1, 0);
        const mastGeo = mergeGeometries([new THREE.CylinderGeometry(0.12, 0.16, 6.6, 6).translate(0, 3.3, 0),
          new THREE.BoxGeometry(2.7, 0.12, 0.12).translate(1.35, 6.2, 0), new THREE.BoxGeometry(0.06, 0.9, 0.06).translate(2.6, 5.75, 0)]);
        const mastMat = new THREE.MeshLambertMaterial({ color: 0x7d838b });
        const masts = new THREE.InstancedMesh(mastGeo, mastMat, M.masts.length);
        M.masts.forEach(([x, y, z, dx, dy], i) => { q.setFromAxisAngle(Y, Math.atan2(dy, dx)); m.compose(new THREE.Vector3(x, z, -y), q, new THREE.Vector3(1, 1, 1)); masts.setMatrixAt(i, m); });
        grp.add(masts);
        const hallMat = new THREE.MeshLambertMaterial({ color: 0xeceef0, side: THREE.DoubleSide });
        const windowMat = new THREE.MeshLambertMaterial({ color: 0x5f7686, side: THREE.DoubleSide });
        const glassMat = new THREE.MeshLambertMaterial({ color: 0x9cc6dc, transparent: true, opacity: 0.3, depthWrite: false, side: THREE.DoubleSide });
        const roofMat = new THREE.MeshLambertMaterial({ color: 0xf3f4f6, side: THREE.DoubleSide });
        for (const st of M.stations) {
          const slab = (poly, z0, z1, mat) => {
            const sg = new THREE.ExtrudeGeometry(new THREE.Shape(poly.map(p => new THREE.Vector2(p[0], p[1]))), { depth: z1 - z0, bevelEnabled: false });
            sg.rotateX(-Math.PI / 2); sg.translate(0, z0, 0); grp.add(new THREE.Mesh(sg, mat));
          };
          slab(st.p, st.floor, st.floor + 1.2, hallMat); slab(st.p, st.floor + 1.2, st.platform - 1.4, windowMat);
          slab(st.p, st.platform - 1.4, st.platform, hallMat); slab(st.p, st.platform + 0.02, st.roof, glassMat);
          slab(st.roof_p || st.p, st.roof, st.roof + 0.7, roofMat);
        }
        flyMats.push(deckMat, piers.mat, mastMat, hallMat, windowMat, glassMat, roofMat);
        return grp;
      };
      const buildingMesh = blds => {
        const pos = [], col = [], tone = new THREE.Color();
        for (const b of blds) {
          const pts = b.p, n = pts.length; if (n < 3) continue;
          const h = b.h;
          const shade = b.hosp ? [0.96, 0.95, 0.95] : (() => { const s = 0.72 + ((Math.abs(pts[0][0] * 7.1 + pts[0][1] * 3.3) % 1) * 0.16); tone.setHSL(0.08, 0.10, s); return [tone.r, tone.g, tone.b]; })();
          const push = (x, y, z) => { pos.push(x, z, -y); col.push(shade[0], shade[1], shade[2]); };
          for (let i = 0; i < n; i++) {
            const a = pts[i], d = pts[(i + 1) % n];
            push(a[0], a[1], -0.25); push(d[0], d[1], -0.25); push(d[0], d[1], h);
            push(a[0], a[1], -0.25); push(d[0], d[1], h); push(a[0], a[1], h);
          }
          const tris = THREE.ShapeUtils.triangulateShape(pts.map(p => new THREE.Vector2(p[0], p[1])), []);
          for (const t of tris) for (const k of t) push(pts[k][0], pts[k][1], h);
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
        g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
        g.computeVertexNormals();
        return new THREE.Mesh(g, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }));
      };
      const furniture = (stoplines, hs) => {
        const grp = new THREE.Group();
        const white = new THREE.MeshBasicMaterial({ color: 0xffffff });
        const box = new THREE.MeshBasicMaterial({ color: 0xd93c3c, transparent: true, opacity: 0.45, side: THREE.DoubleSide, depthWrite: false });
        for (const s of stoplines) {
          const ax = s.a[0], ay = s.a[1], bx = s.b[0], by = s.b[1], len = Math.hypot(bx - ax, by - ay);
          const stop = new THREE.Mesh(new THREE.BoxGeometry(len, 0.05, 0.45), white);
          stop.position.copy(V((ax + bx) / 2, (ay + by) / 2, 0.08)); stop.rotation.y = Math.atan2(by - ay, bx - ax); grp.add(stop);
          if (s.rhk && s.depth > 0) {                     // cars stop s.depth metres back; the box in front is for motorcycles
            const ux = s.ux, uy = s.uy, d = s.depth;
            const quad = new THREE.Shape([new THREE.Vector2(ax, ay), new THREE.Vector2(bx, by), new THREE.Vector2(bx - ux * d, by - uy * d), new THREE.Vector2(ax - ux * d, ay - uy * d)]);
            const g = new THREE.ShapeGeometry(quad); g.rotateX(-Math.PI / 2); g.translate(0, 0.07, 0); grp.add(new THREE.Mesh(g, box));
            const carStop = new THREE.Mesh(new THREE.BoxGeometry(len, 0.05, 0.3), white);
            carStop.position.copy(V((ax + bx) / 2 - ux * d, (ay + by) / 2 - uy * d, 0.08)); carStop.rotation.y = Math.atan2(by - ay, bx - ax); grp.add(carStop);
          }
        }
        const poleMat = new THREE.MeshLambertMaterial({ color: 0x2d3036 });
        for (const h of hs) {
          const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 5.2, 8), poleMat); pole.position.copy(V(h.x, h.y, 2.6)); grp.add(pole);
          const housing = new THREE.Mesh(new THREE.BoxGeometry(0.5, 1.5, 0.5), poleMat); housing.position.copy(V(h.x, h.y, 5.6)); grp.add(housing);
          const lamps = {};
          [['r', 0xff3b30, 6.05], ['y', 0xffb300, 5.6], ['g', 0x34c759, 5.15]].forEach(([k, c, z]) => {
            const m = new THREE.MeshBasicMaterial({ color: 0x222222 });
            const s = new THREE.Mesh(new THREE.SphereGeometry(0.18, 12, 8), m); s.position.copy(V(h.x, h.y, z)); grp.add(s); lamps[k] = { m, c };
          });
          heads.push({ t: h.t || 0, links: h.links, lamps });
        }
        return grp;
      };
      const landmark = (pool, mon) => {                       // Bundaran HI: the pool as OSM draws it, a schematic monument
        const grp = new THREE.Group();
        if (pool && pool.length > 2) {
          const water = new THREE.ShapeGeometry(new THREE.Shape(pool.map(p => new THREE.Vector2(p[0], p[1])))); water.rotateX(-Math.PI / 2); water.translate(0, 0.12, 0);
          grp.add(new THREE.Mesh(water, new THREE.MeshLambertMaterial({ color: 0x4f9fc7, side: THREE.DoubleSide })));
          const rim = new THREE.BufferGeometry().setFromPoints([...pool, pool[0]].map(p => V(p[0], p[1], 0.2)));
          grp.add(new THREE.Line(rim, new THREE.LineBasicMaterial({ color: 0xe6e2d6 })));
        }
        const stone = new THREE.MeshLambertMaterial({ color: 0xd9d4c6 }), bronze = new THREE.MeshLambertMaterial({ color: 0x57462f });
        const at = (geo, mat, x, y, z) => { const m = new THREE.Mesh(geo, mat); m.position.set(mon.x + x, z, -(mon.y + y)); grp.add(m); return m; };
        at(new THREE.CylinderGeometry(6.5, 7.2, 1.4, 32), stone, 0, 0, 0.7);
        at(new THREE.CylinderGeometry(1.9, 3.0, 11, 24), stone, 0, 0, 6.9);
        for (const [dx, h] of [[-0.95, 4.9], [0.95, 5.4]]) {
          at(new THREE.CapsuleGeometry(0.62, h - 1.6, 4, 12), bronze, dx, 0, 12.4 + h / 2 - 0.2);
          at(new THREE.SphereGeometry(0.46, 14, 10), bronze, dx, 0, 12.4 + h + 0.3);
          const arm = at(new THREE.CapsuleGeometry(0.19, 2.2, 3, 8), bronze, dx * 1.75, 0, 12.4 + h + 0.4);
          arm.rotation.z = -Math.sign(dx) * 0.42;
        }
        return grp;
      };

      scene.add(laneStrips(G.lanes.filter(l => !isRaised(l)), 0.03));
      const toll = laneStrips(G.lanes.filter(isRaised), 0.03); flyMats.push(toll.material); scene.add(toll);
      scene.add(laneLines(G.lanes));
      scene.add(junctionMeshes(G.junctions));
      if (G.flyover) scene.add(flyover(G.flyover));
      if (G.mrt) scene.add(metro(G.mrt));
      scene.add(buildingMesh(G.buildings));
      scene.add(furniture(G.stoplines, G.heads));
      if (G.monument) scene.add(landmark(G.pool, G.monument));
      if (G.hospital) {                                       // a red cross over RSUP Fatmawati
        const cross = new THREE.Group(), red = new THREE.MeshBasicMaterial({ color: 0xe02020 });
        cross.add(new THREE.Mesh(new THREE.BoxGeometry(14, 2.5, 3.5), red), new THREE.Mesh(new THREE.BoxGeometry(3.5, 2.5, 14), red));
        cross.position.copy(V(G.hospital.x, G.hospital.y, 34)); scene.add(cross);
      }

      // vehicles: instanced, several parts per type; one set of meshes per recording so two can be drawn side by side
      const box = (lx, ly, lz, x, y, z = 0) => { const g = new THREE.BoxGeometry(lx, ly, lz); g.translate(x, y, z); return g; };
      const wheels = (xs, half, r, w) => { const gs = [];
        for (const x of xs) for (const z of (half ? [-half, half] : [0])) { const g = new THREE.CylinderGeometry(r, r, w, 14); g.rotateX(Math.PI / 2); g.translate(x, r, z); gs.push(g); }
        return mergeGeometries(gs); };
      const ball = (r, x, y) => { const g = new THREE.SphereGeometry(r, 12, 9); g.translate(x, y, 0); return g; };
      const DARK = new THREE.MeshLambertMaterial({ color: 0x1c1e22 }), GLASS = new THREE.MeshLambertMaterial({ color: 0x2c3846 });
      this.LIGHTS = new THREE.MeshBasicMaterial({ color: 0xff2020 });
      const TYPES = this.TYPES = { 0: { L: 4.5 }, 1: { L: 6.0 }, 2: { L: 2.0 }, 3: { L: 10.0 }, 4: { L: 8.0 }, 5: { L: 20.0 } };
      const PARTS = {
        0: [{ geo: mergeGeometries([box(4.3, 0.66, 1.76, 0, 0.6), box(1.95, 0.08, 1.44, -0.3, 1.5)]), paint: true },
            { geo: box(2.15, 0.56, 1.58, -0.3, 1.2), mat: GLASS }, { geo: wheels([1.38, -1.38], 0.79, 0.31, 0.22), mat: DARK }],
        1: [{ geo: box(5.8, 2.05, 2.05, 0, 1.4), mat: new THREE.MeshLambertMaterial({ color: 0xffd400, emissive: 0x332a00 }) },
            { geo: box(5.84, 0.3, 2.08, 0, 1.2), mat: new THREE.MeshBasicMaterial({ color: 0xd93c3c }) },
            { geo: box(0.06, 0.75, 1.85, 2.9, 1.9), mat: GLASS }, { geo: box(0.45, 0.22, 1.7, 1.6, 2.53), mat: this.LIGHTS },
            { geo: wheels([1.95, -1.95], 0.9, 0.37, 0.26), mat: DARK }],
        2: [{ geo: mergeGeometries([box(1.2, 0.34, 0.3, 0.02, 0.55), box(0.3, 0.6, 0.4, -0.22, 1.12)]), paint: true },
            { geo: mergeGeometries([wheels([0.68, -0.68], 0, 0.29, 0.11), box(0.06, 0.06, 0.72, 0.42, 1.0), box(0.55, 0.1, 0.28, -0.3, 0.77), ball(0.17, -0.16, 1.58)]), mat: DARK }],
        3: [{ geo: box(9.8, 2.55, 2.5, 0, 1.78), paint: true }, { geo: box(9.84, 0.85, 2.54, 0, 2.25), mat: GLASS }, { geo: wheels([3.3, -3.0], 1.1, 0.48, 0.3), mat: DARK }],
        4: [{ geo: mergeGeometries([box(2.0, 2.15, 2.3, 2.9, 1.45), box(5.6, 2.6, 2.4, -1.0, 1.65)]), paint: true },
            { geo: box(0.06, 0.8, 2.0, 3.91, 1.95), mat: GLASS }, { geo: wheels([2.9, -0.7, -2.7], 1.05, 0.45, 0.3), mat: DARK }],
        5: [{ geo: box(19.6, 2.95, 2.9, 0, 2.38), mat: new THREE.MeshLambertMaterial({ color: 0xdfe3e8 }) }, { geo: box(19.64, 0.95, 2.94, 0, 2.75), mat: GLASS },
            { geo: box(19.64, 0.32, 2.94, 0, 1.78), mat: new THREE.MeshLambertMaterial({ color: 0x1d3f8c }) },
            { geo: box(19.64, 0.08, 2.94, 0, 2.06), mat: new THREE.MeshLambertMaterial({ color: 0x2aa7df }) },
            { geo: mergeGeometries([box(2.6, 0.75, 2.3, 6.4, 0.45), box(2.6, 0.75, 2.3, -6.4, 0.45), box(2.2, 0.3, 1.2, -3.5, 4.0)]), mat: DARK }],
      };
      // a beacon over each ambulance, so it can be found from any camera
      const beaconGeo = mergeGeometries([new THREE.ConeGeometry(2.6, 6, 4).rotateX(Math.PI).translate(0, 12, 0), new THREE.TorusGeometry(7, 0.35, 8, 40).rotateX(Math.PI / 2).translate(0, 0.25, 0)]);
      const beaconMat = this.beaconMat = new THREE.MeshBasicMaterial({ color: 0xffd400, transparent: true, opacity: 0.9, depthTest: false, fog: false });
      this.sets = this.clips.map(c => {
        const meshes = {}, group = new THREE.Group();
        for (const code in PARTS) {
          const max = Math.max(4, (c.max && c.max[code]) || 0);
          meshes[code] = PARTS[code].map(pt => {
            const m = new THREE.InstancedMesh(pt.geo, pt.paint ? new THREE.MeshLambertMaterial() : pt.mat, max);
            m.count = 0; m.frustumCulled = false; m.userData.paint = !!pt.paint; m.userData.max = max; group.add(m); return m;
          });
        }
        const beacon = new THREE.InstancedMesh(beaconGeo, beaconMat, 8); beacon.count = 0; beacon.frustumCulled = false; beacon.renderOrder = 5; group.add(beacon);
        scene.add(group);
        return { clip: c, meshes, group, beacon, evPos: null };
      });
      const carPalette = [0xf4f4f4, 0xf4f4f4, 0xd0d3d8, 0x2b2d31, 0x8a8f97, 0xb23a3a, 0x3a5fb2, 0xe8e4d8].map(c => new THREE.Color(c));
      const jackets = [0x3fb950, 0x2f6fed, 0xe8833a, 0x9a59d1, 0xd13f6b, 0x1f8f8f, 0x444a55].map(c => new THREE.Color(c));
      const busCols = [0x2a6fd6, 0xe0782b, 0x2f9e75].map(c => new THREE.Color(c)), grey = new THREE.Color(0x8b8f96);
      this.colorFor = (code, id) => { const h = (id * 2654435761) >>> 0;
        return code === 0 ? carPalette[h % carPalette.length] : code === 2 ? jackets[h % jackets.length] : code === 3 ? busCols[h % busCols.length] : grey; };
      this.tmp = { M: new THREE.Matrix4(), Q: new THREE.Quaternion(), P: new THREE.Vector3(), one: new THREE.Vector3(1, 1, 1), up: new THREE.Vector3(0, 1, 0), S: new THREE.Vector3(), I: new THREE.Quaternion() };

      const h = G.hospital || { x: 100, y: -300 }, mon = G.monument || { x: 0, y: 0 }, en = (G.entries && G.entries.N) || { x: 10, y: 60, ux: 0, uy: -1 };
      this.CAMS = this.site === 'fat' ? {
        overview: { pos: V(-470, -420, 400), tgt: V(20, -60, 0) },
        junction: { pos: V(-95, -110, 85), tgt: V(0, 15, 0) },
        closeup: { pos: V(50, -18, 13), tgt: V(18, 4, 0) },
        flyover: { pos: V(95, -75, 38), tgt: V(-15, 30, 3) },
        north: { pos: V(10, -100, 150), tgt: V(6, 62, 0) },          // from the south, looking up Jl. RS Fatmawati over both crossings
        hospital: { pos: V(h.x - 120, h.y - 140, 110), tgt: V(h.x, h.y, 0) },
      } : {
        overview: { pos: V(-520, -480, 470), tgt: V(0, 0, 0) },
        junction: { pos: V(-8, -178, 96), tgt: V(0, -8, 0) },
        monument: { pos: V(mon.x + 30, mon.y - 32, 9), tgt: V(mon.x, mon.y, 13) },
        thamrin: { pos: V(en.x - en.ux * 34 + 11, en.y - en.uy * 34, 11), tgt: V(en.x + en.ux * 20, en.y + en.uy * 20, 0) },
      };
      this.buildUi();
      new ResizeObserver(() => this.resize()).observe(this.root);
    }

    // ------------------------------------------------------------ controls and HUD
    buildUi() {
      const cfg = this.cfg, two = this.clips.length > 1, bar = this.q('.twin-bar');
      const camOpts = cfg.cams.filter(([k]) => k !== 'follow' || this.clips[0].ev).map(([k, l]) => `<option value="${k}">${l}</option>`).join('');
      bar.innerHTML = `
        <button class="tb play" title="Play or pause (P)">❚❚</button>
        <input class="scrub" type="range" min="0" max="1000" value="0" aria-label="Position in the recording">
        <span class="clock">--:--:--</span>
        <label class="tl">speed <select class="speed"><option>1</option><option>2</option><option selected>4</option><option>8</option></select>×</label>
        ${two ? `<span class="seg mode"><button data-mode="a">${this.clips[0].short}</button><button data-mode="b">${this.clips[1].short}</button><button data-mode="split">Side by side</button></span>` : ''}
        <label class="tl">camera <select class="cam">${camOpts}<option value="" hidden>free</option></select></label>
        ${this.G.flyover ? '<button class="tb xray" title="Make the toll flyover and the MRT viaduct see-through">See-through</button>' : ''}
        <button class="tb live" title="Switch to the live twin when its server is running on this machine">Live</button>`;
      const b = s => bar.querySelector(s);
      b('.play').onclick = () => this.setPlaying(!this.playing);
      b('.scrub').oninput = e => { this.tau = e.target.value / 1000 * (this.clips[0].n - 1); };
      b('.scrub').onchange = e => e.target.blur();
      b('.speed').onchange = e => { this.speed = +e.target.value; e.target.blur(); };
      b('.cam').onchange = e => { this.setCam(e.target.value); e.target.blur(); };
      if (two) bar.querySelectorAll('.mode button').forEach(x => x.onclick = () => this.setMode(x.dataset.mode));
      if (b('.xray')) b('.xray').onclick = () => this.setXray(!this.xray);
      b('.live').onclick = () => this.toggleLive();
      this.q('.twin-huds').innerHTML = [0, 1].map(i => `<div class="hud" data-pane="${i}"><div class="hud-name"></div>
        <div class="hud-stage"><i class="lamp"></i><span></span></div><div class="hud-q"></div><div class="hud-ev" hidden></div></div>`).join('');
      this.huds = [...this.root.querySelectorAll('.hud')];
    }
    setPlaying(on) { this.playing = on; const p = this.q('.play'); if (p) { p.textContent = on ? '❚❚' : '▶'; } }
    setMode(mode) {
      if (!this.ready) return;
      if (this.clips.length < 2) mode = 'a';
      this.mode = mode;
      this.root.dataset.mode = mode;
      this.root.querySelectorAll('.mode button').forEach(x => x.classList.toggle('on', x.dataset.mode === mode));
      this.panes = mode === 'split' ? [0, 1] : [mode === 'b' ? 1 : 0];
      this.sets.forEach((s, i) => { s.group.visible = this.panes.includes(i); });
      this.huds.forEach((h, i) => { h.hidden = i >= this.panes.length; });
      this.resize();
    }
    setCam(name) {
      if (!this.ready) return;
      this.cam = name;
      const sel = this.q('.cam'); if (sel) sel.value = name || '';
      if (name && this.CAMS[name]) { this.camera.position.copy(this.CAMS[name].pos); this.controls.target.copy(this.CAMS[name].tgt); }
      this.followJump = name === 'follow';
    }
    setXray(on) {
      this.xray = on;
      for (const mt of this.flyMats) {
        const base = mt.userData.opacity ??= mt.opacity;
        mt.transparent = on || base < 1; mt.opacity = on ? Math.min(0.16, base) : base; mt.depthWrite = !on && base >= 1; mt.needsUpdate = true;
      }
      this.q('.xray').classList.toggle('on', on);
    }
    async toggleLive() {
      if (this.live) return this.setLive(false);
      const pop = this.q('.twin-pop'), btn = this.q('.live');
      btn.textContent = 'Checking…';
      let up = false;
      try { await fetch(this.cfg.live + 'health', { mode: 'no-cors', cache: 'no-store', signal: AbortSignal.timeout(1500) }); up = true; } catch (e) { up = false; }
      btn.textContent = 'Live';
      if (up) return this.setLive(true);
      pop.innerHTML = `<b>The live twin is not running on this machine.</b><span>Start it in a terminal:</span><code>${this.cfg.liveCmd}</code>
        <span>${this.cfg.liveNote}</span><button class="tb">Close</button>`;
      pop.hidden = false; pop.querySelector('button').onclick = () => { pop.hidden = true; };
    }
    setLive(on) {
      this.live = on;
      const fr = this.q('.twin-live');
      fr.hidden = !on; fr.src = on ? this.cfg.live + '?panel=0' : 'about:blank';
      this.root.classList.toggle('is-live', on);
      const btn = this.q('.live'); if (btn) { btn.textContent = on ? 'Back to replay' : 'Live'; btn.classList.toggle('on', on); }
      if (this.renderer) this.renderer.setAnimationLoop(on || !this.active ? null : () => this.tick());
    }

    resize() {
      if (!this.ready) return;
      const el = this.q('.twin-canvas'), w = el.clientWidth, h = el.clientHeight;
      if (!w || !h) return;
      const scale = el.getBoundingClientRect().width / w || 1;          // the slide stage is CSS-scaled
      this.renderer.setPixelRatio(Math.min(2.5, Math.max(0.75, (window.devicePixelRatio || 1) * scale)));
      this.renderer.setSize(w, h, false);
      this.camera.aspect = (this.mode === 'split' ? w / 2 : w) / h; this.camera.updateProjectionMatrix();
      this.size = [w, h];
    }
    start() { this.last = performance.now(); if (!this.live) this.renderer.setAnimationLoop(() => this.tick()); }

    // ------------------------------------------------------------ playback
    index(c, i) { let m = c.idx[i]; if (!m) { m = c.idx[i] = new Map(); const a = c.V[i]; for (let k = 0; k < a.length; k += 5) m.set(a[k], k); } return m; }
    drawSet(s, tau) {
      const c = s.clip, n = c.n, t = Math.min(tau, n - 1), i = Math.floor(t), j = Math.min(i + 1, n - 1), f = t - i;
      const A = c.V[i], B = c.V[j], ib = this.index(c, j), { M, Q, P, one, up, S, I } = this.tmp;
      const counts = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };
      let nb = 0; s.evPos = null;
      for (let k = 0; k < A.length; k += 5) {
        const id = A[k], code = A[k + 4] & 7;
        let x = A[k + 1] / 10, y = A[k + 2] / 10, a = A[k + 3], z = (A[k + 4] >> 3) / 10;
        const o = ib.get(id);
        if (o !== undefined && f > 0) {
          x += (B[o + 1] / 10 - x) * f; y += (B[o + 2] / 10 - y) * f; z += ((B[o + 4] >> 3) / 10 - z) * f;
          a += (((B[o + 3] - a + 540) % 360) - 180) * f;
        }
        const L = this.TYPES[code].L, rad = a * Math.PI / 180, dx = Math.sin(rad), dy = Math.cos(rad);   // SUMO angle: clockwise from north
        P.set(x - dx * L / 2, z, -(y - dy * L / 2));                                                  // the SUMO position is the front bumper
        Q.setFromAxisAngle(up, Math.PI / 2 - rad); M.compose(P, Q, one);
        const ms = s.meshes[code], kk = counts[code]++;
        if (kk < ms[0].userData.max) for (const m of ms) { m.setMatrixAt(kk, M); if (m.userData.paint) m.setColorAt(kk, this.colorFor(code, id)); }
        if (code === 1 && nb < 8) {
          const pulse = 1 + 0.12 * Math.sin(performance.now() / 180);
          S.set(pulse, 1, pulse); M.compose(P, I, S); s.beacon.setMatrixAt(nb++, M);
          if (!s.evPos || (c.ev && id === c.ev.id)) s.evPos = P.clone();
        }
      }
      for (const code in counts) for (const m of s.meshes[code]) {
        m.count = Math.min(counts[code], m.userData.max); m.instanceMatrix.needsUpdate = true; if (m.instanceColor) m.instanceColor.needsUpdate = true;
      }
      s.beacon.count = nb; s.beacon.instanceMatrix.needsUpdate = true;
    }
    signals(s, tau) {
      const c = s.clip, F = c.F[Math.min(Math.round(tau), c.n - 1)];
      let st = c.tlsCache.get(F[0]); if (!st) { st = c.tls[F[0]].split('|'); c.tlsCache.set(F[0], st); }
      for (const h of this.heads) {
        const ch = (st[h.t] || '')[h.links[0]] || 'r';
        const on = ch === 'G' || ch === 'g' ? 'g' : ch === 'y' || ch === 'Y' ? 'y' : 'r';
        for (const k of ['r', 'y', 'g']) h.lamps[k].m.color.setHex(k === on ? h.lamps[k].c : 0x222222);
      }
      return F;
    }
    hud(pane, s, F, tau) {
      const c = s.clip, el = this.huds[pane], legs = this.cfg.legs, fi = Math.min(Math.round(tau), c.n - 1);
      const key = fi + ':' + c.key; if (el.dataset.k === key) return; el.dataset.k = key;
      el.querySelector('.hud-name').textContent = c.label;
      const stage = el.querySelector('.hud-stage');
      if (c.ph && c.ph.length) {
        const pn = c.ph[F[1]] || '', kind = pn.endsWith('green') ? 'g' : pn.endsWith('amber') ? 'y' : 'r';
        stage.hidden = false; stage.dataset.k = kind;
        stage.querySelector('span').textContent = pn.endsWith('green') ? `${pn.replace(' green', '')} green · ${c.stages[pn.split(' ')[0]] || ''}` : pn.replace('>', ' → ');
      } else stage.hidden = true;
      const q = legs.map((a, i) => F[2 + i]), mx = Math.max(20, ...q);
      el.querySelector('.hud-q').innerHTML = `<div class="hud-cap">${c.qLabel}</div>` + legs.map((a, i) =>
        `<div class="qrow"><b>${a}</b><i><u style="width:${(100 * q[i] / mx).toFixed(0)}%"></u></i><span>${q[i]}</span></div>`).join('');
      const ev = el.querySelector('.hud-ev');
      if (c.ev) {
        const e = c.ev, dt = c.dt;
        let txt = '', cls = '';
        if (fi < e.detect) { txt = 'Ambulance on its way'; cls = 'wait'; }
        else if (fi < e.cross) { txt = `Ambulance ${e.from}: ${Math.round((fi - e.detect) * dt)} s since 300 m out`; cls = 'run'; }
        else { txt = `Ambulance through the junction in ${e.seconds} s`; cls = 'done'; }
        ev.hidden = false; ev.textContent = txt; ev.dataset.k = cls;
      } else ev.hidden = true;
    }
    tick() {
      const now = performance.now(), dt = Math.min(0.1, (now - this.last) / 1000); this.last = now;
      const c0 = this.clips[0], n = c0.n;
      if (this.playing) { this.tau += dt * this.speed / c0.dt; if (this.tau >= n - 1) this.tau = 0; }
      const tau = this.tau, [w, h] = this.size || [1, 1], r = this.renderer;
      this.controls.update();
      this.LIGHTS.color.setHex(Math.floor(now / 350) % 2 ? 0xff2020 : 0x2060ff);
      const scrub = this.q('.scrub'); if (document.activeElement !== scrub) scrub.value = Math.round(1000 * tau / (n - 1));
      const secs = c0.clock0 + tau * c0.dt, hh = Math.floor(secs / 3600) % 24, mm = Math.floor(secs % 3600 / 60), ss = Math.floor(secs % 60);
      this.q('.clock').textContent = `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
      for (const p of this.panes) this.drawSet(this.sets[p], tau);
      if (this.cam === 'follow') {
        const tgt = this.sets[this.panes[0]].evPos;
        if (tgt && this.followJump) { this.controls.target.copy(tgt); this.camera.position.copy(tgt).add(new T.THREE.Vector3(-34, 26, 34)); this.followJump = false; }
        else if (tgt) { const d = tgt.clone().sub(this.controls.target).multiplyScalar(1 - Math.exp(-dt / 0.15)); this.controls.target.add(d); this.camera.position.add(d); }
      }
      if (this.panes.length === 2) {
        r.setScissorTest(true);
        this.panes.forEach((p, k) => {
          this.sets.forEach((s, i) => { s.group.visible = i === p; });
          const F = this.signals(this.sets[p], tau); this.hud(k, this.sets[p], F, tau);
          const x = k * Math.floor(w / 2); r.setViewport(x, 0, Math.floor(w / 2), h); r.setScissor(x, 0, Math.floor(w / 2), h);
          r.render(this.scene, this.camera);
        });
        r.setScissorTest(false);
      } else {
        const p = this.panes[0], F = this.signals(this.sets[p], tau); this.hud(0, this.sets[p], F, tau);
        r.setViewport(0, 0, w, h); r.render(this.scene, this.camera);
      }
    }
  }

  const viewers = {};
  return {
    show(slot) { const site = slot.dataset.twin; (viewers[site] ||= new Viewer(site)).mount(slot); return viewers[site]; },
    hideAll(except) { for (const k in viewers) if (viewers[k] !== except) viewers[k].unmount(); },
    current() { return Object.values(viewers).find(v => v.active); },
  };
})();
