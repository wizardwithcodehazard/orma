/* =========================================================================
   ORMA // MEMORY CONNECTOME — viz.js v3 "Hippocampus"
   A particle cloud condenses into a cortex; the seahorse-shaped hippocampus
   kindles inside it, and memories bloom out of it.
     • Hippocampus  — active facts, events, conditional memories (+ entity hubs)
     • Amygdala     — forbidden / inhibitory rules (aversive memory)
     • Dentate gyrus— unconsolidated buffer (hippocampus[] from the API)
     • Fornix       — output arch; replay travels along it
     • Neocortex    — superseded memories, archived and fading
   Sleep replay fires sharp-wave ripples from the hippocampus through cortex.

   Data contract is unchanged:
     GET  /_orma/brain   -> { nodes, links, hippocampus }
     POST /_orma/sleep
     POST /_orma/encode  -> { text }
   ========================================================================= */
(() => {
    'use strict';

    const API = { brain: '/_orma/brain', sleep: '/_orma/sleep', encode: '/_orma/encode' };

    const $ = id => document.getElementById(id);
    const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
    const V3 = THREE.Vector3;
    const REDUCED = !!(window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches);
    const TAU = Math.PI * 2;

    /* ------------------------------------------------------------------ theme */
    const KIND = {
        user: { color: 0xff7a3d, shape: 5, size: 0.24 },
        entity: { color: 0x4fd1c5, shape: 1, size: 0.17 },
        inhibitory: { color: 0xff4d5e, shape: 2, size: 0.13 },
        conditional: { color: 0xc084fc, shape: 6, size: 0.10 },
        event: { color: 0x7f95ff, shape: 3, size: 0.11 },
        superseded: { color: 0xc9a27e, shape: 4, size: 0.16 },
        active: { color: 0xe8b84a, shape: 0, size: 0.085 }
    };
    const COL = {
        clear: 0x07080b,
        cortexA: 0x1a2142, cortexB: 0x7c8ce0,
        hippoA: 0x5a2410, hippoB: 0xffa24d,
        amygA: 0x3d0f17, amygB: 0xff4d5e,
        pulse: 0xffb46b, flow: 0xffc27a, mote: 0x7f95ff,
        hover: 0xa9977a, select: 0xff7a3d,
        dust: 0x262d52, grid: 0x191d2a
    };

    function hashStr(s) {
        let h = 2166136261;
        for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
        return (h >>> 0) / 4294967295;
    }
    const isUserNode = n => n.type === 'entity' && String(n.label).toLowerCase() === 'user';
    function kindOf(n) {
        if (isUserNode(n)) return 'user';
        if (n.status === 'inhibitory') return 'inhibitory';
        if (n.type === 'entity') return 'entity';
        if (n.status === 'conditional') return 'conditional';
        if (n.type === 'event') return 'event';
        if (n.status === 'superseded') return 'superseded';
        return 'active';
    }
    const linkId = x => (x && typeof x === 'object') ? x.id : x;
    const easeOutBack = t => { const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2); };
    const easeInOut = t => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

    /* --------------------------------------------------------------- renderer */
    const glCv = $('gl-cv');
    const R = new THREE.WebGLRenderer({ canvas: glCv, antialias: true, alpha: false, powerPreference: 'high-performance' });
    R.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    R.setSize(window.innerWidth, window.innerHeight, false);
    R.setClearColor(COL.clear, 1);

    const S = new THREE.Scene();
    const C = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.1, 140);

    const U = {
        uScale: { value: 1 },
        uTime: { value: 0 },
        uFocus: { value: 9 },
        uDpr: { value: R.getPixelRatio() },
        uForm: { value: REDUCED ? 100 : 0 },
        uRippleR: { value: -10 },
        uRippleA: { value: 0 },
        uRippleO: { value: new V3(0, -0.4, 0.1) },
        uColR: { value: new THREE.Color(COL.hippoB) }
    };
    function updateScale() { U.uScale.value = R.domElement.height / (2 * Math.tan(C.fov * Math.PI / 360)); }
    updateScale();

    /* ---------------------------------------------------------------- shaders */
    const POINT_VS = `
attribute vec3 aColor;
attribute float aSize;
attribute float aShape;
attribute float aAlpha;
attribute float aPhase;
uniform float uScale;
uniform float uTime;
uniform float uFocus;
uniform float uDpr;
varying vec3 vColor;
varying float vShape;
varying float vAlpha;
varying float vPx;
void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float depth = max(-mv.z, 0.1);
    float breathe = 1.0 + 0.05 * sin(uTime * 1.3 + aPhase * 6.2831);
    float px = aSize * breathe * uScale / depth;
    px = aSize > 0.0 ? clamp(px, 3.0 * uDpr, 110.0 * uDpr) : 0.0;
    gl_PointSize = px;
    gl_Position = projectionMatrix * mv;
    vColor = aColor;
    vShape = aShape;
    vPx = px;
    vAlpha = aAlpha * clamp(1.15 - (depth - uFocus) * 0.06, 0.3, 1.0);
}`;

    const POINT_FS = `
varying vec3 vColor;
varying float vShape;
varying float vAlpha;
varying float vPx;
float fillSD(float d, float aa) { return 1.0 - smoothstep(-aa, aa, d); }
float sdTri(vec2 p, float r) {
    float k = 1.7320508;
    p.x = abs(p.x) - r;
    p.y = p.y + r / k;
    if (p.x + k * p.y > 0.0) p = vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0;
    p.x -= clamp(p.x, -2.0 * r, 0.0);
    return -length(p) * sign(p.y);
}
float sdHex(vec2 p, float r) { p = abs(p); return max(p.x * 0.8660254 + p.y * 0.5, p.y) - r; }
void main() {
    vec2 p = (vec2(gl_PointCoord.x, 1.0 - gl_PointCoord.y) - 0.5) * 2.0;
    float aa = 2.0 / max(vPx, 1.0);
    float r = length(p);
    float a = 0.0;
    float s = vShape;
    if (s < 0.5) {                       // engram (active fact): bright cell body + halo
        a = max(max(fillSD(r - 0.42, aa), fillSD(abs(r - 0.62) - 0.03, aa) * 0.35), exp(-r * r * 5.0) * 0.25);
    } else if (s < 1.5) {                // concept cell (entity): soma ring + nucleus + 6 dendrites
        float ring = fillSD(abs(r - 0.5) - 0.07, aa);
        float core = fillSD(r - 0.2, aa);
        float ang = atan(p.y, p.x);
        float sd = abs(mod(ang + 0.5236, 1.0472) - 0.5236) * r;
        float spikes = fillSD(sd - 0.045, aa) * step(0.62, r) * step(r, 0.96);
        a = max(max(ring, core), max(spikes * 0.9, exp(-r * r * 6.0) * 0.2));
    } else if (s < 2.5) {                // inhibitory (forbidden): GABA-style ⊖ terminal
        float ring = fillSD(abs(r - 0.7) - 0.085, aa);
        float bar = fillSD(max(abs(p.x) - 0.44, abs(p.y) - 0.1), aa);
        a = max(max(ring, bar), exp(-r * r * 3.0) * 0.18);
    } else if (s < 3.5) {                // episode (event): grid-cell hexagonal firing lattice
        float d = length(p);
        for (int i = 0; i < 6; i++) {
            float th = float(i) * 1.0472;
            d = min(d, length(p - vec2(cos(th), sin(th)) * 0.58));
        }
        a = max(fillSD(d - 0.17, aa), fillSD(abs(sdHex(p, 0.84)) - 0.025, aa) * 0.35);
    } else if (s < 4.5) {                // decayed trace (superseded): fading echo
        float ang = atan(p.y, p.x);
        float dots = smoothstep(-0.1, 0.5, sin(ang * 6.0));
        float outer = fillSD(abs(r - 0.78) - 0.07, aa) * dots;
        float inner = fillSD(abs(r - 0.36) - 0.09, aa);
        float slash = fillSD(abs(p.x + p.y) * 0.7071 - 0.07, aa) * step(r, 0.62) * step(0.2, r);
        a = max(max(outer, inner), max(slash * 0.85, exp(-r * r * 4.0) * 0.2));
    } else if (s < 5.5) {                // user: nucleus + orbits
        float core = fillSD(r - 0.30, aa);
        float ring1 = fillSD(abs(r - 0.58) - 0.05, aa);
        float ring2 = fillSD(abs(r - 0.90) - 0.035, aa) * 0.6;
        a = max(max(core, ring1), max(ring2, exp(-r * r * 6.0) * 0.25));
    } else if (s < 6.5) {                // context-gated (conditional): half-open gate ◐
        float ring = fillSD(abs(r - 0.6) - 0.07, aa);
        float gate = fillSD(r - 0.6, aa) * (1.0 - smoothstep(-aa, aa, p.x));
        a = max(ring, gate);
    } else {                             // reticle brackets
        vec2 q = abs(p);
        float box = max(q.x, q.y);
        float frame = fillSD(box - 0.88, aa) - fillSD(box - 0.72, aa);
        a = frame * step(0.42, min(q.x, q.y));
    }
    gl_FragColor = vec4(vColor, a * vAlpha);
    if (gl_FragColor.a < 0.01) discard;
}`;

    const LINE_VS = `
attribute vec3 aColor;
attribute float aAlpha;
uniform float uFocus;
varying vec3 vColor;
varying float vAlpha;
void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float depth = max(-mv.z, 0.1);
    vColor = aColor;
    vAlpha = aAlpha * clamp(1.2 - (depth - uFocus) * 0.06, 0.25, 1.0);
    gl_Position = projectionMatrix * mv;
}`;
    const LINE_FS = `
varying vec3 vColor;
varying float vAlpha;
void main() { gl_FragColor = vec4(vColor, vAlpha); }`;

    /* brain particle cloud: flies from a nebula into anatomy, carries ripples */
    const BRAIN_VS = `
attribute vec3 aStart;
attribute float aDelay;
attribute float aDur;
attribute float aShade;
attribute float aSize;
uniform float uScale;
uniform float uTime;
uniform float uDpr;
uniform float uForm;
uniform float uRippleR;
uniform float uRippleA;
uniform vec3 uRippleO;
uniform vec3 uColR;
uniform vec3 uColA;
uniform vec3 uColB;
uniform float uAlpha;
uniform float uPre;
varying vec3 vCol;
varying float vA;
void main() {
    float p = clamp((uForm - aDelay) / aDur, 0.0, 1.0);
    float e = 1.0 - pow(1.0 - p, 3.0);
    vec3 drift = vec3(sin(uTime * 0.6 + aDelay * 13.0), cos(uTime * 0.5 + aDelay * 7.0), sin(uTime * 0.4 + aDelay * 5.0)) * 0.006;
    vec3 pos = mix(aStart, position + drift, e);
    float sw = (1.0 - e) * 1.6;
    float cs = cos(sw), sn = sin(sw);
    pos.xz = mat2(cs, -sn, sn, cs) * pos.xz;
    vec4 mv = modelViewMatrix * vec4(pos, 1.0);
    float depth = max(-mv.z, 0.1);
    gl_PointSize = clamp(aSize * (1.0 + (1.0 - e) * 1.2) * uScale / depth, 1.0 * uDpr, 6.0 * uDpr);
    gl_Position = projectionMatrix * mv;
    float d = length(position - uRippleO);
    float rip = exp(-pow((d - uRippleR) * 4.0, 2.0)) * uRippleA;
    vCol = mix(uColA, uColB, aShade) + uColR * rip * 0.8;
    float pre = mix(uPre, 1.0, smoothstep(0.0, 0.25, p));
    vA = uAlpha * pre * (mix(0.75, 1.0, e) * mix(0.3, 1.0, aShade) + rip * 0.7);
}`;
    const BRAIN_FS = `
varying vec3 vCol;
varying float vA;
void main() {
    float r = length(gl_PointCoord - 0.5) * 2.0;
    float a = smoothstep(1.0, 0.0, r);
    gl_FragColor = vec4(vCol, a * vA);
    if (gl_FragColor.a < 0.004) discard;
}`;

    function pointMat(additive) {
        return new THREE.ShaderMaterial({
            uniforms: U, vertexShader: POINT_VS, fragmentShader: POINT_FS,
            transparent: true, depthWrite: false,
            blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending
        });
    }
    function brainMat(a, b, alpha, pre) {
        return new THREE.ShaderMaterial({
            uniforms: {
                uScale: U.uScale, uTime: U.uTime, uDpr: U.uDpr, uForm: U.uForm,
                uRippleR: U.uRippleR, uRippleA: U.uRippleA, uRippleO: U.uRippleO, uColR: U.uColR,
                uColA: { value: new THREE.Color(a) }, uColB: { value: new THREE.Color(b) }, uAlpha: { value: alpha }, uPre: { value: pre }
            },
            vertexShader: BRAIN_VS, fragmentShader: BRAIN_FS,
            transparent: true, depthWrite: false, blending: THREE.AdditiveBlending
        });
    }
    function makePointGeo(n) {
        const g = new THREE.BufferGeometry();
        const dyn = (arr, size) => { const a = new THREE.BufferAttribute(arr, size); a.setUsage(THREE.DynamicDrawUsage); return a; };
        g.setAttribute('position', dyn(new Float32Array(n * 3), 3));
        g.setAttribute('aColor', dyn(new Float32Array(n * 3), 3));
        g.setAttribute('aSize', dyn(new Float32Array(n), 1));
        g.setAttribute('aShape', dyn(new Float32Array(n), 1));
        g.setAttribute('aAlpha', dyn(new Float32Array(n), 1));
        g.setAttribute('aPhase', dyn(new Float32Array(n), 1));
        return g;
    }
    function fillColor(geo, hex) {
        const c = new THREE.Color(hex), A = geo.attributes.aColor.array;
        for (let i = 0; i < A.length; i += 3) { A[i] = c.r; A[i + 1] = c.g; A[i + 2] = c.b; }
    }

    /* ------------------------------------------------------- brain anatomy */
    // Coordinates: x = left/right, y = up, z = anterior (+) / posterior (-)
    const HIP_PTS = [[0.66, -0.62, 0.85], [0.70, -0.58, 0.35], [0.64, -0.45, -0.15], [0.50, -0.20, -0.60], [0.34, 0.12, -0.78]];
    const FX_PTS = [[0.34, 0.12, -0.78], [0.20, 0.42, -0.55], [0.09, 0.55, -0.05], [0.07, 0.32, 0.40], [0.07, -0.10, 0.30]];
    const HC = {};
    [-1, 1].forEach(s => {
        const mk = pts => new THREE.CatmullRomCurve3(pts.map(([x, y, z]) => new V3(s * x, y, z)), false, 'centripetal');
        HC[s] = { hip: mk(HIP_PTS), fx: mk(FX_PTS), amyg: new V3(s * 0.68, -0.66, 1.18) };
    });
    const hipR = t => t < 0.25 ? 0.13 + (0.25 - t) * 0.28 : 0.13 * (1 - (t - 0.25) * 0.75);

    const UP = new V3(0, 1, 0), _T = new V3(), _N = new V3(), _B = new V3();
    function tubePoint(curve, t, ang, r, out) {
        out = out || new V3();
        curve.getPointAt(t, out);
        curve.getTangentAt(t, _T);
        _N.crossVectors(_T, UP);
        if (_N.lengthSq() < 1e-6) _N.set(1, 0, 0);
        _N.normalize();
        _B.crossVectors(_T, _N).normalize();
        return out.addScaledVector(_N, Math.cos(ang) * r).addScaledVector(_B, Math.sin(ang) * r);
    }

    function gyri(x, y, z) {
        const a = Math.sin(x * 6.1 + Math.sin(y * 4.3) * 1.6);
        const b = Math.sin(y * 5.7 + Math.sin(z * 3.9) * 1.5);
        const c = Math.sin(z * 4.9 + Math.sin(x * 5.1) * 1.4);
        return 1 - Math.abs((a + b + c) / 3);
    }
    function inHemi(p, s) {
        const lx = p.x - s * 0.2, ax = (s * lx > 0) ? 1.25 : 0.14;
        const ly = p.y - 0.25, ay = ly > 0 ? 1.3 : 1.0;
        return (lx / ax) ** 2 + (ly / ay) ** 2 + (p.z / 1.95) ** 2 < 0.96;
    }
    function inTemporal(p, s) {
        const dx = (p.x - s * 0.95) / 0.55, dy = (p.y + 0.55) / 0.5, dz = (p.z - 0.35) / 1.05;
        return dx * dx + dy * dy + dz * dz < 1;
    }
    const inBrain = p => inHemi(p, 1) || inHemi(p, -1) || inTemporal(p, 1) || inTemporal(p, -1);
    function randDir(v) {
        const u = Math.random() * 2 - 1, th = Math.random() * TAU, k = Math.sqrt(1 - u * u);
        return v.set(Math.cos(th) * k, u, Math.sin(th) * k);
    }
    function nebula() {
        const v = randDir(new V3()).multiplyScalar(5 + Math.random() * 7);
        v.y *= 0.55;
        return v;
    }
    function cortexPoint(s, ux, uy, uz, shrink) {
        const lx = ux * (s * ux > 0 ? 1.25 : 0.14), ly = uy * (uy > 0 ? 1.3 : 1.0), lz = uz * 1.95;
        return new V3(s * 0.2 + lx * shrink, 0.25 + ly * shrink, lz * shrink);
    }

    function buildCloud(pts) {
        const n = pts.length;
        const P = new Float32Array(n * 3), St = new Float32Array(n * 3);
        const D = new Float32Array(n), Du = new Float32Array(n), Sh = new Float32Array(n), Sz = new Float32Array(n);
        pts.forEach((q, i) => {
            P[i * 3] = q.p.x; P[i * 3 + 1] = q.p.y; P[i * 3 + 2] = q.p.z;
            St[i * 3] = q.st.x; St[i * 3 + 1] = q.st.y; St[i * 3 + 2] = q.st.z;
            D[i] = q.d; Du[i] = q.du; Sh[i] = q.sh; Sz[i] = q.sz;
        });
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(P, 3));
        g.setAttribute('aStart', new THREE.BufferAttribute(St, 3));
        g.setAttribute('aDelay', new THREE.BufferAttribute(D, 1));
        g.setAttribute('aDur', new THREE.BufferAttribute(Du, 1));
        g.setAttribute('aShade', new THREE.BufferAttribute(Sh, 1));
        g.setAttribute('aSize', new THREE.BufferAttribute(Sz, 1));
        return g;
    }

    function sampleCortex() {
        const out = [], u = new V3();
        const push = (p, sh, sz, d) => out.push({ p, st: nebula(), d, du: 1.5 + Math.random() * 0.7, sh, sz });
        // hemispheres
        [-1, 1].forEach(s => {
            let got = 0, tries = 0;
            while (got < 7000 && tries < 60000) {
                tries++;
                randDir(u);
                const medial = s * u.x < 0;
                if (medial && Math.random() < 0.65) continue;
                let lx = u.x * (medial ? 0.14 : 1.25), ly = u.y * (u.y > 0 ? 1.3 : 1.0), lz = u.z * 1.95;
                if (lz > 1.0) lx *= 1 - (lz - 1.0) * 0.28;
                if (lz < -1.3) ly *= 1 - (-1.3 - lz) * 0.35;
                const g = gyri(s * 0.2 + lx, 0.25 + ly, lz);
                const k = medial ? 1 : 1 + 0.055 * (g - 0.55);
                const p = new V3(s * 0.2 + lx * k, 0.25 + ly * k, lz * k);
                if (inTemporal(p, s)) continue;
                push(p, Math.pow(g, 1.6) * (medial ? 0.55 : 1), 0.011, 0.15 + Math.random() * 1.25);
                got++;
            }
        });
        // temporal lobes
        [-1, 1].forEach(s => {
            let got = 0, tries = 0;
            while (got < 1500 && tries < 30000) {
                tries++;
                randDir(u);
                const g = gyri(s * 0.95 + u.x * 0.55, -0.55 + u.y * 0.5, 0.35 + u.z * 1.05);
                const k = 1 + 0.05 * (g - 0.55);
                const p = new V3(s * 0.95 + u.x * 0.55 * k, -0.55 + u.y * 0.5 * k, 0.35 + u.z * 1.05 * k);
                if (inHemi(p, s)) continue;
                push(p, Math.pow(g, 1.6), 0.011, 0.3 + Math.random() * 1.2);
                got++;
            }
        });
        // cerebellum (folia striations)
        {
            let got = 0, tries = 0;
            while (got < 2600 && tries < 30000) {
                tries++;
                randDir(u);
                const p = new V3(u.x * 1.05, -0.85 + u.y * 0.45, -1.55 + u.z * 0.6);
                if (inHemi(p, 1) || inHemi(p, -1)) continue;
                const fol = 0.5 + 0.5 * Math.sin(p.y * 42 + p.z * 8 + Math.sin(p.x * 3) * 2);
                push(p, 0.25 + 0.75 * fol * fol, 0.0095, 0.5 + Math.random() * 1.2);
                got++;
            }
        }
        // brainstem
        {
            let got = 0, tries = 0;
            const a = new V3(0, -0.45, -0.5), b = new V3(0, -1.85, -0.85);
            while (got < 650 && tries < 8000) {
                tries++;
                const t = Math.random(), r = 0.26 - 0.08 * t, ang = Math.random() * TAU;
                const c = a.clone().lerp(b, t);
                const p = new V3(c.x + Math.cos(ang) * r, c.y, c.z + Math.sin(ang) * r * 0.85);
                if (inBrain(p)) continue;
                push(p, 0.35 + 0.3 * Math.random(), 0.0095, 0.8 + Math.random() * 1.2);
                got++;
            }
        }
        return out;
    }

    function sampleHippocampus() {
        const out = [], tmp = new V3();
        [-1, 1].forEach(s => {
            const H = HC[s];
            for (let i = 0; i < 2300; i++) {
                const t = Math.pow(Math.random(), 0.85), ang = Math.random() * TAU;
                let r = hipR(t);
                if (t < 0.24) r *= 1 + 0.3 * Math.max(0, Math.sin(ang * 4 + 1.3)); // pes digitations
                const fill = Math.random() < 0.3 ? Math.sqrt(Math.random()) : 1;
                r *= fill * (0.9 + 0.2 * Math.random());
                out.push({ p: tubePoint(H.hip, t, ang, r), st: H.hip.getPointAt(t, tmp).clone(), d: 1.85 + t * 0.9, du: 0.8, sh: 0.45 + 0.55 * fill, sz: 0.013 });
            }
            for (let i = 0; i < 650; i++) {
                const t = Math.random(), ang = Math.random() * TAU, r = 0.032 * (0.6 + 0.4 * Math.random());
                out.push({ p: tubePoint(H.fx, t, ang, r), st: H.fx.getPointAt(t, tmp).clone(), d: 2.6 + t * 0.5, du: 0.6, sh: 0.22, sz: 0.011 });
            }
        });
        return out;
    }

    function sampleAmygdala() {
        const out = [], u = new V3();
        [-1, 1].forEach(s => {
            const c = HC[s].amyg;
            for (let i = 0; i < 420; i++) {
                randDir(u);
                const rr = 0.15 * Math.cbrt(Math.random());
                const p = new V3(c.x + u.x * rr, c.y + u.y * rr * 0.8, c.z + u.z * rr * 1.15);
                out.push({ p, st: c.clone(), d: 2.2 + Math.random() * 0.4, du: 0.7, sh: 0.3 + 0.7 * (1 - rr / 0.15), sz: 0.012 });
            }
        });
        return out;
    }

    /* ------------------------------------------------------------- scene bits */
    const dustGroup = new THREE.Group(), staticGroup = new THREE.Group(), cortexGroup = new THREE.Group(), limbicGroup = new THREE.Group();
    const linksGrp = new THREE.Group(), nodesGrp = new THREE.Group(), pulseGrp = new THREE.Group(), selGrp = new THREE.Group();
    S.add(dustGroup, staticGroup, cortexGroup, limbicGroup, linksGrp, nodesGrp, pulseGrp, selGrp);

    const cortexMat = brainMat(COL.cortexA, COL.cortexB, 0.62, 1);
    const hippoMat = brainMat(COL.hippoA, COL.hippoB, 0.85, 0);
    const amygMat = brainMat(COL.amygA, COL.amygB, 0.7, 0);
    (function buildBrain() {
        const cx = new THREE.Points(buildCloud(sampleCortex()), cortexMat);
        cx.frustumCulled = false; cx.renderOrder = 0;
        cortexGroup.add(cx);
        const hp = new THREE.Points(buildCloud(sampleHippocampus()), hippoMat);
        hp.frustumCulled = false; hp.renderOrder = 1;
        const am = new THREE.Points(buildCloud(sampleAmygdala()), amygMat);
        am.frustumCulled = false; am.renderOrder = 1;
        limbicGroup.add(hp, am);
    })();

    (function buildDust() {
        const n = 800, pos = new Float32Array(n * 3), v = new V3();
        for (let i = 0; i < n; i++) {
            randDir(v).multiplyScalar(16 + Math.random() * 20);
            pos[i * 3] = v.x; pos[i * 3 + 1] = v.y * 0.7; pos[i * 3 + 2] = v.z;
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        const pts = new THREE.Points(g, new THREE.PointsMaterial({ color: COL.dust, size: 1.4, sizeAttenuation: false, transparent: true, opacity: 0.55, depthWrite: false }));
        pts.frustumCulled = false;
        dustGroup.add(pts);
    })();

    (function buildGraticule() {
        const v = [], y = -2.25;
        [1.5, 3, 4.5, 6, 7.5].forEach(r => {
            const seg = 120;
            for (let i = 0; i < seg; i++) {
                const a0 = i / seg * TAU, a1 = (i + 1) / seg * TAU;
                v.push(Math.cos(a0) * r, y, Math.sin(a0) * r, Math.cos(a1) * r, y, Math.sin(a1) * r);
            }
        });
        for (let k = 0; k < 24; k++) {
            const a = k / 24 * TAU;
            v.push(Math.cos(a) * 0.6, y, Math.sin(a) * 0.6, Math.cos(a) * 7.5, y, Math.sin(a) * 7.5);
        }
        for (let d = 0; d < 360; d += 5) {
            const a = d * Math.PI / 180, len = d % 30 === 0 ? 0.28 : 0.1;
            v.push(Math.cos(a) * 7.5, y, Math.sin(a) * 7.5, Math.cos(a) * (7.5 + len), y, Math.sin(a) * (7.5 + len));
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(v), 3));
        const l = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: COL.grid, transparent: true, opacity: 0.55, depthWrite: false }));
        l.frustumCulled = false;
        staticGroup.add(l);
    })();

    /* ------------------------------------------------------------------ state */
    let rawData = null, allNodes = [], allLinks = [];
    let activeFilter = 'all', layoutMode = 'hippocampus', labelDensity = 'clean', brainOn = true;
    let selectedNode = null, hoveredNode = null, searchSet = null, searchList = [], searchCursor = -1;
    let vis = [], visLinks = [], focus = null, firstLoad = true;
    let nowT = 0, sleepStart = -99, lastInteract = 0;
    const nodeMap = new Map(), adj = new Map();
    const INTRO_NODES = REDUCED ? 0 : 2.9;

    const SEG = 10, PULSE_CAP = 36, FLOW_CAP = 30, MOTE_CAP = 64;
    let nodePts = null, nodeGeo = null, linkObj = null, linkGeo = null;

    /* synapse pulses */
    const pulses = [];
    const pulseGeo = makePointGeo(PULSE_CAP);
    (function initPulses() {
        fillColor(pulseGeo, COL.pulse);
        for (let i = 0; i < PULSE_CAP; i++) {
            pulseGeo.attributes.aSize.array[i] = 0.06;
            pulseGeo.attributes.aPhase.array[i] = Math.random();
            pulses.push({ idx: 0, prog: Math.random(), spd: 0.35 + Math.random() * 0.5 });
        }
        const p = new THREE.Points(pulseGeo, pointMat(true));
        p.frustumCulled = false; p.renderOrder = 4;
        pulseGrp.add(p);
    })();

    /* hippocampal flow: sparks running head → tail → fornix */
    const flows = [];
    const flowGeo = makePointGeo(FLOW_CAP);
    (function initFlow() {
        fillColor(flowGeo, COL.flow);
        for (let i = 0; i < FLOW_CAP; i++) {
            flowGeo.attributes.aSize.array[i] = 0.045;
            flowGeo.attributes.aPhase.array[i] = Math.random();
            flows.push({ s: i % 2 ? 1 : -1, u: Math.random() * 1.6, spd: 0.18 + Math.random() * 0.22, ang: Math.random() * TAU });
        }
        const p = new THREE.Points(flowGeo, pointMat(true));
        p.frustumCulled = false; p.renderOrder = 4;
        pulseGrp.add(p);
    })();

    /* dentate buffer motes (one cluster per unconsolidated item) */
    const motes = [];
    let moteCount = 0;
    const moteGeo = makePointGeo(MOTE_CAP);
    (function initMotes() {
        fillColor(moteGeo, COL.mote);
        for (let i = 0; i < MOTE_CAP; i++) {
            moteGeo.attributes.aSize.array[i] = 0.035;
            moteGeo.attributes.aPhase.array[i] = Math.random();
            motes.push({ home: new V3(), target: new V3(), ph: Math.random() * TAU });
        }
        const p = new THREE.Points(moteGeo, pointMat(true));
        p.frustumCulled = false; p.renderOrder = 4;
        pulseGrp.add(p);
    })();
    function seedMotes(n) {
        moteCount = Math.min(MOTE_CAP, n * 4);
        for (let i = 0; i < moteCount; i++) {
            const m = motes[i], s = i % 2 ? 1 : -1;
            tubePoint(HC[s].hip, 0.03 + Math.random() * 0.3, Math.random() * TAU, hipR(0.1) * (1.6 + Math.random() * 0.9), m.home);
            const ux = s * (0.4 + Math.random() * 0.5), uy = Math.random() * 0.9 - 0.1, uz = Math.random() * 1.6 - 0.8;
            const l = Math.hypot(ux, uy, uz);
            m.target.copy(cortexPoint(s, ux / l, uy / l, uz / l, 0.92));
        }
    }

    /* hover / selection reticles */
    const selGeo = makePointGeo(2);
    (function initSel() {
        const a = new THREE.Color(COL.hover), b = new THREE.Color(COL.select);
        selGeo.attributes.aColor.array.set([a.r, a.g, a.b, b.r, b.g, b.b]);
        selGeo.attributes.aShape.array.set([7, 7]);
        const p = new THREE.Points(selGeo, pointMat(false));
        p.frustumCulled = false; p.renderOrder = 5;
        selGrp.add(p);
    })();

    /* ---------------------------------------------------------------- layouts */
    function fib(i, n) {
        const y = n === 1 ? 0 : 1 - (i / (n - 1)) * 2;
        const rad = Math.sqrt(Math.max(0, 1 - y * y)), th = i * 2.399963;
        return new V3(Math.cos(th) * rad, y, Math.sin(th) * rad);
    }

    function hippocampusLayout() {
        const ents = allNodes.filter(n => n.type === 'entity');
        const user = ents.find(isUserNode);
        const peers = ents.filter(n => !isUserNode(n));
        if (user) { user.hippo = new V3(0, -0.08, 0.05); user.side = 0; }

        // entity hubs strung along both hippocampi
        const sides = { '-1': [], '1': [] };
        peers.forEach((e, i) => { e.side = i % 2 === 0 ? -1 : 1; sides[e.side].push(e); });
        [-1, 1].forEach(s => sides[s].forEach((e, k, L) => {
            const t = 0.1 + 0.8 * (k + 0.5) / L.length;
            e.hippo = tubePoint(HC[s].hip, t, (k * 2.3) % TAU, hipR(t) * 0.4);
        }));

        const userFacts = [], forb = { '-1': [], '1': [] }, sup = [], kids = new Map();
        allNodes.forEach(n => {
            if (n.type === 'entity') return;
            const par = n.parent;
            if (n.status === 'inhibitory') {
                const s = par && par.side ? par.side : (n.hash > 0.5 ? 1 : -1);
                n.side = s; forb[s].push(n); return;
            }
            if (n.status === 'superseded') { sup.push(n); return; }
            if (!par || par === user || !par.side) { userFacts.push(n); return; }
            if (!kids.has(par.id)) kids.set(par.id, []);
            kids.get(par.id).push(n);
        });

        // user's own memories: beads wrapping the hippocampal surface
        const half = Math.max(1, Math.ceil(userFacts.length / 2));
        userFacts.forEach((n, i) => {
            const s = i % 2 === 0 ? 1 : -1, k = Math.floor(i / 2);
            const t = 0.04 + 0.92 * (k + 0.5) / half;
            n.side = s;
            n.hippo = tubePoint(HC[s].hip, t, k * 2.399963 + (s > 0 ? 0 : 1.2), hipR(t) * 1.55 + 0.05);
        });
        // other entities' memories orbit their hub
        kids.forEach((list, pid) => {
            const hub = nodeMap.get(pid), c = hub.hippo, base = 0.13 + 0.03 * Math.sqrt(list.length);
            list.forEach((n, i) => {
                const d = fib(i, list.length);
                n.side = hub.side;
                n.hippo = new V3(c.x + d.x * base, c.y + d.y * base * 0.8, c.z + d.z * base);
            });
        });
        // forbidden rules in the amygdala
        [-1, 1].forEach(s => {
            const L = forb[s], c = HC[s].amyg, base = 0.08 + 0.035 * Math.sqrt(L.length);
            L.forEach((n, i) => { const d = fib(i, L.length); n.hippo = new V3(c.x + d.x * base, c.y + d.y * base * 0.8, c.z + d.z * base * 1.1); });
        });
        // superseded: archived into posterior cortex
        sup.forEach(n => {
            const s = n.hash > 0.5 ? 1 : -1;
            const h1 = hashStr(n.id + 'a'), h2 = hashStr(n.id + 'b'), h3 = hashStr(n.id + 'c');
            const ux = s * (0.55 + 0.4 * h1), uz = -(0.35 + 0.6 * h2), uy = (h3 - 0.3) * 0.8;
            const l = Math.hypot(ux, uy, uz);
            n.side = s;
            n.hippo = cortexPoint(s, ux / l, uy / l, uz / l, 0.9);
        });
        allNodes.forEach(n => { if (!n.hippo) n.hippo = new V3(0, 0, 0); });
    }

    function constellationLayout() {
        const ents = allNodes.filter(n => n.type === 'entity');
        const user = ents.find(isUserNode);
        const peers = ents.filter(n => !isUserNode(n));
        if (user) user.planet = new V3(0, 0, 0);
        const m = peers.length, Rr = 3.0 + Math.sqrt(m) * 0.42;
        peers.forEach((e, i) => {
            const y = m === 1 ? 0.15 : 1 - (i + 0.5) / m * 2;
            const rad = Math.sqrt(Math.max(0, 1 - y * y)), th = i * 2.399963 + 0.4;
            const rr = Rr * (0.88 + e.hash * 0.24);
            e.planet = new V3(Math.cos(th) * rad * rr, y * rr * 0.5, Math.sin(th) * rad * rr);
        });
        const children = new Map();
        allNodes.forEach(n => {
            if (n.type === 'entity') return;
            const parent = n.parent || user || ents[0] || null;
            const key = parent ? parent.id : '__none__';
            if (!children.has(key)) children.set(key, []);
            children.get(key).push(n);
        });
        children.forEach((list, pid) => {
            const parent = nodeMap.get(pid);
            const pPos = (parent && parent.planet) || new V3(0, 0, 0);
            const base = 0.75 + 0.1 * Math.sqrt(list.length);
            list.forEach((c, idx) => {
                const d = fib(idx, list.length);
                const sr = (c.status === 'inhibitory' ? base + 0.45 : base) * (0.9 + c.hash * 0.25);
                c.planet = new V3(pPos.x + d.x * sr, pPos.y + d.y * sr * 0.8, pPos.z + d.z * sr);
            });
        });
        allNodes.forEach(n => { if (!n.planet) n.planet = new V3(0, 0, 0); });
    }
    const targetOf = n => layoutMode === 'hippocampus' ? n.hippo : n.planet;

    /* ----------------------------------------------------------------- camera */
    const cam = { theta: 0.95, phi: 1.12, radius: 18, tTheta: 0.95, tPhi: 1.12, tRadius: 8.4, target: new V3(0, -0.15, 0), tTarget: new V3(0, -0.15, 0) };
    const sph = new THREE.Spherical(), tmpV = new V3(), tmpW = new V3();
    let dragging = false;

    function fitCamera() {
        let r;
        if (layoutMode === 'hippocampus') r = 7.6;
        else {
            let ext = 3;
            allNodes.forEach(n => { if (n.planet) ext = Math.max(ext, n.planet.length()); });
            r = ext * 1.9 + 1.2;
        }
        if (C.aspect < 1) r /= Math.max(0.5, C.aspect);
        cam.tRadius = clamp(r, 4, 26);
    }
    function touch() { lastInteract = nowT; }
    function flyTo(n) { if (!n || !n.cur) return; cam.tTarget.copy(n.cur); cam.tRadius = Math.min(cam.tRadius, 4.6); touch(); }
    function resetView() { cam.tTarget.set(0, -0.15, 0); cam.tPhi = 1.12; fitCamera(); touch(); toast('View reset'); }

    function updateCamera(dt) {
        if (!dragging && !selectedNode && !hoveredNode && !REDUCED && nowT - lastInteract > 3) cam.tTheta += dt * 0.05;
        const k = 1 - Math.exp(-dt * (nowT < 3.4 && !REDUCED ? 1.4 : 6));
        cam.theta += (cam.tTheta - cam.theta) * k;
        cam.phi += (cam.tPhi - cam.phi) * k;
        cam.radius += (cam.tRadius - cam.radius) * k;
        cam.target.lerp(cam.tTarget, k);
        sph.set(cam.radius, cam.phi, cam.theta);
        C.position.copy(cam.target).add(tmpV.setFromSpherical(sph));
        C.lookAt(cam.target);
        C.updateMatrixWorld();
        U.uFocus.value = cam.radius;
    }

    /* ------------------------------------------------------------ graph build */
    function passFilter(n) {
        switch (activeFilter) {
            case 'entity': return n.type === 'entity';
            case 'inhibitory': return n.status === 'inhibitory';
            case 'active': return n.status === 'active' && n.type === 'state';
            case 'event': return n.type === 'event';
            case 'superseded': return n.status === 'superseded';
            default: return true;
        }
    }

    function makeLink(s, t) {
        const h = hashStr(s.id + '>' + t.id);
        let base = 0.2;
        if (s.kind === 'inhibitory' || t.kind === 'inhibitory') base = 0.42;
        else if (s.kind === 'superseded' || t.kind === 'superseded') base = 0.16;
        else if (s.type === 'entity' && t.type === 'entity') base = 0.32;
        return { s, t, base, alpha: base, alphaT: base, sign: h > 0.5 ? 1 : -1, bend: 0.1 + h * 0.12, c: new V3() };
    }

    function refreshGraph() {
        allNodes.forEach(n => { n.vis = passFilter(n); n.tag = null; });
        vis = allNodes.filter(n => n.vis);
        const visSet = new Set(vis.map(n => n.id));

        visLinks = [];
        allLinks.forEach(l => {
            const sId = linkId(l.source), tId = linkId(l.target);
            if (visSet.has(sId) && visSet.has(tId)) visLinks.push(makeLink(nodeMap.get(sId), nodeMap.get(tId)));
        });

        if (nodePts) { nodesGrp.remove(nodePts); nodeGeo.dispose(); }
        const N = vis.length;
        nodeGeo = makePointGeo(N);
        const Co = nodeGeo.attributes.aColor.array, Sh = nodeGeo.attributes.aShape.array, Ph = nodeGeo.attributes.aPhase.array;
        vis.forEach((n, i) => {
            Co[i * 3] = n.rgb.r; Co[i * 3 + 1] = n.rgb.g; Co[i * 3 + 2] = n.rgb.b;
            Sh[i] = n.shape; Ph[i] = n.hash;
        });
        nodePts = new THREE.Points(nodeGeo, pointMat(false));
        nodePts.frustumCulled = false; nodePts.renderOrder = 3; nodePts.visible = N > 0;
        nodesGrp.add(nodePts);

        if (linkObj) { linksGrp.remove(linkObj); linkGeo.dispose(); }
        const vc = visLinks.length * SEG * 2;
        linkGeo = new THREE.BufferGeometry();
        const lp = new Float32Array(vc * 3), lc = new Float32Array(vc * 3), la = new Float32Array(vc);
        let o = 0;
        visLinks.forEach(l => {
            const a = l.s.rgb, b = l.t.rgb;
            for (let j = 0; j < SEG; j++) {
                [j / SEG, (j + 1) / SEG].forEach(u => {
                    lc[o++] = (a.r + (b.r - a.r) * u) * 0.85;
                    lc[o++] = (a.g + (b.g - a.g) * u) * 0.85;
                    lc[o++] = (a.b + (b.b - a.b) * u) * 0.85;
                });
            }
        });
        linkGeo.setAttribute('position', new THREE.BufferAttribute(lp, 3).setUsage(THREE.DynamicDrawUsage));
        linkGeo.setAttribute('aColor', new THREE.BufferAttribute(lc, 3));
        linkGeo.setAttribute('aAlpha', new THREE.BufferAttribute(la, 1).setUsage(THREE.DynamicDrawUsage));
        linkObj = new THREE.LineSegments(linkGeo, new THREE.ShaderMaterial({
            uniforms: U, vertexShader: LINE_VS, fragmentShader: LINE_FS,
            transparent: true, depthWrite: false, blending: THREE.AdditiveBlending
        }));
        linkObj.frustumCulled = false; linkObj.renderOrder = 2; linkObj.visible = visLinks.length > 0;
        linksGrp.add(linkObj);
        writeLinkPositions();

        const layer = $('tags-layer');
        layer.replaceChildren();
        vis.forEach(n => {
            const show = n.type === 'entity' || n.status === 'superseded' || (labelDensity === 'all' && (n.status === 'inhibitory' || (n.val || 0) >= 8));
            if (!show) return;
            const tag = document.createElement('div');
            tag.className = 'tag-3d' + (n.kind === 'user' ? ' user-core' : n.type === 'entity' ? ' entity' : n.status === 'superseded' ? ' trace' : ' satellite' + (n.status === 'inhibitory' ? ' inhibitory' : ''));
            tag.textContent = n.kind === 'user' ? 'You · core' : n.status === 'superseded' ? (n.value || n.label) : n.label;
            tag.style.opacity = '0';
            layer.appendChild(tag);
            n.tag = tag;
        });

        applyFocus();
        updateTelemetry();
    }

    /* ------------------------------------------------------------------ focus */
    function neighborhood(n) { return { primary: new Set([n.id]), secondary: new Set(adj.get(n.id) || []) }; }
    function computeFocus() {
        if (selectedNode) return neighborhood(selectedNode);
        if (hoveredNode) return neighborhood(hoveredNode);
        if (searchSet && searchSet.size) {
            const sec = new Set();
            searchSet.forEach(id => (adj.get(id) || new Set()).forEach(x => { if (!searchSet.has(x)) sec.add(x); }));
            return { primary: searchSet, secondary: sec };
        }
        return null;
    }
    function applyFocus() {
        focus = computeFocus();
        vis.forEach(n => {
            n.alphaT = focus
                ? (focus.primary.has(n.id) ? 1 : focus.secondary.has(n.id) ? 0.8 : 0.07)
                : (n.kind === 'superseded' ? 0.9 : 1);
        });
        visLinks.forEach(l => {
            l.alphaT = focus
                ? ((focus.primary.has(l.s.id) || focus.primary.has(l.t.id)) ? Math.max(0.6, l.base * 3) : 0.02)
                : l.base;
        });
    }

    /* ------------------------------------------------------------ frame work */
    const dirV = new V3(), perpV = new V3();
    let linksDirty = true;

    function writeLinkPositions() {
        if (!linkGeo || !visLinks.length) return;
        const P = linkGeo.attributes.position.array;
        let o = 0;
        for (const l of visLinks) {
            const s = l.s.cur, t = l.t.cur;
            dirV.subVectors(t, s);
            const len = dirV.length();
            perpV.crossVectors(dirV, UP);
            if (perpV.lengthSq() < 1e-6) perpV.set(1, 0, 0);
            perpV.normalize();
            l.c.addVectors(s, t).multiplyScalar(0.5).addScaledVector(perpV, len * l.bend * l.sign);
            l.c.y += len * 0.05;
            let px = s.x, py = s.y, pz = s.z;
            for (let j = 1; j <= SEG; j++) {
                const u = j / SEG, iu = 1 - u, w0 = iu * iu, w1 = 2 * iu * u, w2 = u * u;
                const x = w0 * s.x + w1 * l.c.x + w2 * t.x;
                const y = w0 * s.y + w1 * l.c.y + w2 * t.y;
                const z = w0 * s.z + w1 * l.c.z + w2 * t.z;
                P[o++] = px; P[o++] = py; P[o++] = pz; P[o++] = x; P[o++] = y; P[o++] = z;
                px = x; py = y; pz = z;
            }
        }
        linkGeo.attributes.position.needsUpdate = true;
    }

    function bezierAt(l, u, out) {
        const s = l.s.cur, t = l.t.cur, iu = 1 - u, w0 = iu * iu, w1 = 2 * iu * u, w2 = u * u;
        return out.set(w0 * s.x + w1 * l.c.x + w2 * t.x, w0 * s.y + w1 * l.c.y + w2 * t.y, w0 * s.z + w1 * l.c.z + w2 * t.z);
    }

    function updateNodes(dt) {
        if (!nodeGeo) return;
        const kp = 1 - Math.exp(-dt * 4), ka = 1 - Math.exp(-dt * 8);
        const P = nodeGeo.attributes.position.array, Sz = nodeGeo.attributes.aSize.array, Al = nodeGeo.attributes.aAlpha.array;
        let moving = false;
        for (let i = 0; i < vis.length; i++) {
            const n = vis[i];
            const since = nowT - n.born - n.delay;
            if (since > -0.25) {
                if (Math.abs(n.tgt.x - n.cur.x) + Math.abs(n.tgt.y - n.cur.y) + Math.abs(n.tgt.z - n.cur.z) > 1e-3) moving = true;
                n.cur.lerp(n.tgt, kp);
            }
            const b = clamp(since / 0.9, 0, 1);
            n.bf = b;
            if (b < 1) moving = true;
            n.alpha += (n.alphaT - n.alpha) * ka;
            P[i * 3] = n.cur.x; P[i * 3 + 1] = n.cur.y; P[i * 3 + 2] = n.cur.z;
            Sz[i] = n.size * (b < 1 ? Math.max(0, easeOutBack(b)) : 1);
            Al[i] = n.alpha * Math.min(1, b * 1.6);
        }
        nodeGeo.attributes.position.needsUpdate = true;
        nodeGeo.attributes.aSize.needsUpdate = true;
        nodeGeo.attributes.aAlpha.needsUpdate = true;
        if (moving) linksDirty = true;
    }

    function updateLinks(dt) {
        if (!linkGeo || !visLinks.length) return;
        const ka = 1 - Math.exp(-dt * 8);
        if (linksDirty) { writeLinkPositions(); linksDirty = false; }
        const A = linkGeo.attributes.aAlpha.array;
        let o = 0;
        for (const l of visLinks) {
            const d = l.alphaT - l.alpha;
            l.alpha = Math.abs(d) > 0.002 ? l.alpha + d * ka : l.alphaT;
            const a = l.alpha * Math.min(l.s.bf, l.t.bf);
            for (let k = 0; k < SEG * 2; k++) A[o++] = a;
        }
        linkGeo.attributes.aAlpha.needsUpdate = true;
    }

    const sleeping = () => nowT - sleepStart < 3.4;

    function pickPulseLink() {
        if (!visLinks.length) return 0;
        for (let t = 0; t < 6; t++) {
            const i = Math.floor(Math.random() * visLinks.length);
            if (visLinks[i].alphaT >= 0.1) return i;
        }
        return Math.floor(Math.random() * visLinks.length);
    }
    const pulseV = new V3();
    function updatePulses(dt) {
        const P = pulseGeo.attributes.position.array, Al = pulseGeo.attributes.aAlpha.array;
        const mul = sleeping() ? 3 : 1;
        for (let i = 0; i < PULSE_CAP; i++) {
            const p = pulses[i];
            if (!visLinks.length) { Al[i] = 0; continue; }
            p.prog += p.spd * dt * mul;
            if (p.prog >= 1) { p.prog = 0; p.idx = pickPulseLink(); }
            const l = visLinks[p.idx % visLinks.length];
            bezierAt(l, p.prog, pulseV);
            P[i * 3] = pulseV.x; P[i * 3 + 1] = pulseV.y; P[i * 3 + 2] = pulseV.z;
            Al[i] = Math.sin(Math.PI * p.prog) * clamp(l.alpha * 1.8, 0, 1) * Math.min(l.s.bf, l.t.bf);
        }
        pulseGeo.attributes.position.needsUpdate = true;
        pulseGeo.attributes.aAlpha.needsUpdate = true;
    }

    const flowV = new V3();
    function updateFlow(dt, glow) {
        const P = flowGeo.attributes.position.array, Al = flowGeo.attributes.aAlpha.array;
        const mul = sleeping() ? 3.5 : 1;
        const intro = clamp((U.uForm.value - 2.4) / 1.0, 0, 1);
        for (let i = 0; i < FLOW_CAP; i++) {
            const f = flows[i];
            f.u += f.spd * dt * mul;
            if (f.u > 1.6) { f.u = 0; f.ang = Math.random() * TAU; }
            if (f.u < 1) tubePoint(HC[f.s].hip, f.u, f.ang, hipR(f.u) * 0.55, flowV);
            else HC[f.s].fx.getPointAt(Math.min(1, (f.u - 1) / 0.6), flowV);
            P[i * 3] = flowV.x; P[i * 3 + 1] = flowV.y; P[i * 3 + 2] = flowV.z;
            Al[i] = Math.sin(Math.PI * f.u / 1.6) * intro * glow;
        }
        flowGeo.attributes.position.needsUpdate = true;
        flowGeo.attributes.aAlpha.needsUpdate = true;
    }

    const moteV = new V3();
    function updateMotes() {
        const P = moteGeo.attributes.position.array, Al = moteGeo.attributes.aAlpha.array;
        const intro = clamp((U.uForm.value - 2.2) / 1.0, 0, 1);
        const k = sleeping() ? easeInOut(clamp((nowT - sleepStart) / 2.6, 0, 1)) : 0;
        for (let i = 0; i < MOTE_CAP; i++) {
            if (i >= moteCount) { Al[i] = 0; continue; }
            const m = motes[i];
            moteV.set(
                m.home.x + Math.sin(nowT * 0.9 + m.ph) * 0.04,
                m.home.y + Math.cos(nowT * 1.1 + m.ph * 2) * 0.04,
                m.home.z + Math.sin(nowT * 0.7 + m.ph * 3) * 0.04
            ).lerp(m.target, k);
            P[i * 3] = moteV.x; P[i * 3 + 1] = moteV.y; P[i * 3 + 2] = moteV.z;
            Al[i] = (0.45 + 0.55 * Math.sin(nowT * 3 + m.ph * 5) ** 2) * intro * (1 - k);
        }
        moteGeo.attributes.position.needsUpdate = true;
        moteGeo.attributes.aAlpha.needsUpdate = true;
    }

    function updateSelectors() {
        const A = selGeo.attributes, P = A.position.array, Sz = A.aSize.array, Al = A.aAlpha.array;
        const h = hoveredNode && hoveredNode.vis && hoveredNode !== selectedNode ? hoveredNode : null;
        if (h) { P[0] = h.cur.x; P[1] = h.cur.y; P[2] = h.cur.z; Sz[0] = Math.max(h.size * 2.4, 0.2); Al[0] = 0.9; }
        else { Sz[0] = 0; Al[0] = 0; }
        const s = selectedNode && selectedNode.vis ? selectedNode : null;
        if (s) { P[3] = s.cur.x; P[4] = s.cur.y; P[5] = s.cur.z; Sz[1] = Math.max(s.size * 2.6, 0.24) * (1 + 0.06 * Math.sin(nowT * 3)); Al[1] = 1; }
        else { Sz[1] = 0; Al[1] = 0; }
        A.position.needsUpdate = true; A.aSize.needsUpdate = true; A.aAlpha.needsUpdate = true;
    }

    function updateTags() {
        const W = window.innerWidth, H = window.innerHeight, dpr = R.getPixelRatio();
        for (const n of vis) {
            const tag = n.tag;
            if (!tag) continue;
            tmpW.copy(n.cur).project(C);
            if (tmpW.z > 1) { tag.style.opacity = '0'; continue; }
            const dist = C.position.distanceTo(n.cur);
            const lift = (n.size * U.uScale.value / dist / dpr) * 0.5 + 9;
            const x = (tmpW.x * 0.5 + 0.5) * W, y = (-tmpW.y * 0.5 + 0.5) * H - lift;
            tag.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0) translate(-50%, -100%)`;
            const fade = clamp(1.15 - (dist - cam.radius) * 0.06, 0.25, 1);
            tag.style.opacity = (n.alpha * (n.bf || 0) * fade).toFixed(2);
        }
    }

    /* brain / limbic opacity + ripples */
    const lerpTo = (u, t, k) => { u.value += (t - u.value) * k; };
    function updateBrain(dt) {
        const k = 1 - Math.exp(-dt * 3);
        const hip = layoutMode === 'hippocampus', fo = !!focus;
        lerpTo(cortexMat.uniforms.uAlpha, !brainOn ? 0 : hip ? (fo ? 0.3 : 0.62) : 0.14, k);
        const glow = (hip ? (fo ? 0.45 : 0.85) : 0.28) * (1 + 0.12 * Math.sin(nowT * 1.4)) * (sleeping() ? 1.5 : 1);
        lerpTo(hippoMat.uniforms.uAlpha, glow, k);
        lerpTo(amygMat.uniforms.uAlpha, hip ? (fo ? 0.4 : 0.7) : 0.22, k);
        cortexGroup.visible = cortexMat.uniforms.uAlpha.value > 0.005;

        if (sleeping()) {
            const st = nowT - sleepStart;
            U.uRippleR.value = ((st * 1.35) % 1.45) * 3.4;
            U.uRippleA.value = clamp(1 - st / 3.4, 0, 1) * 1.2;
        } else if (!REDUCED && U.uForm.value > 2.6 && U.uForm.value < 4.0) {
            const st = U.uForm.value - 2.6;
            U.uRippleR.value = st * 2.6;
            U.uRippleA.value = 0.8 * (1 - st / 1.4);
        } else {
            U.uRippleA.value *= 1 - k;
        }
        return glow;
    }

    /* ---------------------------------------------------------------- picking */
    const pickV = new V3(), pickV2 = new V3();
    function pickAt(cx, cy) {
        const W = window.innerWidth, H = window.innerHeight, dpr = R.getPixelRatio();
        let best = null, bestScore = 1;
        for (const n of vis) {
            if ((n.bf || 0) < 0.5) continue;
            pickV.copy(n.cur).applyMatrix4(C.matrixWorldInverse);
            const depth = -pickV.z;
            if (depth < 0.1) continue;
            pickV2.copy(n.cur).project(C);
            const sx = (pickV2.x * 0.5 + 0.5) * W, sy = (-pickV2.y * 0.5 + 0.5) * H;
            const rad = Math.max(9, (n.size * U.uScale.value / depth / dpr) * 0.5 + 4);
            const score = Math.hypot(sx - cx, sy - cy) / rad;
            if (score < bestScore) { bestScore = score; best = n; }
        }
        return best;
    }

    /* ---------------------------------------------------------------- tooltip */
    const REGION_OF = { user: 'core', entity: 'hippocampus', inhibitory: 'amygdala', superseded: 'neocortex', active: 'hippocampus', event: 'hippocampus', conditional: 'hippocampus' };
    const SYSTEM_OF = { user: 'self', entity: 'semantic · concept cell', inhibitory: 'aversive · inhibitory', superseded: 'decayed trace', active: 'semantic · engram', event: 'episodic', conditional: 'context-gated' };
    const tooltip = $('node-tooltip');
    function showTooltip(n, x, y) {
        const head = n.status === 'inhibitory' ? `Forbidden · ${n.attribute || 'rule'}` : `${n.label}`;
        let sub = String(n.details?.citation || n.value || n.type || 'Connected memory node');
        if (sub.length > 70) sub = sub.slice(0, 68) + '…';
        tooltip.replaceChildren();
        const kd = document.createElement('div'); kd.className = 'kind';
        kd.textContent = `${SYSTEM_OF[n.kind]} · ${layoutMode === 'hippocampus' ? REGION_OF[n.kind] : 'constellation'}`;
        const b = document.createElement('b'); b.textContent = head;
        const d = document.createElement('div'); d.className = 'sub'; d.textContent = sub;
        tooltip.append(kd, b, d);
        moveTooltip(x, y);
        tooltip.style.display = 'block';
    }
    function moveTooltip(x, y) {
        tooltip.style.left = Math.min(x, window.innerWidth - 300) + 'px';
        tooltip.style.top = Math.min(y, window.innerHeight - 110) + 'px';
    }
    function hideTooltip() { tooltip.style.display = 'none'; }

    /* -------------------------------------------------------------- inspector */
    const inspector = $('inspector');
    function renderInspector(n) {
        const kind = n.kind;
        inspector.dataset.kind = kind;
        const b = $('insp-badge');
        b.className = 'badge ' + (n.status === 'inhibitory' ? 'inhibitory' : n.type === 'entity' ? 'entity' : n.type === 'event' ? 'event' : n.status === 'superseded' ? 'superseded' : 'active');
        b.textContent = kind === 'user' ? 'You · core' : n.status === 'inhibitory' ? 'Forbidden rule' : n.type === 'entity' ? 'Entity'
            : n.type === 'event' ? 'Episodic event' : n.status === 'superseded' ? 'Superseded' : n.status === 'conditional' ? 'Conditional' : 'Active truth';
        $('insp-region').textContent = `${SYSTEM_OF[kind] || ''} — ${REGION_OF[kind] || ''}`;

        $('insp-title').textContent = n.label;
        $('insp-sub').textContent = `${n.subject || n.label} · ${n.attribute || n.type}`;
        $('insp-val').textContent = n.value || n.label;
        $('insp-cite').textContent = `“${n.details?.citation || n.details?.sentence_text || 'Committed during sleep consolidation.'}”`;
        const vf = n.details?.valid_from ? new Date(n.details.valid_from * 1000).toISOString().split('T')[0] : 'continuous';
        $('insp-meta').textContent = `valid ${vf} · weight ${n.val || 8}`;

        const box = $('insp-links');
        box.replaceChildren();
        let count = 0;
        allLinks.forEach(l => {
            const s = linkId(l.source), t = linkId(l.target);
            let other = null, dir = '';
            if (s === n.id && nodeMap.get(t)) { other = nodeMap.get(t); dir = '→'; }
            else if (t === n.id && nodeMap.get(s)) { other = nodeMap.get(s); dir = '←'; }
            if (!other) return;
            count++;
            const row = document.createElement('button');
            row.className = 'syn'; row.type = 'button';
            row.style.setProperty('--k', '#' + other.rgb.getHexString());
            const d = document.createElement('i'); d.textContent = dir;
            const k = document.createElement('em'); k.textContent = other.type;
            const t2 = document.createElement('span'); t2.textContent = other.label;
            row.append(d, k, t2);
            row.addEventListener('click', () => { if (other.vis) selectNode(other, true); else toast('Hidden by current filter'); });
            box.appendChild(row);
        });
        if (!count) { const e = document.createElement('div'); e.className = 'empty'; e.textContent = 'No external synapses.'; box.appendChild(e); }
        inspector.classList.add('open');
    }
    function selectNode(n, fly) {
        selectedNode = n; hoveredNode = null; hideTooltip();
        renderInspector(n); applyFocus();
        if (fly) flyTo(n);
    }
    function closeInspector() { selectedNode = null; inspector.classList.remove('open'); applyFocus(); }
    $('x-insp').onclick = closeInspector;

    /* -------------------------------------------------------------- telemetry */
    function updateTelemetry() {
        const c = {
            all: allNodes.length,
            entity: allNodes.filter(n => n.type === 'entity').length,
            active: allNodes.filter(n => n.type === 'state' && n.status === 'active').length,
            inhibitory: allNodes.filter(n => n.status === 'inhibitory').length,
            event: allNodes.filter(n => n.type === 'event').length,
            superseded: allNodes.filter(n => n.status === 'superseded').length
        };
        Object.entries(c).forEach(([k, v]) => { const el = $(`c-${k}`); if (el) el.textContent = v; });
        $('t-nodes').textContent = c.all;
        $('t-links').textContent = allLinks.length;
        $('t-inhibitory').textContent = c.inhibitory;
        $('t-buffer').textContent = (rawData?.hippocampus || []).length;
    }

    function renderBuffer() {
        const list = $('buffer-list');
        list.replaceChildren();
        const items = rawData?.hippocampus || [];
        $('buffer-count').textContent = items.length;
        seedMotes(items.length);
        if (!items.length) {
            const li = document.createElement('li'); li.className = 'empty';
            li.textContent = 'Buffer is empty — everything is consolidated.';
            list.appendChild(li); return;
        }
        items.forEach(it => {
            const li = document.createElement('li');
            const role = document.createElement('span'); role.className = 'role';
            role.textContent = typeof it === 'string' ? 'raw' : (it.role || 'raw');
            const p = document.createElement('p'); p.textContent = typeof it === 'string' ? it : (it.text || JSON.stringify(it));
            li.append(role, p); list.appendChild(li);
        });
    }

    function setSource(mode) {
        $('src').dataset.mode = mode;
        $('src-mode').textContent = mode === 'live' ? 'LIVE' : 'DEMO';
        $('sync-time').textContent = new Date().toTimeString().slice(0, 5);
    }

    /* ------------------------------------------------------------------- data */
    function demoData() {
        return {
            nodes: [
                { id: 'entity_1', label: 'user', type: 'entity', val: 20 },
                { id: 'entity_2', label: 'Neha', type: 'entity', val: 14 },
                { id: 'entity_3', label: 'Vikram', type: 'entity', val: 14 },
                { id: 'state_1', label: 'user.role: systems engineer', type: 'state', status: 'active', subject: 'user', attribute: 'role', value: 'systems engineer', val: 8 },
                { id: 'state_2', label: 'user.company: Zerodha', type: 'state', status: 'active', subject: 'user', attribute: 'company', value: 'Zerodha', val: 8 },
                { id: 'state_3', label: 'user.diet: warm ginger tea with raw honey', type: 'state', status: 'active', subject: 'user', attribute: 'diet', value: 'warm ginger tea with raw honey', val: 8 },
                { id: 'state_4', label: 'Neha.occupation: data scientist', type: 'state', status: 'active', subject: 'Neha', attribute: 'occupation', value: 'data scientist', val: 8 },
                { id: 'state_5', label: 'Vikram.occupation: graphic designer', type: 'state', status: 'active', subject: 'Vikram', attribute: 'occupation', value: 'graphic designer', val: 8 },
                { id: 'state_6', label: 'Vikram.location: Mumbai', type: 'state', status: 'active', subject: 'Vikram', attribute: 'location', value: 'Mumbai', val: 8 },
                { id: 'state_7', label: 'Vikram.tool: Figma', type: 'state', status: 'active', subject: 'Vikram', attribute: 'tool', value: 'Figma', val: 8 },
                { id: 'state_8', label: 'user.forbidden_peanuts: Never suggest peanuts (allergy)', type: 'state', status: 'inhibitory', subject: 'user', attribute: 'forbidden_peanuts', value: 'Never suggest peanuts (allergy)', val: 10 },
                { id: 'state_9', label: 'user.mood: prefers tea when stressed', type: 'state', status: 'conditional', subject: 'user', attribute: 'mood', value: 'prefers tea when stressed', val: 6 },
                { id: 'state_10', label: 'user.location: Pune', type: 'state', status: 'superseded', subject: 'user', attribute: 'location', value: 'Pune', val: 5 },
                { id: 'event_1', label: 'Trek to Sinhagad Fort', type: 'event', status: 'active', val: 6, details: { citation: 'Went trekking to Sinhagad Fort with friends' } }
            ],
            links: [
                { source: 'entity_1', target: 'state_1' }, { source: 'entity_1', target: 'state_2' }, { source: 'entity_1', target: 'state_3' },
                { source: 'entity_2', target: 'state_4' }, { source: 'entity_3', target: 'state_5' }, { source: 'entity_3', target: 'state_6' },
                { source: 'entity_3', target: 'state_7' }, { source: 'entity_1', target: 'state_8' }, { source: 'entity_1', target: 'event_1' },
                { source: 'entity_1', target: 'state_9' }, { source: 'entity_1', target: 'state_10' }, { source: 'entity_1', target: 'entity_2' }
            ],
            hippocampus: [{ role: 'user', text: 'I work as a systems engineer at Zerodha.' }]
        };
    }

    async function syncConnectome() {
        let data = null, live = false;
        try {
            const res = await fetch(API.brain, { cache: 'no-store' });
            if (!res.ok) throw 0;
            data = await res.json();
            live = true;
            if (!firstLoad) toast('Connectome synchronized');
        } catch (e) {
            if (!rawData) { data = demoData(); if (!firstLoad) toast('Simulation stream active'); }
            else { toast('Sync failed — showing last state'); return; }
        }
        ingest(data, live);
    }

    function ingest(data, live) {
        const old = new Map();
        allNodes.forEach(n => { if (n.cur) old.set(n.id, { cur: n.cur.clone(), born: n.born, delay: n.delay }); });

        rawData = data;
        allNodes = Array.isArray(data.nodes) ? data.nodes : [];
        allLinks = Array.isArray(data.links) ? data.links : [];
        nodeMap.clear(); adj.clear();
        allNodes.forEach(n => { nodeMap.set(n.id, n); adj.set(n.id, new Set()); });
        allLinks.forEach(l => {
            const s = linkId(l.source), t = linkId(l.target);
            if (adj.has(s) && adj.has(t)) { adj.get(s).add(t); adj.get(t).add(s); }
        });

        allNodes.forEach(n => {
            n.kind = kindOf(n);
            const k = KIND[n.kind];
            n.rgb = new THREE.Color(k.color);
            n.shape = k.shape;
            n.size = k.size * (0.92 + clamp(n.val || 8, 0, 24) / 100);
            n.hash = hashStr(String(n.id));
            n.alpha = 1; n.alphaT = 1; n.bf = 0;
            n.side = undefined; n.hippo = null; n.planet = null; n.parent = null;
        });
        allNodes.forEach(n => {
            if (n.type === 'entity') return;
            for (const id of (adj.get(n.id) || [])) {
                const o = nodeMap.get(id);
                if (o && o.type === 'entity') { n.parent = o; break; }
            }
        });
        hippocampusLayout();
        constellationLayout();

        const wasFirst = firstLoad;
        allNodes.forEach(n => {
            n.tgt = targetOf(n).clone();
            const o = old.get(n.id);
            if (o) { n.cur = o.cur; n.born = o.born; n.delay = o.delay; return; }
            let start = null;
            for (const id of (adj.get(n.id) || [])) { const oo = old.get(id); if (oo) { start = oo.cur.clone(); break; } }
            if (!start) {
                const s = n.side || (n.hash > 0.5 ? 1 : -1);
                start = layoutMode === 'hippocampus' && n.kind !== 'user' ? HC[s].hip.getPointAt(0.03).clone() : new V3(0, -0.08, 0.05);
            }
            n.cur = start;
            n.born = wasFirst ? Math.max(nowT, INTRO_NODES) : nowT;
            n.delay = n.hash * (wasFirst ? 0.9 : 0.5);
        });

        if (selectedNode) {
            selectedNode = nodeMap.get(selectedNode.id) || null;
            if (!selectedNode) inspector.classList.remove('open');
        }
        hoveredNode = null;
        setSource(live ? 'live' : 'demo');
        renderBuffer();
        refreshGraph();
        if (selectedNode) renderInspector(selectedNode);
        if (qSearch.value.trim()) runSearch(false);
        if (wasFirst) { fitCamera(); firstLoad = false; }
    }

    /* ----------------------------------------------------------------- inputs */
    const qSearch = $('q-search');
    const pointer = { x: -1, y: -1, moved: false, inside: false };
    const pts = new Map();
    let downInfo = null, lastPinch = 0;
    const pinchDist = () => { const a = [...pts.values()]; return a.length < 2 ? 0 : Math.hypot(a[0].x - a[1].x, a[0].y - a[1].y); };

    glCv.addEventListener('pointerdown', e => {
        try { glCv.setPointerCapture(e.pointerId); } catch (_) { }
        pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
        if (pts.size === 1) downInfo = { x: e.clientX, y: e.clientY, t: performance.now(), moved: false };
        if (pts.size === 2) { lastPinch = pinchDist(); if (downInfo) downInfo.moved = true; }
        dragging = true; touch();
        if (hoveredNode) { hoveredNode = null; hideTooltip(); applyFocus(); }
    });
    glCv.addEventListener('pointermove', e => {
        const p = pts.get(e.pointerId);
        if (p) {
            const dx = e.clientX - p.x, dy = e.clientY - p.y;
            p.x = e.clientX; p.y = e.clientY;
            if (pts.size === 1) {
                cam.tTheta -= dx * 0.0055;
                cam.tPhi = clamp(cam.tPhi - dy * 0.0055, 0.15, Math.PI - 0.15);
                if (downInfo && Math.hypot(e.clientX - downInfo.x, e.clientY - downInfo.y) > 5) downInfo.moved = true;
            } else if (pts.size === 2) {
                const d = pinchDist();
                if (lastPinch && d) cam.tRadius = clamp(cam.tRadius * lastPinch / d, 1.8, 26);
                lastPinch = d;
            }
            touch();
        } else {
            pointer.x = e.clientX; pointer.y = e.clientY; pointer.moved = true; pointer.inside = true;
            if (tooltip.style.display === 'block') moveTooltip(e.clientX, e.clientY);
        }
    });
    function endPtr(e) {
        if (pts.size === 1 && downInfo && !downInfo.moved && performance.now() - downInfo.t < 500 && e.type === 'pointerup') {
            const n = pickAt(e.clientX, e.clientY);
            if (n) selectNode(n, false); else if (selectedNode) closeInspector();
        }
        pts.delete(e.pointerId);
        if (!pts.size) { dragging = false; downInfo = null; }
        lastPinch = 0;
    }
    glCv.addEventListener('pointerup', endPtr);
    glCv.addEventListener('pointercancel', endPtr);
    glCv.addEventListener('pointerleave', () => {
        pointer.inside = false;
        if (hoveredNode) { hoveredNode = null; hideTooltip(); applyFocus(); }
    });
    glCv.addEventListener('dblclick', e => { const n = pickAt(e.clientX, e.clientY); if (n) selectNode(n, true); });
    glCv.addEventListener('wheel', e => {
        e.preventDefault();
        cam.tRadius = clamp(cam.tRadius * (1 + e.deltaY * 0.0012), 1.8, 26);
        touch();
    }, { passive: false });

    function hoverPick() {
        if (!pointer.moved || dragging) return;
        pointer.moved = false;
        const n = pointer.inside ? pickAt(pointer.x, pointer.y) : null;
        if (n !== hoveredNode) {
            hoveredNode = n;
            if (n) { showTooltip(n, pointer.x, pointer.y); glCv.style.cursor = 'pointer'; }
            else { hideTooltip(); glCv.style.cursor = ''; }
            if (!selectedNode) applyFocus();
        }
    }

    /* filters */
    $('filters').addEventListener('click', e => {
        const btn = e.target.closest('.pill'); if (!btn) return;
        document.querySelectorAll('.pill').forEach(b => b.classList.remove('active'));
        btn.classList.add('active'); activeFilter = btn.dataset.f;
        refreshGraph();
        if (selectedNode && !selectedNode.vis) closeInspector();
        if (qSearch.value.trim()) runSearch(false);
        toast(`Filter: ${btn.querySelector('.lbl')?.firstChild?.textContent.trim() || activeFilter}`);
    });

    /* regions */
    const REGIONS = {
        hippocampus: { at: () => HC[1].hip.getPointAt(0.45), r: 4.2, note: 'Hippocampus — where new memories are encoded and live' },
        amygdala: { at: () => HC[1].amyg, r: 3.4, note: 'Amygdala — aversive memory; forbidden rules sit here' },
        dentate: { at: () => HC[-1].hip.getPointAt(0.12), r: 3.6, note: 'Dentate gyrus — buffered input waiting for sleep' },
        fornix: { at: () => HC[1].fx.getPointAt(0.45), r: 4.2, note: 'Fornix — the output arch; replay travels along it' },
        cortex: { at: () => new V3(0, 0.3, -0.4), r: 8.6, note: 'Neocortex — superseded memories fade into long-term storage' }
    };
    document.querySelectorAll('[data-region]').forEach(el => el.addEventListener('click', () => {
        const reg = REGIONS[el.dataset.region]; if (!reg) return;
        if (layoutMode !== 'hippocampus') setLayout('hippocampus', true);
        if (selectedNode) closeInspector();
        cam.tTarget.copy(reg.at()); cam.tRadius = reg.r; touch();
        toast(reg.note);
    }));

    /* search */
    function runSearch(fly) {
        const q = qSearch.value.trim().toLowerCase();
        const cnt = $('search-count');
        if (!q) { searchSet = null; searchList = []; searchCursor = -1; cnt.textContent = ''; applyFocus(); return; }
        searchList = vis.filter(n => [n.label, n.value, n.attribute, n.subject].some(f => f != null && String(f).toLowerCase().includes(q)));
        searchSet = new Set(searchList.map(n => n.id));
        cnt.textContent = searchList.length ? `${searchList.length} hit${searchList.length > 1 ? 's' : ''}` : 'no match';
        if (searchList.length) { searchCursor = 0; if (fly) flyTo(searchList[0]); }
        applyFocus();
    }
    qSearch.addEventListener('input', () => { if (selectedNode) closeInspector(); runSearch(true); });
    qSearch.addEventListener('keydown', e => {
        if (e.key === 'Enter' && searchList.length) {
            searchCursor = (searchCursor + 1) % searchList.length;
            flyTo(searchList[searchCursor]);
        }
    });

    /* toolbar */
    const btnLayout = $('btn-layout');
    function setLayout(mode, quiet) {
        layoutMode = mode;
        btnLayout.querySelector('span').textContent = mode === 'hippocampus' ? 'HIPPOCAMPUS' : 'CONSTELLATION';
        btnLayout.classList.toggle('active', mode === 'hippocampus');
        allNodes.forEach(n => { n.tgt = targetOf(n).clone(); });
        cam.tTarget.set(0, -0.15, 0); fitCamera(); touch();
        if (!quiet) toast(mode === 'hippocampus' ? 'Anatomical view: memories in the hippocampus' : 'Constellation view: memories spread out');
    }
    btnLayout.onclick = () => setLayout(layoutMode === 'hippocampus' ? 'constellation' : 'hippocampus');

    const btnCortex = $('btn-cortex');
    btnCortex.onclick = () => {
        brainOn = !brainOn;
        btnCortex.classList.toggle('active', brainOn);
        btnCortex.querySelector('span').textContent = brainOn ? 'CORTEX ON' : 'CORTEX OFF';
        toast(brainOn ? 'Cortex visible' : 'Cortex hidden — limbic system only');
    };
    const btnLabels = $('btn-labels');
    btnLabels.onclick = () => {
        labelDensity = labelDensity === 'clean' ? 'all' : 'clean';
        btnLabels.classList.toggle('active', labelDensity === 'all');
        btnLabels.querySelector('span').textContent = labelDensity === 'clean' ? 'LABELS CLEAN' : 'LABELS ALL';
        refreshGraph();
        toast(labelDensity === 'clean' ? 'Entity hubs only' : 'All labels visible');
    };
    $('btn-reset').onclick = resetView;

    $('btn-sleep').onclick = async () => {
        toast('Sleep replay — sharp-wave ripples firing…');
        sleepStart = nowT;
        document.body.classList.add('sleeping');
        try {
            const res = await fetch(API.sleep, { method: 'POST' });
            if (res.ok) { setTimeout(() => { toast('Consolidated into cortex'); syncConnectome(); }, 2600); } else throw 0;
        } catch (e) { setTimeout(() => toast('Simulation: stream consolidated'), 2600); }
    };

    /* buffer drawer */
    const bufferEl = $('buffer');
    function toggleBuffer(force) {
        const open = typeof force === 'boolean' ? force : !bufferEl.classList.contains('open');
        bufferEl.classList.toggle('open', open);
        $('btn-buffer').classList.toggle('on', open);
    }
    $('btn-buffer').onclick = () => toggleBuffer();
    $('x-buffer').onclick = () => toggleBuffer(false);

    /* encode */
    async function encodeText() {
        const inp = $('q-tx'), txt = inp.value.trim(); if (!txt) return;
        inp.value = '';
        toast(`Encoding: "${txt.substring(0, 22)}${txt.length > 22 ? '…' : ''}"`);
        try {
            await fetch(API.encode, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: txt }) });
            syncConnectome();
        } catch (e) { toast('Queued in memory stream'); }
    }
    $('btn-tx').onclick = encodeText;
    $('q-tx').addEventListener('keydown', e => { if (e.key === 'Enter') encodeText(); });

    /* keyboard */
    window.addEventListener('keydown', e => {
        const typing = /^(INPUT|TEXTAREA)$/.test(document.activeElement?.tagName || '');
        if (e.key === 'Escape') {
            closeInspector(); qSearch.value = ''; runSearch(false); hideTooltip(); toggleBuffer(false);
            if (typing) document.activeElement.blur();
            return;
        }
        if (typing) return;
        if (e.key === '/') { e.preventDefault(); qSearch.focus(); qSearch.select(); }
        else if (['1', '2', '3', '4', '5', '6'].includes(e.key)) { const p = document.querySelectorAll('.pill')[parseInt(e.key, 10) - 1]; if (p) p.click(); }
        else if (e.key === 'b') toggleBuffer();
        else if (e.key === 'r') resetView();
        else if (e.key === 'l') btnLayout.click();
    });

    /* toast */
    let toastTimer = 0;
    function toast(msg) {
        const t = $('toast'); t.textContent = msg; t.classList.add('show');
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => t.classList.remove('show'), 2600);
    }

    /* intro caption */
    const introEl = $('intro');
    const STAGES = [[0, 'condensing cortex'], [1.8, 'kindling the hippocampus'], [2.8, 'recalling memories']];
    let introDone = REDUCED;
    if (REDUCED && introEl) introEl.classList.add('done');
    function updateIntro() {
        if (introDone || !introEl) return;
        let txt = '';
        for (const [t, s] of STAGES) if (nowT >= t) txt = s;
        if (introEl.dataset.t !== txt) { introEl.dataset.t = txt; introEl.textContent = txt; }
        if (nowT > 4.6) { introEl.classList.add('done'); introDone = true; }
    }

    /* ------------------------------------------------------------------- loop */
    let lastTs = performance.now();
    function frame(ts) {
        requestAnimationFrame(frame);
        const dt = Math.min(0.05, Math.max(0, (ts - lastTs) / 1000)); lastTs = ts;
        nowT += dt;
        U.uTime.value = nowT;
        if (!REDUCED) U.uForm.value = nowT;
        if (!sleeping() && document.body.classList.contains('sleeping')) document.body.classList.remove('sleeping');

        updateIntro();
        updateCamera(dt);
        dustGroup.rotation.y += dt * 0.004;
        const glow = updateBrain(dt);
        updateNodes(dt);
        updateLinks(dt);
        updatePulses(dt);
        updateFlow(dt, glow);
        updateMotes();
        updateSelectors();
        hoverPick();
        updateTags();
        R.render(S, C);
    }

    window.addEventListener('resize', () => {
        C.aspect = window.innerWidth / window.innerHeight;
        C.updateProjectionMatrix();
        R.setSize(window.innerWidth, window.innerHeight, false);
        updateScale();
        fitCamera();
    });

    fitCamera();
    syncConnectome();
    requestAnimationFrame(frame);
})();