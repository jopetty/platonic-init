import { useState, useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

// --- Math helpers (mirrors the Python exactly) ---
function buildChebyshevBasis(n, degree) {
  const x = Array.from({ length: n }, (_, i) => i / (n - 1)); // [0, 1]
  const xCheb = x.map(v => 2.0 * v - 1.0);                   // [-1, 1]

  const polys = [];
  const t0 = xCheb.map(() => 1.0);
  polys.push(t0);
  if (degree >= 1) {
    const t1 = xCheb.slice();
    polys.push(t1);
    let prev2 = t0, prev1 = t1;
    for (let d = 2; d <= degree; d++) {
      const tNext = xCheb.map((xi, i) => 2.0 * xi * prev1[i] - prev2[i]);
      polys.push(tNext);
      prev2 = prev1;
      prev1 = tNext;
    }
  }
  return { x, xCheb, polys };
}

// Least-squares solve: basis (n×k) @ coeffs = y (n)
function lstsq(basis, y) {
  // Normal equations: (B^T B) coeffs = B^T y
  const k = basis[0].length;
  const n = basis.length;
  // BtB: k×k
  const BtB = Array.from({ length: k }, () => new Array(k).fill(0));
  const Bty = new Array(k).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < k; j++) {
      Bty[j] += basis[i][j] * y[i];
      for (let l = 0; l < k; l++) {
        BtB[j][l] += basis[i][j] * basis[i][l];
      }
    }
  }
  // Simple Cholesky / Gaussian elimination
  return gaussianElimination(BtB, Bty);
}

function gaussianElimination(A, b) {
  const n = b.length;
  const aug = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++)
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];
    const pivot = aug[col][col];
    if (Math.abs(pivot) < 1e-14) continue;
    for (let row = col + 1; row < n; row++) {
      const factor = aug[row][col] / pivot;
      for (let k = col; k <= n; k++) aug[row][k] -= factor * aug[col][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    x[row] = aug[row][n];
    for (let col = row + 1; col < n; col++) x[row] -= aug[row][col] * x[col];
    x[row] /= aug[row][row];
  }
  return x;
}

const COLORS = [
  "#6366f1","#f59e0b","#10b981","#ef4444","#3b82f6",
  "#ec4899","#8b5cf6","#14b8a6","#f97316","#84cc16",
  "#06b6d4","#a855f7","#e11d48"
];

const N = 200;

// Demo signal options
const SIGNALS = {
  "Smooth bump": x => Math.exp(-10 * (x - 0.5) ** 2),
  "Sine wave": x => Math.sin(2 * Math.PI * x),
  "Step function": x => x < 0.5 ? -1 : 1,
  "Polynomial": x => 3 * x ** 3 - 2 * x ** 2 + x - 0.5,
  "Two bumps": x => Math.exp(-20 * (x - 0.3) ** 2) - 0.8 * Math.exp(-20 * (x - 0.7) ** 2),
};

export default function App() {
  const [activeTab, setActiveTab] = useState("polynomials");
  const [degree, setDegree] = useState(4);
  const [selectedPolys, setSelectedPolys] = useState([0, 1, 2, 3, 4]);
  const [fitDegree, setFitDegree] = useState(6);
  const [signalName, setSignalName] = useState("Smooth bump");

  const { x, polys } = useMemo(() => buildChebyshevBasis(N, 12), []);

  // --- Polynomial tab data ---
  const polyData = useMemo(() => {
    const { polys: p } = buildChebyshevBasis(N, degree);
    return x.map((xi, i) => {
      const pt = { x: xi };
      p.forEach((poly, d) => { pt[`T${d}`] = poly[i]; });
      return pt;
    });
  }, [x, degree]);

  // --- Fit tab data ---
  const fitData = useMemo(() => {
    const signal = SIGNALS[signalName];
    const y = x.map(signal);
    const { polys: p } = buildChebyshevBasis(N, fitDegree);
    // Build basis matrix: n × (fitDegree+1)
    const basisMatrix = x.map((_, i) => p.map(poly => poly[i]));
    const coeffs = lstsq(basisMatrix, y);
    const reconstruction = x.map((_, i) => {
      let val = 0;
      for (let d = 0; d <= fitDegree; d++) val += coeffs[d] * p[d][i];
      return val;
    });
    const residuals = y.map((yi, i) => yi - reconstruction[i]);
    const relError = Math.sqrt(residuals.reduce((s, r) => s + r * r, 0)) /
      Math.sqrt(y.reduce((s, yi) => s + yi * yi, 0) + 1e-12);

    return {
      chartData: x.map((xi, i) => ({
        x: xi,
        signal: y[i],
        reconstruction: reconstruction[i],
        error: residuals[i],
      })),
      coeffs,
      relError,
    };
  }, [x, fitDegree, signalName]);

  const togglePoly = (d) => {
    setSelectedPolys(prev =>
      prev.includes(d) ? prev.filter(p => p !== d) : [...prev, d].sort((a, b) => a - b)
    );
  };

  return (
    <div style={{ fontFamily: "'Inter', system-ui, sans-serif", background: "#0f172a", minHeight: "100vh", color: "#e2e8f0", padding: "2rem" }}>
      <div style={{ maxWidth: 900, margin: "0 auto" }}>

        {/* Header */}
        <h1 style={{ fontSize: "1.8rem", fontWeight: 700, marginBottom: "0.25rem", color: "#f8fafc" }}>
          Chebyshev Approximation
        </h1>
        <p style={{ color: "#94a3b8", marginBottom: "2rem", lineHeight: 1.6 }}>
          How <code style={{ background: "#1e293b", padding: "1px 6px", borderRadius: 4, color: "#7dd3fc" }}>build_basis_numpy</code> and{" "}
          <code style={{ background: "#1e293b", padding: "1px 6px", borderRadius: 4, color: "#7dd3fc" }}>fit_vector</code> work in <em>platonic-init</em>
        </p>

        {/* Tabs */}
        <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1.5rem" }}>
          {["polynomials", "fitting", "howItWorks"].map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)} style={{
              padding: "0.5rem 1.25rem", borderRadius: 8, border: "none", cursor: "pointer", fontWeight: 600,
              background: activeTab === tab ? "#6366f1" : "#1e293b",
              color: activeTab === tab ? "#fff" : "#94a3b8",
              transition: "all 0.15s",
            }}>
              {tab === "polynomials" ? "The Basis Functions" : tab === "fitting" ? "Fitting a Signal" : "How It Works"}
            </button>
          ))}
        </div>

        {/* ───── TAB 1: Polynomials ───── */}
        {activeTab === "polynomials" && (
          <div>
            <div style={{ background: "#1e293b", borderRadius: 12, padding: "1.25rem", marginBottom: "1.5rem" }}>
              <p style={{ margin: "0 0 1rem 0", color: "#cbd5e1", lineHeight: 1.7 }}>
                Chebyshev polynomials <strong style={{ color: "#f8fafc" }}>T₀, T₁, T₂, …</strong> are a family of orthogonal functions
                defined on [−1, 1]. Your code maps inputs from [0,1] to [−1,1] first:{" "}
                <code style={{ background: "#0f172a", padding: "2px 8px", borderRadius: 4, color: "#86efac" }}>x_cheb = 2.0 * x − 1.0</code>.
                Each degree adds one more "wiggle" to the basis.
              </p>
              <div style={{ background: "#0f172a", borderRadius: 8, padding: "0.75rem 1rem", fontFamily: "monospace", fontSize: "0.85rem", color: "#fcd34d" }}>
                T₀(x) = 1<br />
                T₁(x) = x<br />
                Tₙ(x) = 2x·Tₙ₋₁(x) − Tₙ₋₂(x)   ← recurrence used in the code
              </div>
            </div>

            {/* Degree slider */}
            <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "1rem" }}>
              <label style={{ color: "#94a3b8", minWidth: 120 }}>Max degree: <strong style={{ color: "#f8fafc" }}>{degree}</strong></label>
              <input type="range" min={1} max={12} value={degree}
                onChange={e => { const d = +e.target.value; setDegree(d); setSelectedPolys(Array.from({ length: d + 1 }, (_, i) => i)); }}
                style={{ flex: 1, accentColor: "#6366f1" }} />
            </div>

            {/* Toggle buttons */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem", marginBottom: "1rem" }}>
              {Array.from({ length: degree + 1 }, (_, d) => (
                <button key={d} onClick={() => togglePoly(d)} style={{
                  padding: "0.25rem 0.75rem", borderRadius: 6, border: `2px solid ${COLORS[d]}`,
                  background: selectedPolys.includes(d) ? COLORS[d] + "33" : "transparent",
                  color: COLORS[d], cursor: "pointer", fontWeight: 600, fontSize: "0.85rem",
                }}>
                  T{d}
                </button>
              ))}
            </div>

            <ResponsiveContainer width="100%" height={380}>
              <LineChart data={polyData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="x" tickFormatter={v => v.toFixed(1)} stroke="#475569" label={{ value: "x (input, [0,1])", position: "insideBottom", offset: -2, fill: "#64748b", fontSize: 12 }} />
                <YAxis domain={[-1.2, 1.2]} stroke="#475569" />
                <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8 }}
                  formatter={(v, n) => [v.toFixed(4), n]} labelFormatter={v => `x = ${(+v).toFixed(3)}`} />
                {selectedPolys.map(d => (
                  <Line key={d} type="monotone" dataKey={`T${d}`} stroke={COLORS[d]} dot={false} strokeWidth={2} name={`T${d}`} />
                ))}
              </LineChart>
            </ResponsiveContainer>

            <p style={{ color: "#64748b", fontSize: "0.85rem", marginTop: "0.5rem", textAlign: "center" }}>
              Notice each Tₙ has exactly n roots — T₀ is flat, T₁ is linear, T₂ is a parabola, and so on.
            </p>
          </div>
        )}

        {/* ───── TAB 2: Fitting ───── */}
        {activeTab === "fitting" && (
          <div>
            <div style={{ background: "#1e293b", borderRadius: 12, padding: "1.25rem", marginBottom: "1.5rem" }}>
              <p style={{ margin: 0, color: "#cbd5e1", lineHeight: 1.7 }}>
                <code style={{ background: "#0f172a", padding: "2px 8px", borderRadius: 4, color: "#86efac" }}>fit_vector</code> takes a PCA component (a vector of values),
                builds the Chebyshev basis matrix, then calls{" "}
                <code style={{ background: "#0f172a", padding: "2px 8px", borderRadius: 4, color: "#86efac" }}>np.linalg.lstsq</code> to find the best-fit coefficients.
                Try increasing the degree to see the reconstruction improve.
              </p>
            </div>

            <div style={{ display: "flex", gap: "1.5rem", flexWrap: "wrap", marginBottom: "1.25rem" }}>
              <div style={{ flex: 1, minWidth: 200 }}>
                <label style={{ color: "#94a3b8", display: "block", marginBottom: "0.4rem" }}>Signal to approximate</label>
                <select value={signalName} onChange={e => setSignalName(e.target.value)} style={{
                  background: "#0f172a", border: "1px solid #334155", color: "#f8fafc",
                  borderRadius: 6, padding: "0.4rem 0.75rem", width: "100%", fontSize: "0.9rem"
                }}>
                  {Object.keys(SIGNALS).map(s => <option key={s}>{s}</option>)}
                </select>
              </div>
              <div style={{ flex: 1, minWidth: 200 }}>
                <label style={{ color: "#94a3b8", display: "block", marginBottom: "0.4rem" }}>
                  chebyshev_degree: <strong style={{ color: "#f8fafc" }}>{fitDegree}</strong>{" "}
                  <span style={{ color: "#64748b", fontWeight: 400 }}>({fitDegree + 1} basis functions)</span>
                </label>
                <input type="range" min={0} max={12} value={fitDegree} onChange={e => setFitDegree(+e.target.value)}
                  style={{ width: "100%", accentColor: "#6366f1" }} />
              </div>
            </div>

            {/* Error badge */}
            <div style={{ display: "inline-flex", alignItems: "center", gap: "0.5rem", background: "#1e293b", borderRadius: 8, padding: "0.4rem 1rem", marginBottom: "1rem" }}>
              <span style={{ color: "#94a3b8" }}>Relative reconstruction error:</span>
              <span style={{
                fontWeight: 700, fontSize: "1.1rem",
                color: fitData.relError < 0.01 ? "#4ade80" : fitData.relError < 0.05 ? "#fbbf24" : "#f87171"
              }}>
                {(fitData.relError * 100).toFixed(2)}%
              </span>
            </div>

            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={fitData.chartData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="x" tickFormatter={v => v.toFixed(1)} stroke="#475569" />
                <YAxis stroke="#475569" />
                <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8 }}
                  formatter={(v, n) => [v.toFixed(4), n]} labelFormatter={v => `x = ${(+v).toFixed(3)}`} />
                <Legend />
                <Line type="monotone" dataKey="signal" stroke="#7dd3fc" dot={false} strokeWidth={2} name="Original signal" />
                <Line type="monotone" dataKey="reconstruction" stroke="#f59e0b" dot={false} strokeWidth={2} strokeDasharray="5 3" name={`Chebyshev fit (deg ${fitDegree})`} />
              </LineChart>
            </ResponsiveContainer>

            {/* Residual */}
            <p style={{ color: "#64748b", fontSize: "0.85rem", margin: "0.25rem 0 0.75rem", textAlign: "center" }}>Residual error</p>
            <ResponsiveContainer width="100%" height={100}>
              <LineChart data={fitData.chartData} margin={{ top: 0, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="x" hide />
                <YAxis stroke="#475569" width={45} tickFormatter={v => v.toFixed(2)} />
                <Line type="monotone" dataKey="error" stroke="#ef4444" dot={false} strokeWidth={1.5} name="Error" />
              </LineChart>
            </ResponsiveContainer>

            {/* Coefficients */}
            <div style={{ marginTop: "1rem", background: "#1e293b", borderRadius: 10, padding: "1rem" }}>
              <p style={{ margin: "0 0 0.75rem 0", color: "#94a3b8", fontWeight: 600 }}>Fitted coefficients (one per degree)</p>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
                {fitData.coeffs.map((c, d) => (
                  <div key={d} style={{ background: "#0f172a", borderRadius: 6, padding: "0.4rem 0.75rem", border: `1px solid ${COLORS[d]}55` }}>
                    <span style={{ color: COLORS[d], fontWeight: 700, fontSize: "0.85rem" }}>c{d}</span>
                    <span style={{ color: "#94a3b8", fontSize: "0.8rem", marginLeft: "0.4rem" }}>{c.toFixed(4)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ───── TAB 3: How It Works ───── */}
        {activeTab === "howItWorks" && (
          <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>

            <Step n={1} color="#6366f1" title="PCA first, then compression">
              The code first runs PCA on a weight matrix, giving you a handful of <strong>principal components</strong> —
              each one is a long vector of numbers (one value per parameter position).
              Storing these raw vectors is expensive, so the next step compresses them.
            </Step>

            <Step n={2} color="#f59e0b" title="Build the Chebyshev basis matrix">
              <code style={{ background: "#0f172a", padding: "2px 8px", borderRadius: 4, color: "#86efac" }}>build_basis_numpy(n, "chebyshev", chebyshev_degree=12)</code> creates an
              {" "}<strong>n × 13</strong> matrix (n = number of parameters). Each column is one Chebyshev polynomial
              evaluated at evenly-spaced points in [0,1]. With <code style={{ background: "#0f172a", padding: "2px 8px", borderRadius: 4, color: "#fcd34d" }}>chebyshev_degree=12</code>,
              you get columns T₀ through T₁₂ — 13 columns total.
            </Step>

            <Step n={3} color="#10b981" title='"Degree" = how many polynomials you use'>
              Setting <code style={{ background: "#0f172a", padding: "2px 8px", borderRadius: 4, color: "#fcd34d" }}>chebyshev_degree=d</code> gives you <strong>d+1 basis functions</strong> (T₀ … Tₐ).
              More degrees → more expressive → better fit → but also more coefficients to store.
              The default of 12 gives 13 numbers to describe each PCA component, regardless of how large n is.
              That's the whole point of the compression.
            </Step>

            <Step n={4} color="#3b82f6" title="Least-squares fit">
              <code style={{ background: "#0f172a", padding: "2px 8px", borderRadius: 4, color: "#86efac" }}>fit_vector</code> solves:
              <div style={{ background: "#0f172a", borderRadius: 8, padding: "0.75rem 1rem", fontFamily: "monospace", color: "#fcd34d", margin: "0.5rem 0" }}>
                min ‖ Basis · coefficients − component ‖₂
              </div>
              This produces one coefficient per degree. Those 13 numbers <em>replace</em> the entire component vector (which could be millions of entries long).
            </Step>

            <Step n={5} color="#ec4899" title="Reconstruction at runtime">
              To get the component back, <code style={{ background: "#0f172a", padding: "2px 8px", borderRadius: 4, color: "#86efac" }}>reconstruct_component</code> re-builds the basis matrix
              and multiplies: <code style={{ background: "#0f172a", padding: "2px 8px", borderRadius: 4, color: "#fcd34d" }}>Basis @ coefficients</code>. This is cheap and differentiable —
              perfect for initializing neural network weights.
            </Step>

            <div style={{ background: "#1e293b", borderRadius: 12, padding: "1.25rem", marginTop: "0.5rem" }}>
              <p style={{ margin: "0 0 0.5rem", fontWeight: 700, color: "#f8fafc" }}>Why Chebyshev over regular polynomials?</p>
              <p style={{ margin: 0, color: "#cbd5e1", lineHeight: 1.7 }}>
                Regular polynomials (x, x², x³…) suffer from <em>Runge's phenomenon</em> — they oscillate wildly near the
                edges of the interval. Chebyshev polynomials are <strong>orthogonal</strong> and spread their interpolation
                points more densely near the edges (they're related to cosines), which gives much better
                numerical stability and faster convergence for smooth functions.
                That's why they're the default in this codebase over the <code style={{ background: "#0f172a", padding: "2px 8px", borderRadius: 4, color: "#7dd3fc" }}>"poly"</code> basis type.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Step({ n, color, title, children }) {
  return (
    <div style={{ display: "flex", gap: "1rem", background: "#1e293b", borderRadius: 12, padding: "1.25rem" }}>
      <div style={{
        width: 32, height: 32, borderRadius: "50%", background: color, color: "#fff",
        display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 800,
        fontSize: "0.9rem", flexShrink: 0, marginTop: 2,
      }}>{n}</div>
      <div>
        <p style={{ margin: "0 0 0.5rem", fontWeight: 700, color: "#f8fafc" }}>{title}</p>
        <p style={{ margin: 0, color: "#cbd5e1", lineHeight: 1.7 }}>{children}</p>
      </div>
    </div>
  );
}
