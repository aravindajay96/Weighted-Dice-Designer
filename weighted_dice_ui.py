"""
Weighted Dice Designer - live UI
Pick a die type, move the sliders, and the six face probabilities update.

  Type 1: through-hole (face 1 -> 6), offset toward face 4
  Type 2: corner-to-corner cavity (corner 1·2·4 -> corner 3·6·5) with a lead ball

Model: uniform density, removed material treated as negative mass, ball as a point
mass; P(face on bottom) = solid angle seen from the centre of gravity / 4*pi and the
face that SHOWS is the opposite one. Needs: numpy, matplotlib (tkinter ships with Python).
"""
import os
import tkinter as tk
from functools import lru_cache
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

OPP = {1: 6, 6: 1, 2: 5, 5: 2, 3: 4, 4: 3}
LEAD = 11.34                                   # g/cm^3
DIR = os.path.dirname(os.path.abspath(__file__))


def _tri(a, b, c):
    na, nb, nc = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
    den = na * nb * nc + a @ b * nc + a @ c * nb + b @ c * na
    return 2 * np.arctan2(abs(a @ np.cross(b, c)), den)


def show_probabilities(cg, h):
    """Return {face: P(face shows)} for a cube of half-edge h with CG at `cg`."""
    def c(x, y, z): return np.array([x * h, y * h, z * h]) - cg
    faces = {3: [c(1, -1, -1), c(1, 1, -1), c(1, 1, 1), c(1, -1, 1)],
             4: [c(-1, -1, -1), c(-1, -1, 1), c(-1, 1, 1), c(-1, 1, -1)],
             2: [c(-1, 1, -1), c(-1, 1, 1), c(1, 1, 1), c(1, 1, -1)],
             5: [c(-1, -1, -1), c(1, -1, -1), c(1, -1, 1), c(-1, -1, 1)],
             1: [c(-1, -1, 1), c(1, -1, 1), c(1, 1, 1), c(-1, 1, 1)],
             6: [c(-1, -1, -1), c(-1, 1, -1), c(1, 1, -1), c(1, -1, -1)]}
    om = {f: _tri(*v[:3]) + _tri(v[0], v[2], v[3]) for f, v in faces.items()}
    tot = sum(om.values())
    return {f: om[OPP[f]] / tot for f in om}


@lru_cache(maxsize=256)
def diagonal_cavity_volume(edge, dia, n=90):
    """Volume (mm^3) of a cylinder along the space diagonal, clipped to the cube."""
    h = edge / 2; u = np.array([1., -1., -1.]) / np.sqrt(3)
    g = (np.arange(n) + .5) / n * edge - h
    Y, Z = np.meshgrid(g, g, indexing="ij"); cnt = 0
    for x in g:
        along = x * u[0] + Y * u[1] + Z * u[2]
        cnt += np.count_nonzero(x * x + Y * Y + Z * Z - along ** 2 <= (dia / 2) ** 2)
    return cnt * (edge / n) ** 3


def compute(kind, edge, dens, p1, p2):
    """kind 1: p1=hole diameter, p2=wall gap.  kind 2: p1=ball mass g, p2=position 0..1."""
    h = edge / 2; V = edge ** 3
    if kind == 1:
        r = p1 / 2; off = h - r - p2
        if off < 0: raise ValueError("Hole too large for this wall gap")
        Vh = np.pi * r * r * edge
        cg = -Vh * np.array([-off, 0, 0]) / (V - Vh)
        info = f"hole offset {off:.2f} mm | die mass {dens * (V - Vh) / 1000:.2f} g"
    else:
        Rb = (3 * (p1 / LEAD) * 1000 / (4 * np.pi)) ** (1 / 3); dia = 2 * Rb + 0.2
        m_die = dens * (V - diagonal_cavity_volume(edge, dia)) / 1000
        tmin = (Rb + 1.0) / edge; t = tmin + p2 * (1 - 2 * tmin)
        A, B = np.array([-h, h, h]), np.array([h, -h, -h])
        cg = p1 * (A + t * (B - A)) / (m_die + p1)
        info = f"ball Ø{2 * Rb:.2f} mm | cavity Ø{dia:.2f} mm | die mass {m_die:.2f} g"
    p = show_probabilities(cg, h)
    return p, np.linalg.norm(cg), info


class App:
    def __init__(self, root):
        self.root = root; root.title("Weighted Dice Designer")
        left = tk.Frame(root, padx=10, pady=10); left.pack(side="left", fill="y")
        self.kind = tk.IntVar(value=1)
        for v, txt in [(1, "Type 1: off-centre hole (1→6, toward face 4)"),
                       (2, "Type 2: diagonal cavity + lead ball")]:
            tk.Radiobutton(left, text=txt, variable=self.kind, value=v,
                           command=self.on_mode, anchor="w").pack(fill="x")
        self.s = {}
        self.slider(left, "edge", "Die edge (mm)", 10, 30, 0.5, 16)
        self.slider(left, "dens", "Die density (g/cm³)", 0.8, 2.0, 0.05, 1.2)
        self.slider(left, "hole", "Hole diameter (mm)", 0.5, 8, 0.1, 2)
        self.slider(left, "wall", "Wall left near face 4 (mm)", 0.5, 4, 0.1, 1)
        self.slider(left, "ball", "Lead ball mass (g)", 0.2, 5, 0.1, 2)
        self.slider(left, "pos", "Ball position (0 = corner 1·2·4, 1 = corner 3·6·5)", 0, 1, 0.01, 0)
        self.info = tk.Label(left, justify="left", wraplength=290, pady=8); self.info.pack(fill="x")
        tk.Button(left, text="Save chart as PNG", command=self.save).pack(fill="x")
        self.fig = Figure(figsize=(7, 5)); self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side="right", fill="both", expand=True)
        self.on_mode()

    def slider(self, parent, key, label, lo, hi, step, val):
        f = tk.Frame(parent); f.pack(fill="x", pady=2); self.s[key] = (f, tk.DoubleVar(value=val))
        tk.Label(f, text=label, anchor="w").pack(fill="x")
        tk.Scale(f, from_=lo, to=hi, resolution=step, orient="horizontal", variable=self.s[key][1],
                 command=lambda _=None: self.update()).pack(fill="x")

    def on_mode(self):
        k = self.kind.get()
        for key, on in [("hole", k == 1), ("wall", k == 1), ("ball", k == 2), ("pos", k == 2)]:
            self.s[key][0].pack_forget()
            if on: self.s[key][0].pack(fill="x", pady=2, before=self.info)
        self.update()

    def v(self, key): return self.s[key][1].get()

    def update(self):
        k = self.kind.get(); ax = self.ax; ax.clear()
        try:
            a, b = (self.v("hole"), self.v("wall")) if k == 1 else (self.v("ball"), self.v("pos"))
            p, shift, info = compute(k, self.v("edge"), self.v("dens"), a, b)
        except ValueError as e:
            self.info.config(text=str(e), fg="red"); return
        vals = [p[f] * 100 for f in range(1, 7)]
        cols = ["#3b82f6" if x >= 100 / 6 else "#ef4444" for x in vals]
        ax.bar(range(1, 7), vals, color=cols)
        ax.axhline(100 / 6, color="gray", ls="--", lw=1)
        for i, x in enumerate(vals, 1): ax.text(i, x + .2, f"{x:.2f}%", ha="center", fontsize=9)
        ax.set_xlabel("Face showing"); ax.set_ylabel("Probability (%)")
        ax.set_ylim(0, max(25, max(vals) * 1.15)); ax.set_title("Face probabilities (dashed = fair die 16.67%)")
        self.fig.tight_layout(); self.canvas.draw()
        self.info.config(text=f"CG shift {shift:.3f} mm\n{info}", fg="black")

    def save(self):
        path = os.path.join(DIR, "weighted_dice_chart.png"); self.fig.savefig(path, dpi=150)
        self.info.config(text=self.info.cget("text") + f"\nSaved: {path}")


if __name__ == "__main__":
    root = tk.Tk(); App(root); root.mainloop()
