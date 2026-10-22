"""
Loaded die with a corner-to-corner (space-diagonal) cavity and a lead ball
--------------------------------------------------------------------------
Geometry (die faces: opposite pairs 1-6, 2-5, 3-4)
    x-axis : face 3 at +x, face 4 at -x
    y-axis : face 2 at +y, face 5 at -y
    z-axis : face 1 at +z, face 6 at -z

    Corner A (surrounded by faces 1, 2, 4) = (-h, +h, +h)
    Corner B (surrounded by faces 3, 6, 5) = (+h, -h, -h)
    The cylindrical cavity runs along the space diagonal A -> B and holds a
    2 g lead ball that can sit anywhere along it.

Model
-----
* Removed cavity is symmetric about the die centre, so it does not move the
  die's own centre of gravity (it only removes mass).
* The lead ball is treated as a point mass at its centre.
* Net CG = (m_ball * ball_position) / (m_die + m_ball)      [die CG at origin]
* Face probabilities use the same solid-angle-from-CG model as before:
  P(face is on the bottom) = solid angle of that face seen from the CG / 4*pi,
  and the face that SHOWS is the opposite one.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

OPPOSITE = {1: 6, 6: 1, 2: 5, 5: 2, 3: 4, 4: 3}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ geometry
def _tri_solid_angle(apex, p1, p2, p3):
    """Solid angle at `apex` of triangle p1-p2-p3 (Van Oosterom-Strackee)."""
    a = np.asarray(p1) - np.asarray(apex)
    b = np.asarray(p2) - np.asarray(apex)
    c = np.asarray(p3) - np.asarray(apex)
    na, nb, nc = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
    num = abs(np.dot(a, np.cross(b, c)))
    den = na * nb * nc + np.dot(a, b) * nc + np.dot(a, c) * nb + np.dot(b, c) * na
    return 2.0 * np.arctan2(num, den)


def _face_solid_angle(apex, quad):
    v0, v1, v2, v3 = quad
    return _tri_solid_angle(apex, v0, v1, v2) + _tri_solid_angle(apex, v0, v2, v3)


def cube_faces(h):
    def c(sx, sy, sz):
        return np.array([sx * h, sy * h, sz * h])
    return {
        3: [c(1, -1, -1), c(1, 1, -1), c(1, 1, 1), c(1, -1, 1)],
        4: [c(-1, -1, -1), c(-1, -1, 1), c(-1, 1, 1), c(-1, 1, -1)],
        2: [c(-1, 1, -1), c(-1, 1, 1), c(1, 1, 1), c(1, 1, -1)],
        5: [c(-1, -1, -1), c(1, -1, -1), c(1, -1, 1), c(-1, -1, 1)],
        1: [c(-1, -1, 1), c(1, -1, 1), c(1, 1, 1), c(-1, 1, 1)],
        6: [c(-1, -1, -1), c(-1, 1, -1), c(1, 1, -1), c(1, -1, -1)],
    }


def face_probabilities(cg, h):
    """P(face SHOWS) for a die of half-edge h whose centre of gravity is `cg`."""
    faces = cube_faces(h)
    omega = {f: _face_solid_angle(cg, v) for f, v in faces.items()}
    total = sum(omega.values())                     # = 4*pi for a point inside
    p_bottom = {f: omega[f] / total for f in omega}
    return {f: p_bottom[OPPOSITE[f]] for f in p_bottom}


def cavity_volume_mm3(edge_mm, cavity_diameter_mm, n=200):
    """Volume of (cylinder along the space diagonal) ∩ (cube), by midpoint grid."""
    h = edge_mm / 2.0
    r2 = (cavity_diameter_mm / 2.0) ** 2
    u = np.array([1.0, -1.0, -1.0]) / np.sqrt(3.0)   # diagonal direction
    g = (np.arange(n) + 0.5) / n * edge_mm - h        # cell centres
    Y, Z = np.meshgrid(g, g, indexing="ij")
    cell = (edge_mm / n) ** 3
    inside = 0
    for x in g:
        p2 = x * x + Y * Y + Z * Z
        along = x * u[0] + Y * u[1] + Z * u[2]
        inside += np.count_nonzero(p2 - along * along <= r2)
    return inside * cell


# ------------------------------------------------------------------ main model
def sweep_ball_position(edge_mm=19.2,
                        ball_mass_g=2.0,
                        lead_density_g_cm3=11.34,
                        die_density_g_cm3=1.2,      # acrylic / resin die (adjust!)
                        clearance_mm=0.2,           # cavity is this much wider than ball
                        wall_thickness_mm=1.0,      # ball keeps this gap from the corner faces
                        n_points=81,
                        save_path=None,
                        show=True):
    h = edge_mm / 2.0

    # lead ball size from its mass
    ball_vol_cm3 = ball_mass_g / lead_density_g_cm3
    ball_d_mm = 2.0 * (3.0 * ball_vol_cm3 * 1000.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
    R = ball_d_mm / 2.0
    cavity_d_mm = ball_d_mm + clearance_mm

    # die mass after removing the cavity
    V_cube = edge_mm ** 3
    V_cav = cavity_volume_mm3(edge_mm, cavity_d_mm)
    m_die = die_density_g_cm3 * (V_cube - V_cav) / 1000.0

    # ball travel limits (ball centre distance from a corner: (R+wall)*sqrt(3))
    t_min = (R + wall_thickness_mm) / edge_mm
    t_max = 1.0 - t_min
    ts = np.linspace(t_min, t_max, n_points)

    A = np.array([-h, +h, +h])      # corner surrounded by faces 1, 2, 4
    B = np.array([+h, -h, -h])      # corner surrounded by faces 3, 6, 5

    print(f"Die edge            : {edge_mm} mm")
    print(f"Lead ball           : {ball_mass_g} g -> diameter {ball_d_mm:.2f} mm")
    print(f"Cavity diameter     : {cavity_d_mm:.2f} mm, volume {V_cav:.0f} mm^3")
    print(f"Die mass (w/ cavity): {m_die:.3f} g   (density {die_density_g_cm3} g/cm^3)")
    print(f"Ball travel         : t = {t_min:.3f} .. {t_max:.3f} "
          f"(0 = corner 1-2-4, 1 = corner 3-6-5)\n")

    res = {f: [] for f in range(1, 7)}
    shifts = []
    for t in ts:
        pos = A + t * (B - A)
        cg = (ball_mass_g * pos) / (m_die + ball_mass_g)
        shifts.append(np.linalg.norm(cg))
        p = face_probabilities(cg, h)
        for f in res:
            res[f].append(p[f] * 100.0)
    res = {f: np.array(v) for f, v in res.items()}

    # symmetry check: faces 1,2,4 identical; faces 3,5,6 identical
    d124 = max(np.ptp([res[1], res[2], res[4]], axis=0))
    d356 = max(np.ptp([res[3], res[5], res[6]], axis=0))
    print(f"Symmetry check (should be ~0): {d124:.2e}, {d356:.2e}\n")

    # table at a few positions
    print(f"{'t':>6} {'CG shift mm':>12} {'P(1,2,4) %':>12} {'P(3,5,6) %':>12}")
    for i in np.linspace(0, len(ts) - 1, 9).astype(int):
        print(f"{ts[i]:6.3f} {shifts[i]:12.3f} {res[1][i]:12.4f} {res[3][i]:12.4f}")

    # plot
    plt.figure(figsize=(8.5, 5.2))
    plt.plot(ts, res[1], lw=2.5, color="tab:blue", label="Faces 1, 2, 4 (identical)")
    plt.plot(ts, res[3], lw=2.5, color="tab:red", label="Faces 3, 5, 6 (identical)")
    plt.axhline(100 / 6, color="gray", ls="--", lw=1, label="Baseline (16.6667%)")
    plt.axvline(0.5, color="gray", ls=":", lw=1)
    plt.xlabel("Lead ball position along cavity\n"
               "(0 = corner touching faces 1, 2, 4  →  1 = corner touching faces 3, 6, 5)")
    plt.ylabel("Probability of face showing (%)")
    plt.title(f"{ball_mass_g} g lead ball in a corner-to-corner cavity\n"
              f"(die edge {edge_mm} mm, die density {die_density_g_cm3} g/cm³)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(SCRIPT_DIR, "loaded_die_diagonal_plot.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")
    if show:
        plt.show()
    return ts, res


if __name__ == "__main__":
    sweep_ball_position()
