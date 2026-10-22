"""
Loaded-die probability estimator
---------------------------------
Model
-----
A standard cube die (opposite faces: 1-6, 2-5, 3-4) is turned into a
"loaded" die by drilling a 2 mm diameter cylindrical hole all the way
through it, parallel to the 1-6 axis, positioned off-centre so that it
runs close to face 4 (removing material on the face-4 side).

Removing that material shifts the die's centre of gravity (CG) away
from face 4, towards face 3 - exactly as described.

Physical model used
--------------------
This uses the standard "solid-angle from the centre of gravity"
approximation for loaded dice:

  * Split the die into 6 pyramids, each with its apex at the CG and
    its base being one face of the die.
  * The face whose pyramid subtends the LARGEST solid angle (as seen
    from the CG) is the face most likely to end up on the BOTTOM,
    because that face is geometrically "closest" to the CG.
  * The value that actually SHOWS (the top face) is the OPPOSITE face
    of whichever face is most likely to be on the bottom.

This reduces exactly to 1/6 for every face when the die is solid, and
gives a sensible first-order estimate for small mass perturbations
(such as a small hole) without needing a full rigid-body bounce/roll
simulation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

OPPOSITE = {1: 6, 6: 1, 2: 5, 5: 2, 3: 4, 4: 3}

# Folder the script itself lives in - the plot is saved next to it,
# regardless of where you run the script from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _tri_solid_angle(apex, p1, p2, p3):
    """Solid angle (steradians) subtended at `apex` by triangle p1-p2-p3
    (Van Oosterom-Strackee formula)."""
    a = np.asarray(p1) - np.asarray(apex)
    b = np.asarray(p2) - np.asarray(apex)
    c = np.asarray(p3) - np.asarray(apex)

    na, nb, nc = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)

    numerator = abs(np.dot(a, np.cross(b, c)))
    denominator = (na * nb * nc
                   + np.dot(a, b) * nc
                   + np.dot(a, c) * nb
                   + np.dot(b, c) * na)
    return 2.0 * np.arctan2(numerator, denominator)


def _face_solid_angle(apex, quad):
    """Solid angle subtended by a planar quadrilateral face (4 ordered,
    non-crossing corner points), split into two triangles."""
    v0, v1, v2, v3 = quad
    return (_tri_solid_angle(apex, v0, v1, v2) +
            _tri_solid_angle(apex, v0, v2, v3))


def loaded_die_probabilities(edge_mm=16.0,
                              hole_diameter_mm=2.0,
                              hole_offset_mm=None,
                              wall_thickness_mm=1.0,
                              verbose=True):
    """
    edge_mm            : die edge length (mm)
    hole_diameter_mm    : diameter of the through-hole (mm)
    hole_offset_mm      : distance of the hole axis from the die centre,
                          towards face 4 (mm). If None, the hole is
                          placed as close to face 4 as possible while
                          leaving `wall_thickness_mm` of material.
    wall_thickness_mm   : material left between hole and face-4 surface,
                          used only when hole_offset_mm is None
    """
    r = hole_diameter_mm / 2.0
    h = edge_mm / 2.0                       # half edge

    if hole_offset_mm is None:
        hole_offset_mm = h - r - wall_thickness_mm
    if hole_offset_mm < 0 or hole_offset_mm + r >= h:
        raise ValueError("Hole does not fit inside the die with the given "
                          "diameter / offset / wall thickness.")

    # --- coordinate system ---------------------------------------------
    # x-axis: face3 at +x, face4 at -x
    # y-axis: face2 at +y, face5 at -y
    # z-axis: face1 at +z, face6 at -z
    # hole runs parallel to z (face1 -> face6), offset towards face4 (-x)
    x0, y0 = -hole_offset_mm, 0.0

    # --- volumes / masses (uniform density -> volume ~ mass) -----------
    V_cube = edge_mm ** 3
    V_hole = np.pi * r ** 2 * edge_mm          # through-hole, full length
    V_net = V_cube - V_hole

    CG_cube = np.zeros(3)
    CG_hole = np.array([x0, y0, 0.0])          # hole z-centroid = 0 (symmetric)
    CG = (V_cube * CG_cube - V_hole * CG_hole) / V_net   # net CG

    # --- cube geometry ---------------------------------------------------
    def corner(sx, sy, sz):
        return np.array([sx * h, sy * h, sz * h])

    faces = {
        3: [corner(1, -1, -1), corner(1, 1, -1), corner(1, 1, 1), corner(1, -1, 1)],
        4: [corner(-1, -1, -1), corner(-1, -1, 1), corner(-1, 1, 1), corner(-1, 1, -1)],
        2: [corner(-1, 1, -1), corner(-1, 1, 1), corner(1, 1, 1), corner(1, 1, -1)],
        5: [corner(-1, -1, -1), corner(1, -1, -1), corner(1, -1, 1), corner(-1, -1, 1)],
        1: [corner(-1, -1, 1), corner(1, -1, 1), corner(1, 1, 1), corner(-1, 1, 1)],
        6: [corner(-1, -1, -1), corner(-1, 1, -1), corner(1, 1, -1), corner(1, -1, -1)],
    }

    omega = {f: _face_solid_angle(CG, verts) for f, verts in faces.items()}
    total = sum(omega.values())                # should equal 4*pi

    p_bottom = {f: omega[f] / total for f in omega}
    p_top = {f: p_bottom[OPPOSITE[f]] for f in p_bottom}   # face that SHOWS

    if verbose:
        print(f"Die edge            : {edge_mm} mm")
        print(f"Hole diameter       : {hole_diameter_mm} mm")
        print(f"Hole offset from ctr: {hole_offset_mm:.3f} mm (towards face 4)")
        print(f"CG shift            : {np.round(CG, 5)} mm (die centre = origin)")
        print(f"Sum of solid angles : {total:.6f} sr  (expect {4*np.pi:.6f})\n")
        print(f"{'Face':>4} | {'P(shown)':>10}")
        print("-" * 19)
        for f in range(1, 7):
            print(f"{f:>4} | {p_top[f]*100:9.4f}%")
        print("-" * 19)
        print(f"{'sum':>4} | {sum(p_top.values())*100:9.4f}%")

    return p_top, CG


def sweep_hole_diameter(edge_mm=19.2,
                         diameters_mm=None,
                         wall_thickness_mm=1.0,
                         save_path=None):
    """
    Compute P(face shows) across a range of hole diameters (wall_thickness_mm
    stays fixed, so the hole always sits right next to face 4 as it grows)
    and plot every face's probability curve vs. hole diameter.
    """
    if save_path is None:
        save_path = os.path.join(SCRIPT_DIR, "loaded_die_plot.png")

    h = edge_mm / 2.0
    if diameters_mm is None:
        # stay comfortably inside the "offset >= 0" limit: d <= 2*(h - wall)
        d_max = 2 * (h - wall_thickness_mm) * 0.9
        diameters_mm = np.linspace(0.0, d_max, 60)

    results = {f: [] for f in range(1, 7)}
    for d in diameters_mm:
        p_top, _ = loaded_die_probabilities(edge_mm=edge_mm,
                                             hole_diameter_mm=d,
                                             wall_thickness_mm=wall_thickness_mm,
                                             verbose=False)
        for f in range(1, 7):
            results[f].append(p_top[f] * 100.0)

    plt.figure(figsize=(8, 5))
    for f in range(1, 7):
        plt.plot(diameters_mm, results[f], label=f"Face {f}", linewidth=2)
    plt.axhline(100 / 6, color="gray", linestyle="--", linewidth=1,
                label="Baseline (16.6667%)")
    plt.xlabel("Hole diameter (mm)")
    plt.ylabel("Probability of face showing (%)")
    plt.title(f"Face probability vs. hole diameter\n"
              f"(die edge = {edge_mm} mm, hole near face 4, "
              f"wall left = {wall_thickness_mm} mm)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    plt.show()

    return diameters_mm, results


if __name__ == "__main__":
    print("=== Baseline (solid die, no hole) ===")
    loaded_die_probabilities(edge_mm=19.2, hole_diameter_mm=0.0)

    print("\n=== With 2 mm hole drilled near face 4 (1 mm wall left) ===")
    loaded_die_probabilities(edge_mm=19.2, hole_diameter_mm=4.0, wall_thickness_mm=1.0)

    print("\n=== Sweeping hole diameter from 0 mm up ===")
    sweep_hole_diameter(edge_mm=19.2, wall_thickness_mm=1.0)