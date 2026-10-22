# Weighted Dice Designer

Estimate how the face probabilities of a six-sided die change when material is removed or a weight is added inside it, and explore the designs in a live desktop UI.

A fair die gives every face 16.6667 %. Drilling a hole or loading a lead ball moves the centre of gravity (CG), and that bias shows up in which face lands on top.

![Die 2 render]
(die1_side_hole.png)
(die2_accurate.png)

## Die designs

Opposite faces always add up to 7: **1-6, 2-5, 3-4**.

| Design | Description |
|--------|-------------|
| **Type 1 – off-centre hole** | A cylindrical hole (default Ø 2 mm) runs from face 1 to face 6, offset toward face 4. Removing material near face 4 shifts the CG toward face 3. |
| **Type 2 – diagonal cavity + lead ball** | A cavity runs along the space diagonal from the corner touching faces **1, 2, 4** to the corner touching faces **3, 6, 5**. It holds a 2 g lead ball that can sit anywhere along it. |

## How the probabilities are computed

1. **Mass and CG.** Uniform density. Removed material counts as negative mass, and the lead ball is a point mass.
2. **Solid-angle model.** The die is split into six pyramids, each with its apex at the CG and its base on one face. A face that subtends a larger solid angle from the CG is more likely to end up on the **bottom**, so the face that **shows** is the opposite one.
3. With the CG at the centre, every face gets exactly 1/6, which is the sanity check for the model.

Solid angles use the Van Oosterom–Strackee triangle formula, with each square face split into two triangles.

## Example results (16 mm die)

**Type 1:** Ø 2 mm hole, 1 mm wall left near face 4

| Face | 1 | 2 | 3 | 4 | 5 | 6 |
|------|---|---|---|---|---|---|
| P (%) | 16.666 | 16.666 | 16.496 | 16.839 | 16.666 | 16.666 |

**Type 2:** 2 g lead ball, die density 1.2 g/cm³ (ball Ø 6.96 mm, cavity Ø 7.16 mm)

| Ball position | P (faces 1, 2, 4) | P (faces 3, 5, 6) |
|---------------|-------------------|-------------------|
| Near corner 1·2·4 | 13.93 % | 19.40 % |
| Centre | 16.67 % | 16.67 % |
| Near corner 3·6·5 | 19.40 % | 13.93 % |

By the symmetry around the diagonal, faces 1, 2, 4 behave identically, and so do faces 3, 5, 6.

## Repository layout

```
.
├── weighted_dice_ui.py          # live desktop UI (Tkinter + matplotlib)
├── loaded_die.py                # Type 1: probabilities + hole-diameter sweep plot
├── loaded_die_diagonal.py       # Type 2: ball-position sweep plot
├── draw_dice_diagrams.py        # Type 1 / Type 2 schematic diagrams
├── draw_die2_accurate.py        # ray-traced Type 2 render (exterior, cutaway, section)
└── images/                      # generated PNGs used in this README
```

## Installation

Requires Python 3.9+ and [Tkinter](https://docs.python.org/3/library/tkinter.html) (included with the standard Windows and macOS installers).

```bash
pip install numpy matplotlib
```

## Usage

### Live UI

```bash
python weighted_dice_ui.py
```

- Choose **Type 1** or **Type 2**.
- Adjust die edge, die density, hole diameter and wall gap (Type 1), or ball mass and ball position (Type 2).
- The six face probabilities update live, with the CG shift and die mass shown beside the chart.
- **Save chart as PNG** writes the chart next to the script.

### Scripts

```bash
python loaded_die.py            # Type 1 table + probability vs. hole diameter plot
python loaded_die_diagonal.py   # Type 2 probability vs. ball position plot
python draw_dice_diagrams.py    # schematic diagrams
python draw_die2_accurate.py    # ray-traced render of Type 2
```

Plots and images are saved in the same folder as the script. Edit the parameters at the top of each script (die edge, density, hole diameter, ball mass, wall gap) to match your own die.

## Assumptions and limitations

- The solid-angle model is a first-order heuristic. It ignores bouncing, spin, friction and the surface the die lands on, and it has **not** been validated against physical rolls.
- Uniform density is assumed. Defaults are a 16 mm die and 1.2 g/cm³ (acrylic or resin). Change these to match your die.
- The Type 2 cavity is wide (about 22 % of the die volume) and breaks through the faces at both corners, so both ends need a plug. The plug's mass is ignored.
- The ball is treated as a point mass at its centre, and the cavity is assumed symmetric about the die centre, so it does not move the CG by itself.
- Large CG shifts push the model further from reality, so treat those results as trends rather than exact values.

## License

Released under the **MIT License**.

```
MIT License

Copyright (c) 2026 Ara

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
