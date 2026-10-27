"""Bures geometry of a qubit section, written to WORK/figures/metric_dome.pdf (paths.FIG_DIR).

(a) The upper half of the x-z section of the Bloch ball with the sets of states at Euclidean coordinate distance 0.1
(left, gray) and at Bures distance 0.05 (right, orange) from a polar grid of states. (b) The same sets lifted to the
hemisphere that the section becomes under the Bures metric (r = sin chi, ds^2 = (dchi^2 + sin^2 chi dphi^2) / 4).
The Bures sets shrink towards the pure boundary, where the Euclidean sets keep their size.

usage: python src/fig_metric_dome.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paths

FIG = paths.FIG_DIR + os.sep
ORANGE = (0.84, 0.37, 0.00)
plt.rcParams.update({"font.size": 7, "legend.fontsize": 6.5, "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})


def bures3(a, b):
    """Bures distance between the qubit states with Bloch vectors a (3,) and b (..., 3)."""
    F = (1 + b @ a + np.sqrt(np.clip((1 - a @ a) * (1 - (b * b).sum(-1)), 0, None))) / 2
    return np.sqrt(np.clip(2 * (1 - np.sqrt(np.clip(F, 0, 1))), 0, None))


def bures_contour_disk(x0, z0, eps, n=240):
    """Exact closed curve of the states in the x-z section at Bures distance eps from (x0, 0, z0), as (n, 2)."""
    r0 = np.array([x0, 0.0, z0])
    out = []
    for p in np.linspace(0, 2 * np.pi, n):
        d = np.array([np.cos(p), 0.0, np.sin(p)])
        lo, hi = 0.0, 0.6
        for _ in range(50):
            mid = (lo + hi) / 2
            b = r0 + mid * d
            if np.linalg.norm(b) > 1 - 1e-12 or bures3(r0, b) > eps:
                hi = mid
            else:
                lo = mid
        out.append([x0 + (lo + hi) / 2 * np.cos(p), z0 + (lo + hi) / 2 * np.sin(p)])
    return np.array(out)


def figure(eps_b=0.05, eps_e=0.1, elev=28, azim=-90):
    """(a) Upper half of the x-z section with the sets at Euclidean coordinate distance eps_e (left, gray) and at
    Bures distance eps_b (right, orange) from a polar grid of states. (b) The same sets lifted to the hemisphere that
    the section becomes under the Bures metric. One column wide, written to metric_dome.pdf."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    fig = plt.figure(figsize=(3.4, 3.6))
    ax = fig.add_axes([0.11, 0.615, 0.86, 0.33])
    bx = fig.add_axes([-0.02, -0.07, 1.04, 0.60], projection="3d", computed_zorder=False)
    bx.patch.set_alpha(0)
    GRAYL = (0.45, 0.45, 0.45)
    t = np.linspace(0, 2 * np.pi, 400)
    centres = [(0.0, 0.0)]
    for r, degs in ((0.35, (25, 65, 115, 155)), (0.62, (12, 42, 72, 108, 138, 168)), (0.88, (12, 42, 72, 108, 138, 168))):
        for deg in degs:
            centres.append((r * np.cos(np.radians(deg)), r * np.sin(np.radians(deg))))
    shapes = []                                                    # (kind, (n, 2) curve in the section)
    for x0, z0 in centres:
        centre = abs(x0) < 1e-9 and abs(z0) < 1e-9
        if x0 >= -1e-9:
            shapes.append(("bures", bures_contour_disk(x0, z0, eps_b)))
        if x0 <= 1e-9:
            shapes.append(("eucl" if not centre else "eucl0", np.stack([x0 + eps_e * np.cos(t), z0 + eps_e * np.sin(t)], -1)))
    # (a) the section
    ax.fill(np.cos(t), np.sin(t), color=(0.97, 0.97, 0.97), lw=0)
    ax.plot(np.cos(t), np.sin(t), color="k", lw=0.6)
    for kind, C in shapes:
        if kind == "bures":
            ax.fill(C[:, 0], C[:, 1], color=ORANGE, alpha=0.18, lw=0)
            ax.plot(C[:, 0], C[:, 1], color=ORANGE, lw=0.9)
        else:
            if kind == "eucl":
                ax.fill(C[:, 0], C[:, 1], color=GRAYL, alpha=0.18, lw=0)
            ax.plot(C[:, 0], C[:, 1], color=GRAYL, lw=0.8, ls="--")
    for x0, z0 in centres:
        ax.plot([x0], [z0], "k.", ms=2.5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.07, 1.07)
    ax.set_ylim(-0.12, 1.07)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=6.5, width=0.6, length=2.5)
    ax.set_xlabel(r"$\langle X\rangle$", labelpad=1)
    ax.set_ylabel(r"$\langle Z\rangle$", labelpad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # (b) the dome, viewed from the front so that the lifted upper half faces the viewer
    bx.set_proj_type("ortho")
    bx.view_init(elev=elev, azim=azim)
    e, a = np.radians(elev), np.radians(azim)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    light = np.array([-0.4, -0.5, 1.0]) + 0.6 * d
    light /= np.linalg.norm(light)
    tint0, tint1 = np.array([1.0, 1.0, 1.0]), np.array([0.70, 0.78, 0.88])
    tint = lambda P: tint0 + (np.clip((np.asarray(P) - 0.5) / 0.5, 0, 1) ** 1.6)[..., None] * (tint1 - tint0)
    u = np.linspace(0, 2 * np.pi, 480)
    v = np.linspace(0, np.pi / 2, 200)
    U, V = np.meshgrid(u, v)
    X, Y, Z = np.sin(V) * np.cos(U), np.sin(V) * np.sin(U), np.cos(V)
    lam = np.clip(X * light[0] + Y * light[1] + Z * light[2], 0, 1)
    h = np.array([X, Y, Z]).transpose(1, 2, 0)
    half = light + d
    half /= np.linalg.norm(half)
    spec = np.clip(h @ half, 0, 1) ** 24
    base = tint((1 + np.sin(V) ** 2) / 2)
    fc = np.clip(base * (0.74 + 0.26 * lam)[..., None] + 0.12 * spec[..., None], 0, 1)
    bx.plot_surface(X, Y, Z, facecolors=np.concatenate([fc, np.ones(fc.shape[:2] + (1,))], -1), linewidth=0,
                    edgecolor="none", antialiased=False, shade=False, zorder=1)
    sh = np.linspace(0, 2 * np.pi, 200)
    bx.add_collection3d(Poly3DCollection([np.stack([1.06 * np.cos(sh) + 0.05, 1.06 * np.sin(sh) - 0.05, 0 * sh - 0.01], -1)],
                                         facecolor=(0.75, 0.75, 0.75), alpha=0.35, edgecolor="none", zorder=0))

    def front(P):
        P = P.copy()
        P[(P @ d) < 0.02] = np.nan
        return P

    rings = [np.sqrt(2 * P - 1) for P in (0.6, 0.8, 0.95, 0.99)]
    for r in rings:
        P = front(np.stack([r * np.cos(t), r * np.sin(t), np.sqrt(1 - r * r) + 0 * t], -1))
        bx.plot(P[:, 0], P[:, 1], P[:, 2], color=(0.5, 0.5, 0.5), lw=0.35, zorder=2)
    chi = np.linspace(0, np.pi / 2, 160)
    for p in np.radians(np.arange(0, 360, 30)):
        P = front(np.stack([np.sin(chi) * np.cos(p), np.sin(chi) * np.sin(p), np.cos(chi)], -1))
        bx.plot(P[:, 0], P[:, 1], P[:, 2], color=(0.5, 0.5, 0.5), lw=0.35, zorder=2)
    P = front(np.stack([np.cos(t), np.sin(t), 0 * t], -1))
    bx.plot(P[:, 0], P[:, 1], P[:, 2], color="k", lw=0.7, zorder=2)
    lift = lambda C: np.stack([C[:, 0], -C[:, 1], np.sqrt(np.clip(1 - C[:, 0] ** 2 - C[:, 1] ** 2, 0, None))], -1)
    for kind, C in shapes:
        inside = C[:, 0] ** 2 + C[:, 1] ** 2 <= 1
        L = lift(C[inside])
        if len(L) < 3:
            continue
        if kind == "bures":
            bx.add_collection3d(Poly3DCollection([L], facecolor=ORANGE, alpha=0.22, edgecolor="none", zorder=3))
            bx.add_collection3d(Line3DCollection([L], colors=[ORANGE], linewidths=0.9, zorder=3))
        else:
            if kind == "eucl":
                bx.add_collection3d(Poly3DCollection([L], facecolor=GRAYL, alpha=0.18, edgecolor="none", zorder=3))
            bx.add_collection3d(Line3DCollection([L], colors=[GRAYL], linewidths=0.8, linestyles="--", zorder=3))
    for x0, z0 in centres:
        p = lift(np.array([[x0, z0]]))[0]
        bx.plot([p[0]], [p[1]], [p[2]], "k.", ms=2.5, zorder=4)
    bx.set_xlim(-1.0, 1.0)
    bx.set_ylim(-1.0, 1.0)
    bx.set_zlim(-0.3, 1.0)
    bx.set_box_aspect((1, 1, 0.65), zoom=1.32)
    bx.set_axis_off()
    fig.text(0.02, 0.945, "(a)", fontsize=7, ha="left", va="center")
    fig.text(0.02, 0.49, "(b)", fontsize=7, ha="left", va="center")
    handles = [matplotlib.lines.Line2D([], [], color=GRAYL, lw=0.8, ls="--", label=f"Euclidean distance {eps_e:g}"),
               matplotlib.lines.Line2D([], [], color=ORANGE, lw=0.9, label=f"Bures distance {eps_b:g}")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.2, columnspacing=2.0)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(FIG + "metric_dome.pdf", dpi=300)
    plt.close(fig)
    print("wrote", FIG + "metric_dome.pdf")


if __name__ == "__main__":
    figure()
