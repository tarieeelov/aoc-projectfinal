"""
paper_analysis.py - extra analyses reported in the paper, computed from results/ written by
bdd_night_experiment.py (needs analyze.py next to it; no images or GPU needed):
  * paired YOLOv8n vs YOLO11n differences (Section 4.1)
  * height-adjusted recall by pedestrian luma with bootstrap CIs (Table 5, Figure 2)
  * Figure 1 with hardware labels
Usage: python paper_analysis.py path/to/results [output_dir]
"""
import os, sys, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import analyze as A
R = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else R
meta, z = A.load(R)
ev, gt_rec, stats = A.build(meta, z, 25)
M = meta["models"]; rng = np.random.default_rng(7); B = 1000
out = {}
# 1) paired bootstrap: night differences YOLOv8n - YOLO11n, and difference of day-night drops
boot = {m: {c: [] for c in A.CONDS} for m in M}
for c in A.CONDS:
    N = ev[M[0]][c].N
    for _ in range(B):
        w = np.bincount(rng.integers(0, N, N), minlength=N).astype(float)
        for m in M:
            boot[m][c].append(ev[m][c].metrics(w, full=False))
pt = {m: {c: ev[m][c].metrics(full=False) for c in A.CONDS} for m in M}
a, b = "YOLOv8n", "YOLO11n"
for k in ["AP50", "recall", "precision"]:
    for c in A.CONDS:
        d = [x[k] - y[k] for x, y in zip(boot[a][c], boot[b][c])]
        out[f"{c} {k} v8-11"] = (pt[a][c][k] - pt[b][c][k], A.ci(d))
    dd = [(xd[k] - xn[k]) - (yd[k] - yn[k]) for xd, xn, yd, yn in zip(boot[a]["daytime"], boot[a]["night"], boot[b]["daytime"], boot[b]["night"])]
    out[f"drop {k} v8 minus drop 11"] = ((pt[a]["daytime"][k] - pt[a]["night"][k]) - (pt[b]["daytime"][k] - pt[b]["night"][k]), A.ci(dd))
for k, (v, c) in out.items():
    print(f"{k:32s} {100*v:6.1f} pp  [{100*c[0]:.1f}, {100*c[1]:.1f}]")
# 2) recall by luma within height bins
print("\nRecall (%) by luma bin within height bin  [cond: n/recall]")
lum_e = [0, 30, 60, 100, 256]; h_e = [25, 50, 100, 1e4]
tab = {}
for m in ["YOLOv8n", "YOLO11n"]:
    g = gt_rec[m]
    for hl, hh in zip(h_e[:-1], h_e[1:]):
        row = []
        for ll, lh in zip(lum_e[:-1], lum_e[1:]):
            cell = []
            for c in (0, 1):
                s = (g[:, 0] == c) & (g[:, 2] >= hl) & (g[:, 2] < hh) & (g[:, 1] >= ll) & (g[:, 1] < lh)
                n = int(s.sum()); r = g[s, 3].mean() * 100 if n else float("nan")
                cell.append((n, r))
            row.append(cell)
        tab[(m, hl)] = row
        print(m, f"h {hl}-{hh:.0f}", " | ".join(f"Y{ll}-{lh-1}: D {n0}/{r0:.0f} N {n1}/{r1:.0f}" for (ll, lh), ((n0, r0), (n1, r1)) in zip(zip(lum_e[:-1], lum_e[1:]), row)))
# height-adjusted luma comparison: reweight day GTs to night height distribution within each luma bin
print("\nNight vs day recall within luma bins, day reweighted to the night height mix (bins with >=30 night boxes):")
for m in ["YOLOv8n", "YOLO11n", "SSDLite320-MobileNetV3"]:
    g = gt_rec[m]; parts = []
    for ll, lh in zip(lum_e[:-1], lum_e[1:]):
        rd = rn = nn = 0
        for hl, hh in zip(h_e[:-1], h_e[1:]):
            sd = (g[:, 0] == 0) & (g[:, 2] >= hl) & (g[:, 2] < hh) & (g[:, 1] >= ll) & (g[:, 1] < lh)
            sn = (g[:, 0] == 1) & (g[:, 2] >= hl) & (g[:, 2] < hh) & (g[:, 1] >= ll) & (g[:, 1] < lh)
            if sn.sum() and sd.sum():
                rd += sn.sum() * g[sd, 3].mean(); rn += g[sn, 3].sum(); nn += sn.sum()
        if nn >= 30:
            parts.append(f"Y{ll}-{lh-1}: day(adj) {100*rd/nn:.1f} night {100*rn/nn:.1f} (n={nn})")
    print(m, " | ".join(parts))
# 3) SSDLite localisation: best IoU of confident detections with valid GT
print("\nShare of valid GT whose best overlapping detection (score>=0.25) has IoU in [0.3,0.5) vs >=0.5")
gt = z["gt"]; valid = (gt[:, 5] == 0) & ((gt[:, 4] - gt[:, 2]) >= 25)
for m in M:
    D = z["det_" + m]; D = D[D[:, 5] >= 0.25]
    lo = hi = 0
    gi = gt[valid]
    for i in np.unique(gi[:, 0]).astype(int):
        G = gi[gi[:, 0] == i][:, 1:5]; d = D[D[:, 0] == i][:, 1:5]
        if len(d) == 0: continue
        best = A._iou(d, G).max(0)
        lo += ((best >= 0.3) & (best < 0.5)).sum(); hi += (best >= 0.5).sum()
    print(f"{m:24s} IoU 0.3-0.5: {100*lo/len(gi):.1f}%  IoU>=0.5: {100*hi/len(gi):.1f}%")

LUM = [0, 30, 60, 100, 256]; HB = [25, 50, 100, 1e4]
rng = np.random.default_rng(11)
cond = z["img_cond"]; Nimg = len(cond)
def adj(g, wimg):
    """day recall reweighted to the night height mix, night recall, per luma bin; wimg: image weights"""
    w = wimg[g[:, 4].astype(int)]
    res = []
    for ll, lh in zip(LUM[:-1], LUM[1:]):
        rd = rn = nn = 0.0
        for hl, hh in zip(HB[:-1], HB[1:]):
            s = (g[:, 2] >= hl) & (g[:, 2] < hh) & (g[:, 1] >= ll) & (g[:, 1] < lh)
            sd, sn = s & (g[:, 0] == 0), s & (g[:, 0] == 1)
            wd, wn = w[sd].sum(), w[sn].sum()
            if wd > 0 and wn > 0:
                rd += wn * (w[sd] * g[sd, 3]).sum() / wd; rn += (w[sn] * g[sn, 3]).sum(); nn += wn
        res.append((rd / nn if nn else np.nan, rn / nn if nn else np.nan))
    return np.array(res)

T5 = {}
for m in M:
    g = gt_rec[m]
    pt = adj(g, np.ones(Nimg))
    bs = []
    for _ in range(B):
        w = np.zeros(Nimg)
        for c in (0, 1):
            ids = np.nonzero(cond == c)[0]
            w[ids] = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
        bs.append(adj(g, w))
    bs = np.array(bs)
    diff = bs[:, :, 1] - bs[:, :, 0]
    raw = []
    for ll, lh in zip(LUM[:-1], LUM[1:]):
        row = []
        for c in (0, 1):
            s = (g[:, 0] == c) & (g[:, 1] >= ll) & (g[:, 1] < lh); row.append((int(s.sum()), float(g[s, 3].mean())))
        raw.append(row)
    T5[m] = {"raw": raw, "day_adj": pt[:, 0].tolist(), "night": pt[:, 1].tolist(),
             "diff": (pt[:, 1] - pt[:, 0]).tolist(), "diff_ci": [A.ci(diff[:, k]) for k in range(len(LUM) - 1)]}
    print(m)
    for k, (ll, lh) in enumerate(zip(LUM[:-1], LUM[1:])):
        (nd, rd), (nn, rn) = raw[k]
        print(f"  Y{ll}-{lh-1}: day {100*rd:.1f} (n={nd}) night {100*rn:.1f} (n={nn}) | day adj {100*pt[k,0]:.1f} diff {100*(pt[k,1]-pt[k,0]):.1f} [{100*T5[m]['diff_ci'][k][0]:.1f}, {100*T5[m]['diff_ci'][k][1]:.1f}]")


S = json.load(open(R + "/summary.json"))
json.dump(T5, open(OUT + "/table5_adjusted.json", "w"), indent=1)
COL = ["#2a78d6", "#eb6834", "#1baf7a"]; MK = ["o", "s", "^"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888888",
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "font.family": "DejaVu Serif"})
# Figure 1
lat = meta["latency"]; fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
for ax, d, lab in zip(axes, ["cuda", "cpu"], ["Tesla T4 GPU", "Xeon CPU (2 threads)"]):
    for k, m in enumerate(M):
        x = lat[m][d]["mean_ms"]; r = S["main"]["models"][m]["night"]["AP50"]; y = 100 * r["value"]
        ax.errorbar(x, y, yerr=[[y - 100 * r["ci95"][0]], [100 * r["ci95"][1] - y]], fmt=MK[k], ms=7, color=COL[k],
                    elinewidth=1.5, capsize=3, mec="white", mew=1.2, label=m)
    ax.set_xlabel(f"Mean latency (ms), {lab}"); ax.set_xlim(0, max(lat[m][d]["mean_ms"] for m in M) * 1.2); ax.set_ylim(0, 40)
axes[0].set_ylabel("Night-time AP50 (%)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(OUT + "/fig1.png", dpi=220); plt.close(fig)
# Figure 2: height-adjusted day vs night recall by luma bin
fig, ax = plt.subplots(figsize=(6.6, 3.0)); labels = ["0–29", "30–59", "60–99", "100–255"]; xs = np.arange(4)
for k, m in enumerate(M):
    off = (k - 1) * 0.08
    ax.plot(xs + off, 100 * np.array(T5[m]["day_adj"]), "-", marker=MK[k], color=COL[k], lw=2, ms=6, label=f"{m}, day (height-adjusted)")
    ax.plot(xs + off, 100 * np.array(T5[m]["night"]), "--", marker=MK[k], color=COL[k], lw=2, ms=6, mfc="white", mew=1.5, label=f"{m}, night")
ax.set_xticks(xs, labels); ax.set_xlabel("Mean luma of the pedestrian box (Y, 0–255)"); ax.set_ylabel("Recall at score ≥ 0.25 (%)"); ax.set_ylim(0, 45)
from matplotlib.lines import Line2D
H = [Line2D([], [], color=COL[k], marker=MK[k], lw=2, ms=6, label=m) for k, m in enumerate(M)]
H += [Line2D([], [], color="#555555", lw=2, ls="-", label="day, reweighted to night heights"), Line2D([], [], color="#555555", lw=2, ls="--", marker="o", mfc="white", label="night")]
fig.legend(handles=H, loc="upper center", ncol=3, frameon=False, fontsize=7.5)
fig.tight_layout(rect=(0, 0, 1, 0.84)); fig.savefig(OUT + "/fig2.png", dpi=220); plt.close(fig)
# Appendix D figure: two failure cases side by side
from PIL import Image
ims = [Image.open(R + "/failures/" + f) for f in sorted(os.listdir(R + "/failures"))[2:4]]
W = Image.new("RGB", (1280 * 2 + 20, 720), "white"); W.paste(ims[0], (0, 0)); W.paste(ims[1], (1300, 0)); W = W.resize((1290, 360)); W.save(OUT + "/figD.jpg", quality=90)
print("figs ok")
