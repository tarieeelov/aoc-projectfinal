"""
bdd_night_experiment.py  -  ONE file, ONE run.
Day vs night pedestrian detection on BDD100K val with three lightweight, COCO-pretrained
detectors (YOLOv8n, YOLO11n, SSDLite320-MobileNetV3-Large), zero-shot.

It (1) finds the BDD100K val images + labels automatically, (2) runs every model once on
every daytime and night image, (3) measures latency (GPU and CPU), (4) computes all metrics
with 95% bootstrap intervals, tables, figures, and (5) packs everything into results.zip.

KAGGLE (recommended, free GPU, no download):
  New Notebook -> Add Input -> dataset "solesensei/solesensei_bdd100k"
  Settings: Accelerator = GPU T4 x2 (or P100), Internet = On
  Cell 1:  !pip install -q ultralytics
  Cell 2:  paste this whole file and run it (or upload it and run: !python bdd_night_experiment.py)
  Output:  /kaggle/working/results.zip  (Output tab -> download)

LOCAL:
  pip install ultralytics torch torchvision opencv-python matplotlib psutil
  python bdd_night_experiment.py --root path/to/unzipped/bdd100k
Quick test first:  add  --n 50 --B 50
"""
import argparse, json, math, os, platform, random, shutil, subprocess, sys, time
import cv2, numpy as np, torch, torchvision


IOU_THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)   # 10 thresholds
REC_THRS = np.linspace(0, 1, 101)
MAX_DETS = 100
CONF_OP = 0.25
HEIGHT_MARGIN = 1.25
PED, RIDER, OTHER = 0, 1, 2
CONDS = ["daytime", "night"]
LUM_EDGES = [0, 30, 60, 100, 150, 256]                 # mean luma (Y, 0-255) of the GT box
HEIGHT_EDGES = [25, 50, 100, 10_000]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]             # fixed categorical order


# ----------------------------------------------------------------------------- matching
def _iou(d, g):
    """IoU matrix [len(d), len(g)] for xyxy boxes."""
    ix1 = np.maximum(d[:, None, 0], g[None, :, 0]); iy1 = np.maximum(d[:, None, 1], g[None, :, 1])
    ix2 = np.minimum(d[:, None, 2], g[None, :, 2]); iy2 = np.minimum(d[:, None, 3], g[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    ad = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1]); ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return inter / np.maximum(ad[:, None] + ag[None, :] - inter, 1e-9)


def _ioa(d, g):
    """Intersection over detection area [len(d), len(g)] (COCO crowd / Caltech ignore rule)."""
    ix1 = np.maximum(d[:, None, 0], g[None, :, 0]); iy1 = np.maximum(d[:, None, 1], g[None, :, 1])
    ix2 = np.minimum(d[:, None, 2], g[None, :, 2]); iy2 = np.minimum(d[:, None, 3], g[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    ad = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])
    return inter / np.maximum(ad[:, None], 1e-9)


def match_image(dets, gv, gi):
    """COCO-style greedy matching for one image.
    dets: [D,5] xyxy+score (any order); gv: [G,4] valid GT; gi: [K,4] ignore regions.
    Returns scores [D'] (sorted desc, truncated to MAX_DETS), tp [T,D'], ig [T,D'],
    gt_score [G] = score of the detection matched to each valid GT at IoU 0.5 (-1 if none)."""
    T, G = len(IOU_THRS), len(gv)
    gt_score = -np.ones(G, np.float32)
    if len(dets) == 0:
        return np.zeros(0, np.float32), np.zeros((T, 0), bool), np.zeros((T, 0), bool), gt_score
    order = np.argsort(-dets[:, 4], kind="mergesort")[:MAX_DETS]
    dets = dets[order]; D = len(dets)
    tp = np.zeros((T, D), bool); ig = np.zeros((T, D), bool)
    iou = _iou(dets[:, :4], gv) if G else np.zeros((D, 0))
    ioa = _ioa(dets[:, :4], gi) if len(gi) else np.zeros((D, 0))
    best_any = np.maximum(iou.max(1) if G else np.zeros(D), ioa.max(1) if len(gi) else np.zeros(D))
    used = np.zeros((T, G), bool)
    for d in np.nonzero(best_any >= IOU_THRS[0])[0]:   # others are FP at every threshold
        for t, thr in enumerate(IOU_THRS):
            if G:
                cand = np.where(~used[t] & (iou[d] >= thr), iou[d], -1.0)
                j = int(cand.argmax())
                if cand[j] >= 0:
                    used[t, j] = True; tp[t, d] = True
                    if t == 0:
                        gt_score[j] = dets[d, 4]
                    continue
            if len(gi) and ioa[d].max() >= thr:
                ig[t, d] = True
    return dets[:, 4].astype(np.float32), tp, ig, gt_score


# ----------------------------------------------------------------------------- AP
def ap_from_sorted(tp, fp, npig):
    """COCO 101-point interpolated AP; tp/fp are (weighted) per-detection arrays already
    sorted by descending score; npig = (weighted) number of valid GT boxes."""
    if npig <= 0:
        return float("nan")
    tps = np.cumsum(tp); fps = np.cumsum(fp)
    rc = tps / npig
    pr = tps / np.maximum(tps + fps, np.finfo(np.float64).eps)
    pr = np.maximum.accumulate(pr[::-1])[::-1] if len(pr) else pr
    idx = np.searchsorted(rc, REC_THRS, side="left")
    q = np.zeros(len(REC_THRS))
    ok = idx < len(pr)
    q[ok] = pr[idx[ok]]
    return float(q.mean())


class CondEval:
    """Matched detections of one model on one condition, flattened for fast (bootstrap) metrics."""

    def __init__(self, per_image):
        # per_image: list of (scores, tp, ig, n_valid_gt)
        self.N = len(per_image)
        self.ngt = np.array([p[3] for p in per_image], np.float64)
        img = np.concatenate([np.full(len(p[0]), i) for i, p in enumerate(per_image)]) if self.N else np.zeros(0, int)
        sc = np.concatenate([p[0] for p in per_image]) if self.N else np.zeros(0)
        tp = np.concatenate([p[1] for p in per_image], 1) if self.N else np.zeros((len(IOU_THRS), 0), bool)
        ig = np.concatenate([p[2] for p in per_image], 1) if self.N else np.zeros((len(IOU_THRS), 0), bool)
        o = np.argsort(-sc, kind="mergesort")
        self.img, self.sc, self.tp, self.fp = img[o], sc[o], tp[:, o], (~tp & ~ig)[:, o]
        op = self.sc >= CONF_OP
        self.tp_op = np.bincount(self.img[op], weights=self.tp[0, op], minlength=self.N)
        self.fp_op = np.bincount(self.img[op], weights=self.fp[0, op], minlength=self.N)

    def metrics(self, w=None, full=True):
        w = np.ones(self.N) if w is None else w
        wd = w[self.img]
        npig = float((w * self.ngt).sum())
        aps = [ap_from_sorted(self.tp[t] * wd, self.fp[t] * wd, npig)
               for t in (range(len(IOU_THRS)) if full else [0])]
        TP = float((w * self.tp_op).sum()); FP = float((w * self.fp_op).sum()); FN = npig - TP
        rec = TP / npig if npig else float("nan")
        out = {"AP50": aps[0], "precision": TP / max(TP + FP, 1e-9), "recall": rec, "miss_rate": 1 - rec,
               "TP": TP, "FP": FP, "FN": FN}
        if full:
            out["AP"] = float(np.mean(aps))
        return out


# ----------------------------------------------------------------------------- data
def load(results_dir):
    meta = json.load(open(os.path.join(results_dir, "run_meta.json")))
    z = np.load(os.path.join(results_dir, "raw_detections.npz"))
    return meta, z


def build(meta, z, min_h):
    """Returns {model: {cond: CondEval}}, per-GT records, and per-condition dataset stats."""
    cond = z["img_cond"]; gt = z["gt"]
    h = gt[:, 4] - gt[:, 2]
    valid = (gt[:, 5] == PED) & (h >= min_h)
    gi_all = gt[:, 0].astype(int)
    o = np.argsort(gi_all, kind="mergesort")
    cut = np.searchsorted(gi_all[o], np.arange(len(cond) + 1))
    gt_by_img = {i: o[cut[i]:cut[i + 1]] for i in range(len(cond)) if cut[i + 1] > cut[i]}
    stats = {}
    for c, name in enumerate(CONDS):
        ims = np.nonzero(cond == c)[0]
        nb = int(sum(valid[gt_by_img.get(i, [])].sum() for i in ims))
        nimg_ped = int(sum(valid[gt_by_img.get(i, [])].any() for i in ims))
        stats[name] = {"images": int(len(ims)), "images_with_pedestrian": nimg_ped, "pedestrian_boxes": nb,
                       "boxes_per_image_with_ped": nb / max(nimg_ped, 1),
                       "mean_image_luma": float(z["img_lum"][ims].mean()) if len(ims) else float("nan")}
    out, gt_rec = {}, {}
    for m in meta["models"]:
        D = z[f"det_{m}"]
        D = D[(D[:, 4] - D[:, 2]) >= min_h / HEIGHT_MARGIN]
        det_idx = D[:, 0].astype(int)
        starts = np.searchsorted(det_idx, np.arange(len(cond) + 1))   # dets saved sorted by image
        out[m] = {}
        recs = []
        for c, name in enumerate(CONDS):
            per = []
            for i in np.nonzero(cond == c)[0]:
                gidx = gt_by_img.get(i, np.zeros(0, int))
                gv_idx = gidx[valid[gidx]]; gi_idx = gidx[~valid[gidx]]
                s, tp, ig, gsc = match_image(D[starts[i]:starts[i + 1], 1:6], gt[gv_idx, 1:5], gt[gi_idx, 1:5])
                per.append((s, tp, ig, len(gv_idx)))
                for k, j in enumerate(gv_idx):
                    recs.append((c, gt[j, 6], h[j], gsc[k] >= CONF_OP, i, j))
            out[m][name] = CondEval(per)
        gt_rec[m] = np.array(recs, dtype=np.float64).reshape(-1, 6)
    return out, gt_rec, stats


# ----------------------------------------------------------------------------- stats helpers
def ci(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [float("nan")] * 2


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - r, c + r]


def analyse(meta, z, min_h, B, seed, full=True):
    ev, gt_rec, stats = build(meta, z, min_h)
    models = meta["models"]
    rng = np.random.default_rng(seed)
    point = {m: {c: ev[m][c].metrics(full=full) for c in CONDS} for m in models}
    boots = {m: {c: [] for c in CONDS} for m in models}
    for c in CONDS:
        N = ev[models[0]][c].N
        for _ in range(B):
            w = np.bincount(rng.integers(0, N, N), minlength=N).astype(float)   # same resample for all models
            for m in models:
                boots[m][c].append(ev[m][c].metrics(w, full=full))
    keys = ["AP50", "AP", "precision", "recall", "miss_rate"] if full else ["AP50", "precision", "recall", "miss_rate"]
    res = {"min_h": min_h, "dataset": stats, "models": {}}
    for m in models:
        r = {}
        for c in CONDS:
            r[c] = {k: {"value": point[m][c][k], "ci95": ci([b[k] for b in boots[m][c]])} for k in keys}
            r[c]["counts"] = {k: point[m][c][k] for k in ("TP", "FP", "FN")}
        for k in keys:
            dv = point[m]["daytime"][k] - point[m]["night"][k]
            db = [bd[k] - bn[k] for bd, bn in zip(boots[m]["daytime"], boots[m]["night"])]
            r[f"delta_{k}"] = {"value": dv, "ci95": ci(db),
                               "relative_pct": 100 * dv / point[m]["daytime"][k] if point[m]["daytime"][k] else None}
        res["models"][m] = r
    # rankings and paired model differences (AP50)
    res["ranking"] = {}
    for c in CONDS:
        order = sorted(models, key=lambda m: -point[m][c]["AP50"])
        pairs = {}
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                d = [x["AP50"] - y["AP50"] for x, y in zip(boots[a][c], boots[b][c])]
                lo, hi = ci(d)
                pairs[f"{a} - {b}"] = {"diff": point[a][c]["AP50"] - point[b][c]["AP50"], "ci95": [lo, hi],
                                       "resolved": bool(lo > 0 or hi < 0)}
        res["ranking"][c] = {"order_by_AP50": order, "pairwise_AP50": pairs}
    res["ranking_changed"] = res["ranking"]["daytime"]["order_by_AP50"] != res["ranking"]["night"]["order_by_AP50"]
    # recall by GT luminance and by GT height (operating point)
    res["by_luminance"], res["by_height"] = {}, {}
    for m in models:
        g = gt_rec[m]
        res["by_luminance"][m] = _binned(g, 1, LUM_EDGES)
        res["by_height"][m] = _binned(g, 2, HEIGHT_EDGES)
    return res, gt_rec


def _binned(g, col, edges):
    out = {}
    for c, name in enumerate(CONDS):
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (g[:, 0] == c) & (g[:, col] >= lo) & (g[:, col] < hi)
            n = int(sel.sum()); k = int(g[sel, 3].sum())
            rows.append({"bin": [lo, hi], "n": n, "recall": k / n if n else None, "ci95": wilson(k, n)})
        out[name] = rows
    return out


# ----------------------------------------------------------------------------- reporting
def f(x, p=1, pct=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    v = (100 * x if pct else x) + 0.0
    s = f"{v:.{p}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def rel(x):
    return "n/a" if x is None else f"{x:.0f}%"


def fc(d, p=1):
    return f"{f(d['value'], p)} [{f(d['ci95'][0], p)}, {f(d['ci95'][1], p)}]"


def tables_md(meta, main, sens):
    M = meta["models"]; L = []
    s = main["dataset"]
    L += ["## Table 1. Evaluation set (BDD100K val; pedestrians >= %d px)" % main["min_h"], "",
          "| Condition | Images | Images with >=1 pedestrian | Pedestrian boxes | Boxes per image (with ped.) | Mean image luma |",
          "|---|---|---|---|---|---|"]
    for c in CONDS:
        x = s[c]
        L.append(f"| {c} | {x['images']} | {x['images_with_pedestrian']} | {x['pedestrian_boxes']} | "
                 f"{x['boxes_per_image_with_ped']:.2f} | {x['mean_image_luma']:.1f} |")
    L += ["", "## Table 2. AP50 and AP (%), value [95% bootstrap CI]; delta in percentage points", "",
          "| Model | AP50 day | AP50 night | dAP50 (pp) | rel. drop | AP day | AP night | dAP (pp) |", "|---|---|---|---|---|---|---|---|"]
    for m in M:
        r = main["models"][m]
        L.append(f"| {m} | {fc(r['daytime']['AP50'])} | {fc(r['night']['AP50'])} | {fc(r['delta_AP50'])} | "
                 f"{rel(r['delta_AP50']['relative_pct'])} | {fc(r['daytime']['AP'])} | {fc(r['night']['AP'])} | {fc(r['delta_AP'])} |")
    L += ["", "## Table 3. Operating point (score >= 0.25, IoU 0.5), %", "",
          "| Model | Precision day | Precision night | Recall day | Recall night | Miss rate day | Miss rate night | dMiss (pp) |",
          "|---|---|---|---|---|---|---|---|"]
    for m in M:
        r = main["models"][m]
        L.append(f"| {m} | {fc(r['daytime']['precision'])} | {fc(r['night']['precision'])} | {fc(r['daytime']['recall'])} | "
                 f"{fc(r['night']['recall'])} | {fc(r['daytime']['miss_rate'])} | {fc(r['night']['miss_rate'])} | "
                 f"{f(-r['delta_miss_rate']['value'])} [{f(-r['delta_miss_rate']['ci95'][1])}, {f(-r['delta_miss_rate']['ci95'][0])}] |")
    lat = meta.get("latency", {})
    devs = sorted({d for m in M for d in lat.get(m, {})}, key=lambda d: d == "cpu")
    L += ["", "## Table 4. Model size and single-image latency (batch 1, night images)", ""]
    hdr = "| Model | Params (M) |" + "".join(f" {d.upper()} mean (ms) | {d.upper()} p95 (ms) | {d.upper()} FPS |" for d in devs)
    L += [hdr, "|" + "---|" * (2 + 3 * len(devs))]
    for m in M:
        row = f"| {m} | {meta['params_M'][m]:.2f} |"
        for d in devs:
            x = lat.get(m, {}).get(d)
            row += f" {x['mean_ms']:.1f} | {x['p95_ms']:.1f} | {x['fps']:.1f} |" if x else " - | - | - |"
        L.append(row)
    L += ["", "## Table 5. Recall (%) at the operating point by mean luma of the pedestrian box (n = boxes)", "",
          "| Model | Condition | " + " | ".join(f"Y {lo}-{hi - 1}" for lo, hi in zip(LUM_EDGES[:-1], LUM_EDGES[1:])) + " |",
          "|---|---|" + "---|" * (len(LUM_EDGES) - 1)]
    for m in M:
        for c in CONDS:
            L.append(f"| {m} | {c} | " + " | ".join(
                (f"{f(r['recall'])} (n={r['n']})" if r["n"] else "-") for r in main["by_luminance"][m][c]) + " |")
    L += ["", "## Table 6. Sensitivity: minimum pedestrian height", "",
          f"| Model | dAP50 (pp), >= {main['min_h']} px | dAP50 (pp), >= {sens['min_h']} px | Rank day | Rank night | Rank night (>= {sens['min_h']} px) |",
          "|---|---|---|---|---|---|"]
    for m in M:
        rk = lambda r, c: r["ranking"][c]["order_by_AP50"].index(m) + 1
        L.append(f"| {m} | {fc(main['models'][m]['delta_AP50'])} | {fc(sens['models'][m]['delta_AP50'])} | "
                 f"{rk(main, 'daytime')} | {rk(main, 'night')} | {rk(sens, 'night')} |")
    L += ["", "Pairwise AP50 differences (pp) [95% CI], resolved = CI excludes 0:", ""]
    for c in CONDS:
        for k, v in main["ranking"][c]["pairwise_AP50"].items():
            L.append(f"- {c}: {k}: {f(v['diff'])} [{f(v['ci95'][0])}, {f(v['ci95'][1])}] {'resolved' if v['resolved'] else 'unresolved'}")
    L += ["", "## Table A2. Recall (%) at the operating point by pedestrian height", "",
          "| Model | Condition | 25-49 px | 50-99 px | >= 100 px |", "|---|---|---|---|---|"]
    for m in M:
        for c in CONDS:
            L.append(f"| {m} | {c} | " + " | ".join(
                (f"{f(r['recall'])} (n={r['n']})" if r["n"] else "-") for r in main["by_height"][m][c]) + " |")
    return "\n".join(L) + "\n"


MARKERS = ["o", "s", "^"]


def figures(meta, main, out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888888", "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})
    M = meta["models"]; lat = meta.get("latency", {})
    devs = [d for d in ("cuda", "mps", "cpu") if any(d in lat.get(m, {}) for m in M)]
    if devs:
        fig, axes = plt.subplots(1, len(devs), figsize=(3.3 * len(devs) + 0.4, 3.0), squeeze=False)
        for ax, d in zip(axes[0], devs):
            xmax = 0
            for k, m in enumerate(M):
                if d not in lat.get(m, {}):
                    continue
                x = lat[m][d]["mean_ms"]; r = main["models"][m]["night"]["AP50"]; xmax = max(xmax, x)
                y = 100 * r["value"]; lo, hi = 100 * r["ci95"][0], 100 * r["ci95"][1]
                ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt=MARKERS[k], ms=7, color=COLORS[k], ecolor=COLORS[k],
                            elinewidth=1.5, capsize=3, mec="white", mew=1.2, label=m)
            ax.set_xlabel(f"Mean latency, {d.upper()} (ms, batch 1)"); ax.set_ylabel("Night AP50 (%), 95% CI")
            ax.set_xlim(0, xmax * 1.25 if xmax else 1)
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=len(M), frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(os.path.join(out_dir, "fig1_ap50_vs_latency.png"), dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    centers = [(lo + min(hi, 255)) / 2 for lo, hi in zip(LUM_EDGES[:-1], LUM_EDGES[1:])]
    for k, m in enumerate(M):
        for c, ls, fill in (("daytime", "-", COLORS[k]), ("night", "--", "white")):
            rows = main["by_luminance"][m][c]
            pts = [(cx, 100 * r["recall"]) for cx, r in zip(centers, rows) if r["n"] >= 30]
            if pts:
                ax.plot(*zip(*pts), ls, marker=MARKERS[k], ms=6, lw=2, color=COLORS[k], mfc=fill, mew=1.5,
                        label=f"{m}, {c}")
    for e in LUM_EDGES[1:-1]:
        ax.axvline(e, color="#cccccc", lw=0.6, zorder=0)
    ax.set_xlim(0, 255); ax.set_ylim(0, 100)
    ax.set_xlabel("Mean luma of the pedestrian box (Y, 0-255); bins with >= 30 boxes")
    ax.set_ylabel("Recall at score >= 0.25 (%)")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.86)); fig.savefig(os.path.join(out_dir, "fig2_recall_vs_luminance.png"), dpi=200); plt.close(fig)


def analysis_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--min_h", type=int, default=25)
    ap.add_argument("--min_h_sens", type=int, default=50)
    ap.add_argument("--B", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    meta, z = load(a.results)
    print(f"Analysing {len(z['img_cond'])} images, models {meta['models']}, B={a.B} ...")
    main_r, gt_rec = analyse(meta, z, a.min_h, a.B, a.seed)
    sens_r, _ = analyse(meta, z, a.min_h_sens, a.B, a.seed + 1, full=False)
    summary = {"settings": {**meta["settings"], "analysis": vars(a), "conf_op": CONF_OP, "max_dets": MAX_DETS,
                            "height_margin": HEIGHT_MARGIN, "lum_edges": LUM_EDGES},
               "env": meta["env"], "params_M": meta["params_M"], "latency": meta.get("latency", {}),
               "main": main_r, "sensitivity": sens_r}
    json.dump(summary, open(os.path.join(a.results, "summary.json"), "w"), indent=2)
    # failure-case candidates for Appendix D: night pedestrians >= 50 px missed by every model
    missed = None
    for m in meta["models"]:
        g = gt_rec[m]
        s = {int(r[5]) for r in g if r[0] == 1 and r[2] >= 50 and not r[3]}
        missed = s if missed is None else missed & s
    names = meta["image_names"]
    cand = sorted({int(z["gt"][j, 0]) for j in (missed or set())})[:20]
    json.dump([names[i] for i in cand], open(os.path.join(a.results, "failure_candidates.json"), "w"), indent=1)
    md = tables_md(meta, main_r, sens_r)
    open(os.path.join(a.results, "tables.md"), "w").write(md)
    figures(meta, main_r, a.results)
    print(md)
    print("Saved summary.json, tables.md, figures and failure_candidates.json in", a.results)


CAT = {"pedestrian": 0, "person": 0, "rider": 1, "other person": 2}   # 0 = evaluated, 1-2 = ignore regions
MODELS = ["YOLOv8n", "YOLO11n", "SSDLite320-MobileNetV3"]
SCORE_MIN = 0.001
MAX_SAVE = 300


# ----------------------------------------------------------------------------- environment
def hardware(device):
    env = {"os": platform.platform(), "python": platform.python_version(), "machine": platform.machine(),
           "torch": torch.__version__, "torchvision": torchvision.__version__, "opencv": cv2.__version__,
           "numpy": np.__version__, "device_used": device, "torch_cpu_threads": torch.get_num_threads(),
           "logical_cpus": os.cpu_count()}
    try:
        import ultralytics; env["ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    cpu = platform.processor()
    try:
        if sys.platform.startswith("linux"):
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip(); break
        elif sys.platform == "darwin":
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        elif sys.platform == "win32":
            out = subprocess.check_output(["powershell", "-NoProfile", "-Command",
                                           "(Get-CimInstance Win32_Processor).Name"], text=True).strip()
            cpu = out or cpu
    except Exception:
        pass
    env["cpu"] = cpu
    try:
        import psutil
        env["ram_GB"] = round(psutil.virtual_memory().total / 2**30, 1)
        env["physical_cpus"] = psutil.cpu_count(logical=False)
    except Exception:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update({"gpu": p.name, "gpu_memory_GB": round(p.total_memory / 2**30, 1), "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version()})
    return env


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# ----------------------------------------------------------------------------- data
def load_labels(path, index, n, seed):
    data = json.load(open(path))
    if isinstance(data, dict):                     # some exports wrap frames in {"frames": [...]}
        data = data.get("frames", data.get("images", []))
    items = {c: [] for c in CONDS}
    missing = 0
    for d in data:
        tod = (d.get("attributes") or {}).get("timeofday")
        if tod not in items:
            continue
        if d["name"] not in index:
            missing += 1; continue
        boxes = []
        for l in d.get("labels") or []:
            c = CAT.get(l.get("category"))
            b = l.get("box2d")
            if c is None or not b:
                continue
            boxes.append([b["x1"], b["y1"], b["x2"], b["y2"], c])
        items[tod].append((d["name"], boxes))
    print(f"labelled daytime/night images found: { {c: len(v) for c, v in items.items()} }, missing image files: {missing}")
    if n:
        rnd = random.Random(seed)
        for c in CONDS:
            rnd.shuffle(items[c]); items[c] = items[c][:n]
    return items


def find_dataset(root, labels=None):
    """Locate the BDD100K val label JSON and index every .jpg under root by file name."""
    if labels is None:
        found = {}
        for dp, dn, fn in os.walk(root):
            for f in fn:
                if f in ("det_val.json", "bdd100k_labels_images_val.json"):
                    found.setdefault(f, os.path.join(dp, f))
        labels = found.get("det_val.json") or found.get("bdd100k_labels_images_val.json")
        if labels is None:
            sys.exit(f"Could not find det_val.json or bdd100k_labels_images_val.json under {root}. "
                     "Pass --labels explicitly.")
    print("Labels:", labels)
    index = {}
    val_dirs = [dp for dp, dn, fn in os.walk(root) if os.path.basename(dp) == "val" and any(f.endswith(".jpg") for f in fn)]
    for d in val_dirs:                              # official val images first
        for f in os.listdir(d):
            if f.endswith(".jpg"):
                index.setdefault(f, os.path.join(d, f))
    names = {d["name"] for d in json.load(open(labels)) if isinstance(d, dict) and "name" in d}
    if len(names & index.keys()) < 0.9 * len(names):  # unusual layout: index every jpg
        for dp, dn, fn in os.walk(root):
            for f in fn:
                if f in names:
                    index.setdefault(f, os.path.join(dp, f))
    print(f"Image folders used: {val_dirs or 'full scan'}; {len(names & index.keys())} of {len(names)} labelled val images found")
    return labels, index


# ----------------------------------------------------------------------------- models
def make_predictors(device):
    from ultralytics import YOLO
    preds, params = {}, {}
    for name, w in [("YOLOv8n", "yolov8n.pt"), ("YOLO11n", "yolo11n.pt")]:
        m = YOLO(w)
        params[name] = sum(p.numel() for p in m.model.parameters()) / 1e6
        def f(img_bgr, m=m):
            r = m.predict(img_bgr, conf=SCORE_MIN, classes=[0], imgsz=640, max_det=MAX_SAVE,
                          device=device, verbose=False)[0]
            return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        preds[name] = f
    ssd = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights="DEFAULT")
    ssd.score_thresh = SCORE_MIN
    ssd.detections_per_img = MAX_SAVE
    params["SSDLite320-MobileNetV3"] = sum(p.numel() for p in ssd.parameters()) / 1e6
    ssd.eval().to(device)
    def g(img_bgr):
        x = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)
        with torch.no_grad():
            o = ssd([x.to(device)])[0]
        keep = o["labels"] == 1                   # COCO "person" in torchvision's label map
        return o["boxes"][keep].cpu().numpy(), o["scores"][keep].cpu().numpy()
    preds["SSDLite320-MobileNetV3"] = g
    info = {"yolo_imgsz": 640, "yolo_nms_iou": 0.7, "ssd_input": "320x320 (resized, no letterbox)",
            "ssd_nms_iou": float(ssd.nms_thresh), "max_saved_dets": MAX_SAVE, "score_min": SCORE_MIN}
    return preds, params, info


def latency(pred, imgs, device, runs, warm):
    for i in range(warm):
        pred(imgs[i % len(imgs)])
    ts = []
    for i in range(runs):
        sync(device); t = time.perf_counter()
        pred(imgs[i % len(imgs)])
        sync(device); ts.append((time.perf_counter() - t) * 1000)
    return {"mean_ms": float(np.mean(ts)), "median_ms": float(np.median(ts)),
            "p95_ms": float(np.percentile(ts, 95)), "fps": 1000 / float(np.mean(ts)), "runs": runs, "warmup": warm}


def box_luma(y, b):
    h, w = y.shape
    x1, y1 = max(int(b[0]), 0), max(int(b[1]), 0)
    x2, y2 = min(int(np.ceil(b[2])), w), min(int(np.ceil(b[3])), h)
    return float(y[y1:y2, x1:x2].mean()) if x2 > x1 and y2 > y1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/kaggle/input" if os.path.isdir("/kaggle/input") else ".",
                    help="folder that contains the (unzipped) BDD100K data; searched recursively")
    ap.add_argument("--labels", default=None, help="optional explicit path to the val label JSON")
    ap.add_argument("--out", default="/kaggle/working/results" if os.path.isdir("/kaggle/working") else "results")
    ap.add_argument("--n", type=int, default=0, help="images per condition (0 = all; use small n only for testing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available)")
    ap.add_argument("--B", type=int, default=1000, help="bootstrap resamples for analyze.py")
    ap.add_argument("--skip_latency", action="store_true")
    a, _ = ap.parse_known_args()   # tolerate notebook kernel arguments
    os.makedirs(a.out, exist_ok=True)
    dev = a.device or ("cuda" if torch.cuda.is_available() else
                       "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    env = hardware(dev)
    print("ENVIRONMENT:", json.dumps(env, indent=1))

    labels, index = find_dataset(a.root, a.labels)
    items = load_labels(labels, index, a.n, a.seed)
    print({c: len(v) for c, v in items.items()}, "images per condition")
    order = [(c, name, boxes) for c in CONDS for name, boxes in items[c]]
    preds, params, info = make_predictors(dev)

    img_cond, img_lum, names, gts, dets = [], [], [], [], {m: [] for m in preds}
    t0 = time.time()
    for i, (c, name, boxes) in enumerate(order):
        img = cv2.imread(index[name])
        y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        img_cond.append(CONDS.index(c)); img_lum.append(float(y.mean())); names.append(name)
        for b in boxes:
            gts.append([i, *b[:4], b[4], box_luma(y, b)])
        for m, pred in preds.items():
            bx, sc = pred(img)
            o = np.argsort(-sc)[:MAX_SAVE]
            if len(o):
                dets[m].append(np.column_stack([np.full(len(o), i), bx[o], sc[o]]))
        if (i + 1) % 200 == 0 or i + 1 == len(order):
            el = time.time() - t0
            print(f"  {i + 1}/{len(order)} images, {el / 60:.1f} min elapsed, ~{el / (i + 1) * (len(order) - i - 1) / 60:.1f} min left",
                  flush=True)
    arrays = {"img_cond": np.array(img_cond, np.int8), "img_lum": np.array(img_lum, np.float32),
              "gt": np.array(gts, np.float32).reshape(-1, 7)}
    for m in preds:
        arrays[f"det_{m}"] = np.concatenate(dets[m]).astype(np.float32) if dets[m] else np.zeros((0, 6), np.float32)
    np.savez_compressed(os.path.join(a.out, "raw_detections.npz"), **arrays)

    lat = {m: {} for m in preds}
    if not a.skip_latency:
        night = [n for c, n, _ in order if c == "night"][:100] or names[:100]
        imgs = [cv2.imread(index[n]) for n in night]
        print("Measuring latency on", dev, "...")
        for m, pred in preds.items():
            lat[m][dev] = latency(pred, imgs, dev, runs=100, warm=20)
        if dev != "cpu":                            # CPU-only latency as a weaker-hardware proxy
            print("Measuring latency on cpu ...")
            cpu_preds, _, _ = make_predictors("cpu")
            for m, pred in cpu_preds.items():
                lat[m]["cpu"] = latency(pred, imgs, "cpu", runs=50, warm=10)
        print(json.dumps(lat, indent=1))

    meta = {"settings": {k: v for k, v in vars(a).items()}, "env": env, "params_M": params, "models": list(preds),
            "latency": lat, "image_names": names, "categories": CAT, "inference": info,
            "inference_minutes": (time.time() - t0) / 60}
    json.dump(meta, open(os.path.join(a.out, "run_meta.json"), "w"), indent=1)
    print("Saved raw detections and run_meta.json in", a.out)

    analysis_main(["--results", a.out, "--B", str(a.B)])
    draw_failures(index, a.out)
    zip_path = shutil.make_archive(a.out, "zip", a.out)
    print("\nALL DONE. Results archive:", zip_path)
    print("""
NOTE FOR LIMITATIONS SECTION: all numbers come from offline inference on a desktop/cloud
machine, not an in-vehicle computer. Latency covers model inference including image
preprocessing and NMS, not image decoding, camera capture, sensor fusion, or actuation.
BDD100K is an offline dataset, not a closed-loop driving test. Results are relative model comparisons.""")


def draw_failures(index, out, k=6):
    """Appendix D: night images where every model misses a >= 50 px pedestrian at score >= 0.25."""
    try:
        cand = json.load(open(os.path.join(out, "failure_candidates.json")))
    except Exception:
        return
    meta = json.load(open(os.path.join(out, "run_meta.json")))
    z = np.load(os.path.join(out, "raw_detections.npz"))
    os.makedirs(os.path.join(out, "failures"), exist_ok=True)
    colors = {"YOLOv8n": (214, 120, 42), "YOLO11n": (52, 104, 235), "SSDLite320-MobileNetV3": (122, 175, 27)}
    for name in cand[:k]:
        i = meta["image_names"].index(name)
        img = cv2.imread(index[name])
        for g in z["gt"][z["gt"][:, 0] == i]:
            if g[5] == 0:
                cv2.rectangle(img, (int(g[1]), int(g[2])), (int(g[3]), int(g[4])), (255, 255, 255), 2)
        for m in meta["models"]:
            d = z[f"det_{m}"]; d = d[(d[:, 0] == i) & (d[:, 5] >= 0.25)]
            for b in d:
                cv2.rectangle(img, (int(b[1]), int(b[2])), (int(b[3]), int(b[4])), colors.get(m, (0, 255, 255)), 2)
        cv2.imwrite(os.path.join(out, "failures", name), img)


if __name__ == "__main__":
    main()
