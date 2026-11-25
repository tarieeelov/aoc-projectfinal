"""Cross-check analyze.py's matching + AP against pycocotools on random synthetic data,
with ignore regions passed to pycocotools as iscrowd=1 (same semantics)."""
import sys, io, contextlib
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import analyze as A
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

rng = np.random.default_rng(1)
images, anns, dts = [], [], []
per = []
aid = 1
for i in range(300):
    images.append({"id": i, "width": 1280, "height": 720})
    G = rng.integers(0, 6); K = rng.integers(0, 3)
    def rbox():
        x, y = rng.uniform(0, 1100), rng.uniform(0, 550); w, h = rng.uniform(15, 150), rng.uniform(30, 170)
        return [x, y, x + w, y + h]
    gv = np.array([rbox() for _ in range(G)]).reshape(-1, 4)
    gi = np.array([rbox() for _ in range(K)]).reshape(-1, 4)
    dl = []
    for g in np.vstack([gv, gi]):          # noisy copies of GTs + random boxes
        for _ in range(rng.integers(0, 3)):
            j = rng.normal(0, 8, 4); dl.append([*(g + j), rng.uniform()])
    for _ in range(rng.integers(0, 8)):
        dl.append([*rbox(), rng.uniform()])
    dets = np.array(dl).reshape(-1, 5)
    dets[:, 2:4] = np.maximum(dets[:, 2:4], dets[:, 0:2] + 2)
    for g, crowd in [(g, 0) for g in gv] + [(g, 1) for g in gi]:
        w, h = g[2] - g[0], g[3] - g[1]
        anns.append({"id": aid, "image_id": i, "category_id": 1, "bbox": [g[0], g[1], w, h], "area": w * h, "iscrowd": crowd}); aid += 1
    for d in dets:
        dts.append({"image_id": i, "category_id": 1, "bbox": [d[0], d[1], d[2] - d[0], d[3] - d[1]], "score": float(d[4])})
    s, tp, ig, _ = A.match_image(dets, gv, gi)
    per.append((s, tp, ig, len(gv)))

gt = COCO(); gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "p"}]}
with contextlib.redirect_stdout(io.StringIO()):
    gt.createIndex(); dt = gt.loadRes(dts)
    E = COCOeval(gt, dt, "bbox"); E.params.areaRng = [[0, 1e10]]; E.params.areaRngLbl = ["all"]
    E.evaluate(); E.accumulate()
p = E.eval["precision"][:, :, 0, 0, -1]            # [T, R] at maxDets=100
coco_ap = p.mean(1)
ev = A.CondEval(per)
mine = [A.ap_from_sorted(ev.tp[t], ev.fp[t], ev.ngt.sum()) for t in range(10)]
print("pycocotools AP per IoU:", np.round(coco_ap, 4))
print("analyze.py  AP per IoU:", np.round(mine, 4))
print("max abs diff:", np.max(np.abs(coco_ap - np.array(mine))))
m = ev.metrics()
print("AP50 %.4f  AP %.4f  (coco: %.4f %.4f)" % (m["AP50"], m["AP"], coco_ap[0], coco_ap.mean()))
# bootstrap weights of all ones must reproduce point estimate; integer weights == duplicated images
w = np.ones(ev.N); assert abs(ev.metrics(w)["AP"] - m["AP"]) < 1e-12
w2 = np.zeros(ev.N); w2[:150] = 2
dup = A.CondEval([per[i] for i in range(150) for _ in range(2)])
print("weighted vs duplicated AP50:", ev.metrics(w2)["AP50"], dup.metrics()["AP50"])
assert abs(ev.metrics(w2)["AP50"] - dup.metrics()["AP50"]) < 1e-9
assert np.max(np.abs(coco_ap - np.array(mine))) < 1e-6
print("OK")
