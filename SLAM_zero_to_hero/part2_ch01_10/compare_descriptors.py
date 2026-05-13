"""Run every torch-based descriptor on GardensPoint and plot them together."""
import configparser
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/VPR_Tutorial")
from datasets.load_dataset import GardensPointDataset  # noqa: E402
from evaluation.metrics import createPR, recallAt100precision, recallAtK  # noqa: E402

DESCRIPTORS = ["AlexNet", "NetVLAD", "PatchNetVLAD", "CosPlace", "EigenPlaces", "SAD"]


def make_extractor(name):
    if name == "AlexNet":
        from feature_extraction.feature_extractor_holistic import AlexNetConv3Extractor
        return AlexNetConv3Extractor()
    if name == "SAD":
        from feature_extraction.feature_extractor_holistic import SAD
        return SAD()
    if name in ("NetVLAD", "PatchNetVLAD"):
        from feature_extraction.feature_extractor_patchnetvlad import PatchNetVLADFeatureExtractor
        from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR
        ini = "configs/netvlad_extract.ini" if name == "NetVLAD" else "configs/speed.ini"
        cfg = configparser.ConfigParser()
        cfg.read(os.path.join(PATCHNETVLAD_ROOT_DIR, ini))
        return PatchNetVLADFeatureExtractor(cfg)
    if name == "CosPlace":
        from feature_extraction.feature_extractor_cosplace import CosPlaceFeatureExtractor
        return CosPlaceFeatureExtractor()
    if name == "EigenPlaces":
        from feature_extraction.feature_extractor_eigenplaces import EigenPlacesFeatureExtractor
        return EigenPlacesFeatureExtractor()
    raise ValueError(name)


def compute_S(name, extractor, imgs_db, imgs_q):
    if name == "PatchNetVLAD":
        _, db_patches = extractor.compute_features(imgs_db)
        _, q_patches = extractor.compute_features(imgs_q)
        return extractor.local_matcher_from_numpy_single_scale(q_patches, db_patches)
    db = extractor.compute_features(imgs_db)
    q = extractor.compute_features(imgs_q)
    if name == "SAD":
        S = np.empty([len(imgs_db), len(imgs_q)], "float32")
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                diff = db[i] - q[j]
                dim = len(db[0]) - np.sum(np.isnan(diff))
                diff[np.isnan(diff)] = 0
                S[i, j] = -np.sum(np.abs(diff)) / dim
        return S
    db = db / np.linalg.norm(db, axis=1, keepdims=True)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return db @ q.T


print("===== Loading GardensPoint")
imgs_db, imgs_q, GThard, GTsoft = GardensPointDataset().load()

results = {}
for name in DESCRIPTORS:
    print(f"\n===== {name}")
    t0 = time.time()
    try:
        S = compute_S(name, make_extractor(name), imgs_db, imgs_q)
    except Exception as e:
        print(f"  FAILED: {e!r}")
        continue
    P, R = createPR(S, GThard, GTsoft, matching="multi", n_thresh=100)
    auc = float(np.trapz(P, R))
    r100p = float(recallAt100precision(S, GThard, GTsoft, matching="multi", n_thresh=100))
    rk = {k: float(recallAtK(S, GThard, K=k)) for k in (1, 5, 10)}
    results[name] = dict(S=S, P=P, R=R, auc=auc, r100p=r100p, rk=rk, secs=time.time() - t0)
    print(f"  AUC={auc:.3f} R@100P={r100p:.2f} R@1={rk[1]:.3f} ({results[name]['secs']:.1f}s)")

print("\n===== Summary")
print(f"{'descriptor':<14} {'AUC':>6} {'R@100P':>7} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'sec':>6}")
for name, r in results.items():
    print(f"{name:<14} {r['auc']:>6.3f} {r['r100p']:>7.2f} {r['rk'][1]:>6.3f} {r['rk'][5]:>6.3f} {r['rk'][10]:>6.3f} {r['secs']:>6.1f}")

n = len(results)
example_queries = [10, 35]
n_cols = n + 1

fig = plt.figure(figsize=(18, 13), constrained_layout=True)
fig.canvas.manager.set_window_title("VPR descriptor comparison - GardensPoint day_right vs night_right")
gs = fig.add_gridspec(2 + len(example_queries), n_cols, height_ratios=[1.1, 1.2] + [1.1] * len(example_queries))

split = max(1, n_cols // 2)
ax_pr = fig.add_subplot(gs[0, :split])
for name, r in results.items():
    ax_pr.plot(r["R"], r["P"], label=f"{name} (AUC={r['auc']:.2f})")
ax_pr.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision", title="Precision-Recall")
ax_pr.grid(True)
ax_pr.legend(loc="lower left", fontsize=9)

ax_tbl = fig.add_subplot(gs[0, split:])
ax_tbl.axis("off")
header = ["descriptor", "AUC", "R@100P", "R@1", "R@5", "R@10", "sec"]
rows = [[name, f"{r['auc']:.3f}", f"{r['r100p']:.2f}",
         f"{r['rk'][1]:.3f}", f"{r['rk'][5]:.3f}", f"{r['rk'][10]:.3f}",
         f"{r['secs']:.1f}"] for name, r in results.items()]
tbl = ax_tbl.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.4)
ax_tbl.set_title("Metrics")

for i, (name, r) in enumerate(results.items()):
    ax = fig.add_subplot(gs[1, i + 1])
    ax.imshow(r["S"])
    ax.set_title(f"S - {name}", fontsize=10)
    ax.axis("off")
ax_lbl = fig.add_subplot(gs[1, 0])
ax_lbl.axis("off")
ax_lbl.text(0.5, 0.5, "Similarity\nmatrices", ha="center", va="center", fontsize=11, fontweight="bold")

for row_i, q_idx in enumerate(example_queries):
    ax_q = fig.add_subplot(gs[2 + row_i, 0])
    ax_q.imshow(imgs_q[q_idx])
    ax_q.set_title(f"Query #{q_idx} (night)", fontsize=10)
    ax_q.axis("off")
    for col_i, (name, r) in enumerate(results.items()):
        S = r["S"]
        pred = int(np.argmax(S[:, q_idx]))
        correct = bool(GTsoft[pred, q_idx])
        ax = fig.add_subplot(gs[2 + row_i, col_i + 1])
        ax.imshow(imgs_db[pred])
        ax.set_title(f"{name} -> db #{pred}", fontsize=10, color="green" if correct else "red")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("green" if correct else "red")
            spine.set_linewidth(3)

out = "/out/all_descriptors_comparison.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"\nSaved combined figure to {out}")
plt.show()
