"""Controlled image edits, not a real-world defect-detection dataset."""
import argparse
from dataclasses import asdict
import platform
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageOps
from .core import Config, inspect, difference, regions_from_score
from .report import example, write_report


def load_scenes():
    from skimage import data
    scenes = {}
    for name in ('coffee', 'brick', 'chelsea', 'astronaut'):
        a = getattr(data, name)()
        im = Image.fromarray(a).convert('RGB')
        scenes[name] = np.array(ImageOps.fit(im, (384, 256), method=Image.Resampling.LANCZOS))
    return scenes


def make_pair(before, seed, changed=True):
    rng = np.random.default_rng(seed)
    h, w = before.shape[:2]
    edited = before.copy()
    truth = np.zeros((h, w), np.uint8)
    if changed:
        for _ in range(1 + seed % 2):
            x, y = int(rng.integers(35, w-85)), int(rng.integers(35, h-65))
            rw, rh = int(rng.integers(25, 58)), int(rng.integers(20, 44))
            # A colored patch replaces local pixels; all replaced pixels are labelled.
            color = np.array([235, 45, 110]) if seed % 2 else np.array([25, 220, 175])
            edited[y:y+rh, x:x+rw] = color
            truth[y:y+rh, x:x+rw] = 1
    H = np.vstack([cv2.getRotationMatrix2D((w/2, h/2), float(rng.uniform(-4, 4)), float(rng.uniform(.97, 1.03))), [0, 0, 1]])
    H[:2, 2] += rng.uniform(-12, 12, size=2)
    H[2, :2] = rng.uniform(-.00005, .00005, size=2)
    lit = np.clip(edited.astype(np.float32) + rng.uniform(-14, 14), 0, 255).astype(np.uint8)
    after = cv2.warpPerspective(lit, H, (w, h))
    return after, truth.astype(bool), H


def counts(pred, truth, valid):
    return dict(tp=int((pred & truth & valid).sum()), fp=int((pred & ~truth & valid).sum()),
                fn=int((~pred & truth & valid).sum()), tn=int((~pred & ~truth & valid).sum()))


def rates(c):
    def ratio(a, b):
        return a/b if b else None
    return {**c, 'precision': ratio(c['tp'], c['tp']+c['fp']),
            'recall': ratio(c['tp'], c['tp']+c['fn']),
            'iou': ratio(c['tp'], c['tp']+c['fp']+c['fn']),
            'false_positive_rate': ratio(c['fp'], c['fp']+c['tn'])}


def sum_counts(values):
    return {k: sum(v[k] for v in values) for k in ('tp', 'fp', 'fn', 'tn')}


def run(output):
    scenes = load_scenes()
    calibration = []
    for name in ('coffee', 'brick'):
        for seed in range(100, 108):
            after, truth, _ = make_pair(scenes[name], seed, changed=seed % 4 != 0)
            r = inspect(scenes[name], after)
            calibration.append((truth, r))
    candidates = []
    for threshold in (18, 24, 30, 36):
        cs = []
        for truth, r in calibration:
            if r.status == 'REVIEW':
                cs.append(counts(np.zeros_like(truth), truth, np.ones_like(truth)))
            else:
                pred, _ = regions_from_score(r.score, r.valid, threshold, 50)
                cs.append(counts(pred, truth, r.valid))
        candidates.append(dict(threshold=threshold, **rates(sum_counts(cs))))
    selected = max(candidates, key=lambda c: c['iou'] or 0)['threshold']
    config = Config(threshold=selected)
    rows, gallery, aligned_counts, naive_counts = [], [], [], []
    true_total, true_seen = 0, 0
    for name in ('chelsea', 'astronaut'):
        before = scenes[name]
        for seed in range(200, 212):
            changed = seed % 4 != 0
            after, truth, forward = make_pair(before, seed, changed)
            r = inspect(before, after, config)
            row = dict(scene=name, seed=seed, changed=changed, **r.summary())
            true_total += int(truth.sum())
            if r.status != 'REVIEW':
                # Both methods scored on the SAME accepted overlap for a fair comparison.
                c = counts(r.mask, truth, r.valid)
                naive_score, _ = difference(before, after, r.valid)
                naive, _ = regions_from_score(naive_score, r.valid, selected, 50)
                nc = counts(naive, truth, r.valid)
                aligned_counts.append(c)
                naive_counts.append(nc)
                true_seen += int((truth & r.valid).sum())
                corners = np.float32([[[0, 0], [383, 0], [383, 255], [0, 255]]])
                actual = cv2.perspectiveTransform(corners, r.homography)
                expected = cv2.perspectiveTransform(corners, np.linalg.inv(forward))
                row.update(aligned=rates(c), unaligned=rates(nc),
                           corner_error_px=float(np.linalg.norm(actual-expected, axis=2).mean()),
                           naive_alarm=bool(naive.any()))
            rows.append(row)
            if seed in (200, 201, 202, 203):
                gallery.append(example(f'{name.title()} / {seed} / ' + ('added patches' if changed else 'no content change'),
                                       before, after, r, truth, row))
    controls = []
    for name, before, after in (
        ('Flat image / insufficient texture', np.full_like(scenes['coffee'], 125), np.full_like(scenes['coffee'], 135)),
        ('Different scenes / unrelated pair', scenes['chelsea'], scenes['astronaut']),
    ):
        r = inspect(before, after, config)
        controls.append(dict(name=name, rejected=r.status == 'REVIEW', **r.summary()))
        gallery.append(example(name, before, after, r))
    ac = rates(sum_counts(aligned_counts))
    nc = rates(sum_counts(naive_counts))
    accepted = [r for r in rows if r['status'] != 'REVIEW']
    unchanged = [r for r in accepted if not r['changed']]
    summary = dict(pairs=len(rows), accepted=len(accepted), rejected=len(rows)-len(accepted),
                   aligned=ac, unaligned=nc, true_changed_pixels=true_total,
                   visible_true_changed_pixels=true_seen,
                   visible_change_fraction=true_seen/true_total,
                   full_frame_recall=ac['tp']/true_total,
                   mean_corner_error_px=float(np.mean([r['corner_error_px'] for r in accepted])) if accepted else None,
                   unchanged_pairs_accepted=len(unchanged), unchanged_pairs_total=sum(not r['changed'] for r in rows),
                   aligned_unchanged_alarms=sum(r['status'] == 'CHANGES' for r in unchanged),
                   unaligned_unchanged_alarms=sum(r['naive_alarm'] for r in unchanged),
                   controls_rejected=sum(c['rejected'] for c in controls), controls_total=len(controls))
    payload = dict(title='Visual Change Inspector', mode='benchmark', config=asdict(config),
                   summary=summary, calibration=dict(scenes=['coffee', 'brick'], pairs=len(calibration),
                       accepted=sum(r.status != 'REVIEW' for _, r in calibration), candidates=candidates,
                       selection='Maximum pooled calibration IoU; first/lower threshold wins ties.'),
                   test_scenes=['chelsea', 'astronaut'], rows=rows, controls=controls, examples=gallery,
                   versions=dict(python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__))
    write_report(output, payload)
    print(json.dumps({'threshold': selected, **summary}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='reports/benchmark')
    run(p.parse_args().out)
