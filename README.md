# Visual Change Inspector

An alignment-aware computer vision project for comparing a reference image with a later capture. It estimates camera motion, highlights changed areas, and returns **REVIEW** when alignment evidence is insufficient. A self-contained interactive report makes the result inspectable.

**Controlled benchmark result:** changed-pixel IoU improved from **4.8% without alignment to 94.6% with alignment**. All 24 held-out pairs aligned; the six no-edit pairs produced zero false alarms after alignment. These are synthetic edits on only two held-out photographs, not evidence of production defect-detection performance.

## Explore the finished report

Download this repository (Code → Download ZIP), extract it, and open [`reports/benchmark/report.html`](reports/benchmark/report.html) in a browser. The report works offline and contains ten image comparisons, original/aligned/detection/ground-truth views, a reveal slider, the full 24-pair audit, and downloadable metrics. GitHub's file viewer displays HTML source; download the file to use it.

## Run locally

Python 3.11 or newer is required. Python 3.12 was used for the published results.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux alternative: source .venv/bin/activate
python -m pip install -e ".[benchmark]"
python -m unittest discover -s tests -v
change-inspector benchmark --out reports/benchmark
```

For the package versions used in the published run, add `-c constraints-tested.txt` to the installation command. Dependency wheels and image caches may require network access on first use; a generated report needs none.

## Compare your own images

```bash
change-inspector compare private-inputs/before.jpg private-inputs/after.jpg --out reports/custom
# Optional adjustments, in processed-image units:
change-inspector compare before.png after.png --threshold 36 --min-area 50 --out reports/custom
```

Use two RGB-compatible photographs with equal dimensions after EXIF orientation correction. The CLI resizes both together to a maximum side of 720 pixels. Masks and box coordinates refer to that resized reference image, not the original resolution. The Python API accepts RGB uint8 arrays from 64 to 2048 pixels per side with equal shapes.

Outputs: `report.html`, `metrics.json`, and, when alignment passes, `aligned.png`, `mask.png`, and `valid-area.png`. Report images use JPEG compression for display; exported masks are lossless. The custom report embeds your image pixels. Processing stays local, but sharing that report shares the images. `private-inputs/` and `reports/custom/` are ignored by Git.

| Status | Meaning | CLI exit code |
| --- | --- | --- |
| `CHANGES` | At least one retained changed region within the inspected overlap | 0 |
| `NO_CHANGE` | No region passed the configured thresholds in the inspected overlap | 0 |
| `REVIEW` | Alignment failed a quality gate; no change decision was made | 2 |
| Input error | Missing/incompatible image or invalid configuration | 1 |

Threshold 36 was selected on calibration data. Increasing it reduces sensitivity to low-contrast edits. `min-area` removes connected components smaller than the given processed-pixel area. Neither setting is a probability or confidence score. Reusing an output directory replaces its report and removes stale mask/alignment exports when a new comparison requires review.

## How it works

1. Extract SIFT features and keep distinct matches using a 0.72 nearest-neighbor ratio test.
2. Estimate an after-to-before homography with RANSAC (2.5 pixel tolerance).
3. Require at least 12 inliers, 45% inlier agreement, 12% spatial coverage, plausible modest camera movement, and 65% usable overlap.
4. Warp the later image into reference coordinates. Erode the valid area to exclude interpolation and image-border artifacts.
5. Blur each image, compensate a median additive RGB brightness offset, and threshold the maximum absolute channel difference.
6. Apply a 3×3 opening and discard regions under 50 pixels. Export regions, alignment diagnostics, and masks.

The geometric gate supports modest movement, not arbitrary camera views: area ratio between 0.5 and 2, convex non-reflected transformed frame, bounded projective denominator, and maximum corner displacement below 40% of the frame diagonal. A minimum convex-hull feature coverage is required in both images.

## Evaluation protocol

Source photographs are center-cropped/resized to 384×256. The generator inserts one or two solid colored rectangular patches for changed cases, then applies rotation (±4°), scale (0.97–1.03), translation (±12 pixels), a small projective component, and an additive brightness shift (±14). Every fourth seed has no content change. It does not simulate object removal, realistic damage, or shadows.

- **Calibration:** coffee and brick; seeds 100–107 per scene; 16 pairs. Select the threshold with highest pooled IoU from 18, 24, 30, 36. Other parameters are fixed engineering choices.
- **Held-out evaluation:** Chelsea and astronaut; seeds 200–211 per scene; 24 pairs, including six no-edit pairs. No source photograph appears in both partitions. Seeds repeat between evaluation scenes, so the 24 pairs are not independent samples.
- **Failure controls:** one flat pair and one unrelated-scene pair, both correctly sent to review. These two controls are illustrative, not a calibrated estimate of rejection reliability.
- **Baseline:** the same brightness correction, threshold, and region filter, but no image alignment. Score both methods on the same overlap accepted by the alignment method. This is an ablation of alignment, not a comparison with another trained system.
- **Metrics:** pixel counts are pooled over accepted shared areas. Undefined ratios are `null`, not a perfect score. Full-frame recall counts changed pixels outside accepted overlap, including all changes in rejected pairs, as missed. Rejections are also reported separately. In this run all edited pixels were visible and all test pairs were accepted.

| Metric | Without alignment | With alignment |
| --- | ---: | ---: |
| Changed-pixel IoU | 4.79% | 94.61% |
| Changed-pixel precision | 4.85% | 94.83% |
| Changed-pixel recall, accepted overlap | 79.68% | 99.76% |
| No-edit pairs with false alarms | 6 / 6 | 0 / 6 |

Mean corner registration error against the known synthetic transformation was **0.093 pixels**. Full-frame changed-pixel recall was **99.76%**. Exact pair-level results, counts, calibration candidates, homographies, and runtime versions are in [`metrics.json`](reports/benchmark/metrics.json).

## Limitations

This is a small, deliberately controlled engineering demonstration. High-contrast patches are much easier than realistic defects. Two held-out photographs are too few for strong generalization claims. Parameters were not validated on a real inspection dataset, and no confidence intervals are claimed.

A single planar transformation cannot model parallax, multiple depths, or non-rigid motion. Shadows, reflections, exposure clipping, textureless regions, repeated patterns, large motion, or changes occupying much of the image can cause false alarms, missed edits, or misleading accepted alignments. Median brightness correction assumes most of the shared scene is unchanged. There is no semantic understanding of whether a detected difference matters. Unseen borders are not inspected. The method requires human review for real-world use.

## Verification and repository layout

The 21 automated tests exercise known geometric transforms, brightness nuisance correction, patch localization, insufficient texture and unrelated-pair rejection, invalid inputs, overlap masking, component filtering, reproducibility, report escaping, resizing, and CLI outputs/exit codes. Tests use generated fixtures and need no downloaded dataset. A GitHub Actions workflow runs them on Python 3.12.

```text
change_inspector/core.py        Alignment, quality gates, difference masks
change_inspector/benchmark.py   Controlled generator and calibration/evaluation
change_inspector/cli.py         Local image-pair command line
change_inspector/report.py      Self-contained report generation
change_inspector/template.html  Interactive comparison interface
tests/                         Deterministic offline tests
reports/benchmark/             Published report and complete metrics
```

See [VALIDATION.md](VALIDATION.md) for the observed local and hosted check status. Code is MIT; sample image attribution and source links are in [ATTRIBUTION.md](ATTRIBUTION.md).
