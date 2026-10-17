"""All masks and detected boxes use BEFORE image coordinates; inputs are RGB."""
from dataclasses import dataclass, field
import cv2
import numpy as np


@dataclass(frozen=True)
class Config:
    threshold: float = 36
    min_area: int = 50
    min_inliers: int = 12
    min_inlier_ratio: float = .45
    min_coverage: float = .12
    min_overlap: float = .65

    def __post_init__(self):
        if not np.isfinite(self.threshold) or not 0 < self.threshold <= 255:
            raise ValueError("threshold must be finite and between 0 and 255")
        if not isinstance(self.min_area, int) or self.min_area < 1:
            raise ValueError("min_area must be a positive integer")
        if not isinstance(self.min_inliers, int) or isinstance(self.min_inliers, bool) or self.min_inliers < 4:
            raise ValueError("min_inliers must be an integer of at least four")
        if not all(np.isfinite(v) and 0 < v <= 1 for v in
                   (self.min_inlier_ratio, self.min_coverage, self.min_overlap)):
            raise ValueError("quality fractions must be in (0, 1]")


@dataclass
class Result:
    status: str
    reason: str
    quality: dict = field(default_factory=dict)
    homography: np.ndarray | None = None
    aligned: np.ndarray | None = None
    valid: np.ndarray | None = None
    score: np.ndarray | None = None
    mask: np.ndarray | None = None
    regions: list = field(default_factory=list)

    def summary(self):
        return {"status": self.status, "reason": self.reason, "quality": self.quality,
                "homography_after_to_before": None if self.homography is None else self.homography.tolist(),
                "regions": self.regions,
                "changed_pixels": None if self.mask is None else int(self.mask.sum())}


def validate_image(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("images must be uint8 numpy arrays")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 64:
        raise ValueError("images must be RGB with both dimensions at least 64")
    if max(image.shape[:2]) > 2048:
        raise ValueError("resize images to at most 2048 pixels per dimension")


def geometry_ok(H, shape):
    """Conservative gate for modest camera movement, not arbitrary panoramas."""
    if H is None or H.shape != (3, 3) or not np.isfinite(H).all() or abs(H[2, 2]) < 1e-9:
        return False
    h, w = shape[:2]
    corners = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    H = H / H[2, 2]
    denom = np.c_[corners, np.ones(4)] @ H[2]
    if np.min(denom) <= .2 or np.max(denom) >= 5:
        return False
    warped = cv2.perspectiveTransform(corners[None], H)[0]
    area_ratio = cv2.contourArea(warped, oriented=True) / ((w-1)*(h-1))
    return bool(cv2.isContourConvex(warped) and .5 < area_ratio < 2
                and np.max(np.linalg.norm(warped-corners, axis=1)) < .4*np.hypot(w, h))


def regions_from_score(score, valid, threshold, min_area):
    mask = ((score > threshold) & valid).astype(np.uint8)
    # Opening removes isolated interpolation noise. Reapply overlap after morphology.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask &= valid.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    selected = np.zeros_like(valid, dtype=bool)
    regions = []
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if area >= min_area:
            selected |= labels == i
            regions.append(dict(x=x, y=y, width=w, height=h, pixels=area))
    return selected, sorted(regions, key=lambda r: -r["pixels"])


def difference(before, aligned, valid):
    """Remove an additive channel-wise light shift; not general relighting."""
    a = cv2.GaussianBlur(before.astype(np.float32), (5, 5), 0)
    b = cv2.GaussianBlur(aligned.astype(np.float32), (5, 5), 0)
    offset = np.median((b-a)[valid], axis=0)
    return np.max(np.abs(b-a-offset), axis=2), offset


def inspect(before, after, config=Config()):
    validate_image(before)
    validate_image(after)
    if before.shape != after.shape:
        raise ValueError("before and after must have identical dimensions")
    cv2.setRNGSeed(41)
    cv2.setNumThreads(1)
    sift = cv2.SIFT_create(nfeatures=2200)
    ka, da = sift.detectAndCompute(cv2.cvtColor(before, cv2.COLOR_RGB2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(after, cv2.COLOR_RGB2GRAY), None)
    q = {"before_keypoints": len(ka), "after_keypoints": len(kb)}
    def reject(reason):
        return Result("REVIEW", reason, q)
    if da is None or db is None or min(len(da), len(db)) < config.min_inliers:
        return reject("Too little image texture to establish alignment.")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(db, da, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < .72*p[1].distance]
    # One target feature must not count repeatedly as independent evidence.
    unique = {}
    for m in sorted(good, key=lambda m: m.distance):
        unique.setdefault(m.trainIdx, m)
    good = list(unique.values())
    q["matches"] = len(good)
    if len(good) < config.min_inliers:
        return reject("Too few distinctive matches between these images.")
    src = np.float32([kb[m.queryIdx].pt for m in good])
    dst = np.float32([ka[m.trainIdx].pt for m in good])
    H, support = cv2.findHomography(src, dst, cv2.RANSAC, 2.5, maxIters=3000, confidence=.995)
    if H is None or support is None:
        return reject("No stable camera alignment was found.")
    inliers = support.ravel().astype(bool)
    q.update(inliers=int(inliers.sum()), inlier_ratio=float(inliers.mean()))
    if q["inliers"] < config.min_inliers or q["inlier_ratio"] < config.min_inlier_ratio:
        return reject("Too many feature matches disagree about the alignment.")
    h, w = before.shape[:2]
    q["feature_coverage"] = min(float(cv2.contourArea(cv2.convexHull(p[inliers]))/(h*w)) for p in (src, dst))
    if q["feature_coverage"] < config.min_coverage:
        return reject("Matching features are concentrated in too small an area.")
    if not geometry_ok(H, before.shape):
        return reject("Camera movement exceeds the supported inspection range.")
    aligned = cv2.warpPerspective(after, H, (w, h))
    support_image = cv2.warpPerspective(np.ones((h, w), np.uint8)*255, H, (w, h))
    valid = cv2.erode((support_image == 255).astype(np.uint8), np.ones((9, 9), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    q["overlap"] = float(valid.mean())
    if q["overlap"] < config.min_overlap:
        return reject("Too little shared image area remains after alignment.")
    reproj = cv2.perspectiveTransform(src[inliers][None], H)[0]
    q["median_match_error_px"] = float(np.median(np.linalg.norm(reproj-dst[inliers], axis=1)))
    score, offset = difference(before, aligned, valid)
    q["brightness_offset_rgb"] = offset.tolist()
    mask, regions = regions_from_score(score, valid, config.threshold, config.min_area)
    return Result("CHANGES" if regions else "NO_CHANGE", "Alignment passed the quality checks.",
                  q, H, aligned, valid, score, mask, regions)
