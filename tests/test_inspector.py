import contextlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image
from change_inspector.core import Config, inspect, geometry_ok, regions_from_score, difference
from change_inspector.benchmark import counts, rates, make_pair
from change_inspector.cli import main, load_pair
from change_inspector.report import write_report


def textured(seed=44):
    rng = np.random.default_rng(seed)
    a = np.full((256, 384, 3), 100, np.uint8)
    for _ in range(240):
        x, y = rng.integers([8, 8], [376, 248])
        cv2.circle(a, (int(x), int(y)), int(rng.integers(2, 9)), tuple(map(int, rng.integers(25, 230, 3))), -1)
    return a


class InspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = textured()

    def test_identity_no_change(self):
        r = inspect(self.image, self.image)
        self.assertEqual(r.status, 'NO_CHANGE')
        self.assertLess(r.quality['median_match_error_px'], .01)

    def test_known_translation_direction(self):
        forward = np.float64([[1, 0, 9], [0, 1, -7], [0, 0, 1]])
        after = cv2.warpPerspective(self.image, forward, (384, 256))
        r = inspect(self.image, after)
        self.assertEqual(r.status, 'NO_CHANGE')
        np.testing.assert_allclose(r.homography, np.linalg.inv(forward), atol=.15)

    def test_additive_brightness_is_not_content_change(self):
        r = inspect(self.image, (self.image.astype(np.int16)+12).astype(np.uint8))
        self.assertEqual(r.status, 'NO_CHANGE')
        np.testing.assert_allclose(r.quality['brightness_offset_rgb'], [12]*3, atol=.2)

    def test_added_patch_localized(self):
        after = self.image.copy()
        after[100:140, 150:200] = (245, 20, 40)
        truth = np.zeros(after.shape[:2], bool)
        truth[100:140, 150:200] = True
        r = inspect(self.image, after)
        self.assertEqual(r.status, 'CHANGES')
        self.assertGreater(rates(counts(r.mask, truth, r.valid))['iou'], .85)

    def test_flat_pair_requires_review(self):
        flat = np.full_like(self.image, 125)
        r = inspect(flat, flat)
        self.assertEqual(r.status, 'REVIEW')
        self.assertIsNone(r.mask)
        self.assertIsNone(r.summary()['changed_pixels'])

    def test_unrelated_images_require_review(self):
        self.assertEqual(inspect(self.image, textured(999)).status, 'REVIEW')

    def test_no_detections_outside_overlap(self):
        after, _, _ = make_pair(self.image, 205)
        r = inspect(self.image, after)
        self.assertNotEqual(r.status, 'REVIEW')
        self.assertFalse(r.mask[~r.valid].any())
        self.assertFalse(r.valid[0].any())

    def test_too_little_overlap_requires_review(self):
        after = cv2.warpAffine(self.image, np.float32([[1, 0, 70], [0, 1, 0]]), (384, 256))
        r = inspect(self.image, after, Config(min_overlap=.9))
        self.assertEqual(r.status, 'REVIEW')
        self.assertIn('shared', r.reason)

    def test_invalid_image_dtype(self):
        with self.assertRaises(ValueError):
            inspect(self.image.astype(float), self.image)

    def test_unequal_image_dimensions(self):
        with self.assertRaises(ValueError):
            inspect(self.image, self.image[:-1])

    def test_invalid_config(self):
        for kwargs in ({'threshold':float('nan')}, {'threshold':0}, {'min_area':0},
                       {'min_inliers':3}, {'min_overlap':1.1}, {'min_area':1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)

    def test_implausible_homographies(self):
        for H in (np.eye(3)*np.nan, np.zeros((3, 3)), np.diag([-1., 1., 1.]),
                  np.array([[1., 0, 300], [0, 1, 0], [0, 0, 1]])):
            self.assertFalse(geometry_ok(H, self.image.shape))
        self.assertTrue(geometry_ok(np.eye(3), self.image.shape))

    def test_small_regions_filtered(self):
        score = np.zeros((64, 64), np.float32)
        score[5:8, 5:8] = 100
        score[20:30, 20:30] = 100
        mask, regions = regions_from_score(score, np.ones_like(score, bool), 36, 50)
        self.assertEqual(len(regions), 1)
        self.assertEqual(int(mask.sum()), 100)

    def test_mask_excludes_invalid_region(self):
        score = np.ones((64, 64), np.float32)*100
        valid = np.zeros((64, 64), bool)
        valid[20:40, 20:40] = True
        mask, _ = regions_from_score(score, valid, 36, 50)
        self.assertFalse(mask[~valid].any())

    def test_counts_distinguish_undefined_from_perfect(self):
        empty = np.zeros((4, 4), bool)
        r = rates(counts(empty, empty, ~empty))
        self.assertIsNone(r['iou'])
        self.assertEqual(r['false_positive_rate'], 0)

    def test_generation_reproducible(self):
        a, mask, H = make_pair(self.image, 201)
        b, mask2, H2 = make_pair(self.image, 201)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(mask, mask2)
        np.testing.assert_array_equal(H, H2)

    def test_no_edit_truth_is_empty(self):
        _, mask, _ = make_pair(self.image, 200, False)
        self.assertFalse(mask.any())

    def test_report_escapes_script_termination(self):
        with tempfile.TemporaryDirectory() as d:
            write_report(d, {'title':'</script><script>alert(1)</script>', 'examples':[]})
            text = (Path(d)/'report.html').read_text(encoding='utf-8')
            self.assertNotIn('</script><script>alert(1)', text)
            self.assertIn('\\u003c/script>', text)

    def test_cli_exports_and_cleans_stale_detections(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            before, after, out = root/'before.png', root/'after.png', root/'out'
            Image.fromarray(self.image).save(before)
            Image.fromarray(self.image).save(after)
            args = ['change-inspector', 'compare', str(before), str(after), '--out', str(out)]
            with patch('sys.argv', args), contextlib.redirect_stdout(StringIO()):
                self.assertEqual(main(), 0)
            self.assertTrue((out/'mask.png').exists())
            Image.fromarray(np.full_like(self.image, 128)).save(after)
            with patch('sys.argv', args), contextlib.redirect_stdout(StringIO()):
                self.assertEqual(main(), 2)
            self.assertFalse((out/'mask.png').exists())
            self.assertEqual(json.loads((out/'metrics.json').read_text())['summary']['status'], 'REVIEW')

    def test_cli_invalid_input_exit_code(self):
        with patch('sys.argv', ['change-inspector','compare','does-not-exist.png','other.png']), contextlib.redirect_stderr(StringIO()):
            self.assertEqual(main(), 1)

    def test_loader_resizes_both_consistently(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'image.png'
            Image.fromarray(self.image).resize((1152, 768)).save(p)
            a, b = load_pair(p, p)
            self.assertEqual(a.shape, (480, 720, 3))
            np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
