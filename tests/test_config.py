import unittest
from change_inspector.core import Config


class ConfigurationTests(unittest.TestCase):
    def test_invalid_minimum_match_count_is_rejected(self):
        for value in (float('nan'), float('inf'), 4.5, '12', True, 3):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Config(min_inliers=value)

    def test_valid_match_counts_remain_supported(self):
        self.assertEqual(Config(min_inliers=4).min_inliers, 4)
        self.assertEqual(Config().min_inliers, 12)
