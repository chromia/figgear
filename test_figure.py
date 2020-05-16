from unittest import TestCase
from gear import make_gear_figure

"""
テスト

使い方 python -m unittest
"""


class TestGearFigure(TestCase):
    def test_duplication(self):
        """
        同じ点が重複するという不具合があったので、その再発防止用テストケース
        """
        m = 8
        z = 20
        alpha = 20.0
        points1, _ = make_gear_figure(m, z, alpha, "line")
        points2, _ = make_gear_figure(m, z, alpha, "spline")
        for points in [points1, points2]:
            last_x, last_y = None, None
            for x, y in points:
                self.assertFalse(x == last_x and y == last_y)
                last_x, last_y = x, y
