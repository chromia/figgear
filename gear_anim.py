"""
This is an example which use gear.py.
copyright (C) chromia<chromia@outlook.jp>

License: MIT License (see ./LICENSE)
"""

from typing import Tuple, Dict, Set, Any
import pygame
import sys
import io
import math
from gear import make_gear_image


class Gear:
    def __init__(self, params: Dict[str, Any], driver: 'Gear' = None):
        """
        歯車を生成する
        :param params: 歯車のパラメータを表す辞書。詳細はgear.make_gear_image、
        またはpython gear.py --helpを参照のこと
        :param driver: 本歯車に影響を及ぼす駆動歯車。自らが駆動歯車の場合はNone
        """
        fp = io.BytesIO()
        make_gear_image(output=fp, **params)
        fp.seek(0)
        self.surface = pygame.image.load(fp, "_.png")
        self.m = params['m']
        self.z = params['z']
        self.radius = self.m * self.z / 2
        self.angle = 0.0
        self.velocity = 0.0
        self.cx = 0
        self.cy = 0
        self.driver = driver
        self.driven_set: Set['Gear'] = set()
        if driver:
            driver.add_driven(self)
            self.adjust_velocity()

    def add_driven(self, driven: 'Gear') -> None:
        """
        従動歯車を登録する。この情報は自らの速度が変更された時に、
        従動歯車の速度も合わせて更新するために用いる
        :param driven: 従動歯車
        """
        self.driven_set.add(driven)

    def set_center(self, xy: Tuple[float, float]) -> None:
        """
        歯車中心の描画位置を指定する
        :param xy: 歯車の中心座標(x, y)。単位はpixel
        """
        self.cx = xy[0]
        self.cy = xy[1]

    def adjust_center(self, direction_angle: float) -> None:
        """
        駆動歯車に合わせて自らの位置を決定する
        :param direction_angle: 相手の歯車に対する自身の方位角(degree)
        """
        if self.driver is None:
            raise ValueError('Parent gear is not set.')
        a = self.radius + self.driver.radius  # 中心距離
        t = math.radians(direction_angle)
        self.cx = self.driver.cx + a * math.cos(t)
        self.cy = self.driver.cy + a * math.sin(t)

    def set_angle(self, angle: float) -> None:
        """
        歯車の回転角を指定する
        :param angle: 回転角(degree)
        """
        self.angle = angle

    def set_velocity(self, velocity: float) -> None:
        """
        歯車の角速度を指定する
        :param velocity: 角速度(degree per frame)
        :return:
        """
        self.velocity = velocity
        # 従動歯車も合わせて更新する
        for driven in self.driven_set:
            driven.adjust_velocity()

    def adjust_velocity(self) -> None:
        """
        駆動歯車に合わせて自らの角速度を決定する
        """
        if self.driver is None:
            raise ValueError('Parent gear is not set.')
        self.velocity = -self.driver.velocity * self.driver.z / self.z
        # 従動歯車も合わせて更新する
        for driven in self.driven_set:
            driven.adjust_velocity()

    def is_driven(self) -> bool:
        """
        自らが従動歯車かどうかを取得する
        :return: 従動歯車の場合True,そうでなければFalse
        """
        return self.driver is not None

    def rotate(self) -> None:
        """
        歯車の回転角を更新する。変化量はvelocityによる
        """
        self.angle += self.velocity

    def _get_corner(self, rotated_image: 'pygame.Surface') -> Tuple[int, int]:
        """
        画像を描画する際に必要な左上隅の座標を求める
        :param rotated_image: 回転処理を施した画像
        :return: 左上隅の座標(x, y)
        """
        w = rotated_image.get_width()
        h = rotated_image.get_height()
        return int(self.cx - w / 2), int(self.cy - h / 2)

    def draw(self, screen: 'pygame.Surface') -> None:
        """
        歯車を描画する
        :param screen: 描画先のサーフェス
        """
        img = pygame.transform.rotozoom(self.surface, self.angle, 1.0)
        screen.blit(img, self._get_corner(img))


def quit_app():
    pygame.quit()
    sys.exit(0)


def main():
    # Pygameの初期化
    pygame.init()
    screen = pygame.display.set_mode((480, 480))
    pygame.display.set_caption("Gear sample")
    font = pygame.font.Font(None, 20)

    # 歯車の生成と設定
    gear1_params = {"m": 8, "z": 24, "r": 255, "g": 0, "b": 0,
                    "ssaa": 2, "draw_center_hole": True}
    gear2_params = {"m": 8, "z": 12, "r": 0, "g": 0, "b": 255,
                    "ssaa": 2}
    gear3_params = {"m": 8, "z": 12, "r": 0, "g": 255, "b": 0,
                    "ssaa": 2}
    gear4_params = {"m": 8, "z": 16, "r": 255, "g": 0, "b": 255,
                    "ssaa": 2}
    gear5_params = {"m": 8, "z": 24, "r": 128, "g": 128, "b": 128,
                    "ssaa": 2, "draw_center_hole": True, "center_hole_size": 0.1}
    gear1 = Gear(gear1_params)  # 歯車(駆動歯車)を作る
    gear1.set_center((140, 140))  # 位置を決める
    gear1.set_velocity(0.01)  # 速度を決める
    gear1.set_angle(0.0)  # 初期角を決める
    gear2 = Gear(gear2_params, gear1)  # 2は1に依存
    gear2.adjust_center(0.0)  # 依存対象を基準とした方位で位置を決定
    gear2.set_angle(2.6)  # かみ合って見えるように角度を調整
    gear3 = Gear(gear3_params, gear1)  # 3は1に依存
    gear3.adjust_center(90.0)
    gear3.set_angle(2.6)
    gear4 = Gear(gear4_params, gear3)  # 4は3に依存
    gear4.adjust_center(60.0)
    gear4.set_angle(7.0)
    gear5 = Gear(gear5_params, gear4)  # 5は4に依存
    gear5.adjust_center(-30.0)
    gear5.set_angle(1.8)
    gears = [gear1, gear2, gear3, gear4, gear5]

    # メインループ
    while True:
        # 画面のクリア
        screen.fill((0, 0, 0))

        # 歯車の更新と描画
        for gear in gears:
            gear.rotate()
            gear.draw(screen)

        # 数値の描画
        text = f'Rotation Velocity: {gear1.velocity:.02}   +: I-key, -: K-key'
        text_velocity = font.render(text, True, (255, 255, 255))
        screen.blit(text_velocity, [10, 10])

        # 画面の更新
        pygame.display.update()

        # イベントを確認
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                # ESCキー押下で終了
                if event.key == pygame.K_ESCAPE:
                    quit_app()
                elif event.key == pygame.K_i:
                    # [i]キー: 速度プラス
                    gear1.set_velocity(gear1.velocity + 0.01)
                elif event.key == pygame.K_k:
                    # [j]キー: 速度マイナス
                    gear1.set_velocity(gear1.velocity - 0.01)
            if event.type == pygame.QUIT:
                # Windowの×ボタン押下で終了
                quit_app()


if __name__ == "__main__":
    main()
