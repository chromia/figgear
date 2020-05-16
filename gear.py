"""
This module implements gear generation(figure and image).
copyright (C) chromia<chromia@outlook.jp>

License: MIT License (see ./LICENSE)
"""

from typing import List, Tuple, Dict, Union, BinaryIO
import argparse
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import splprep, splev

# Type Hints
Point = Tuple[float, float]
PointList = List[Point]

# オプションのデフォルト値
_alpha = 20.0
_ssaa = 1
__draw_circles = False
__draw_center_hole = False
__center_hole_size = 0.15
__bottom_type = "spline"
__image_margin = 0
_col_r = 128
_col_g = 128
_col_b = 128
_col_a = 128
_col_af = 255


def _inv(alpha: float) -> float:
    """
    インボリュート角を求める
    :param alpha: 圧力角(radian)
    :return: インボリュート角(radian)
    """
    return math.tan(alpha) - alpha


def _add_bottom_points_line(points: PointList, new_points: PointList) -> None:
    """
    補間なし(直線)
    :param points: 既存の点列
    :param new_points: 新規に追加する点列
    :return: なし
    """
    for point in new_points[1:-1]:
        points.append(point)


def _add_bottom_points_spline(points: PointList, new_4_points: PointList,
                              division_num: int = 50) -> None:
    """
    スプライン補間で歯底の曲線を作る
    :param points: 既存の点列
    :param new_4_points: 曲線を生成するための4つの点
    :param division_num: 曲線を構成する部分直線の本数(多いほど滑らか)
    :return: なし
    """
    n_anchor = 3  # 底辺を何等分するか
    i_anchor = 1  # 底辺のどの点を使用するか

    p0, p1, p2, p3 = new_4_points
    # 底辺をn個に分割、中間2点を通るように(分割比率はn_anchorによる)
    xs = [p0[0],
          ((n_anchor - i_anchor) * p1[0] + i_anchor * p2[0]) / n_anchor,
          (i_anchor * p1[0] + (n_anchor - i_anchor) * p2[0]) / n_anchor,
          p3[0]]
    ys = [p0[1],
          ((n_anchor - i_anchor) * p1[1] + i_anchor * p2[1]) / n_anchor,
          (i_anchor * p1[1] + (n_anchor - i_anchor) * p2[1]) / n_anchor,
          p3[1]]
    # 4つの点を通るように3次スプライン曲線で補間する
    spline = splprep([xs, ys])[0]
    detail = np.linspace(0, 1, num=division_num, endpoint=True)
    ix, iy = splev(detail, spline)
    for x, y in zip(ix[1:-1], iy[1:-1]):
        points.append((x, y))


def make_gear_figure(m: int, z: int, alpha_deg: float, bottom_type: str) \
        -> Tuple[PointList, Dict[str, float]]:
    """
    歯車の形状を表す点列を生成する。点列を順番に線で結ぶと歯車が描画できる。
    中心は(0,0)となる。
    :param m: モジュール
    :param z: 歯数
    :param alpha_deg: 圧力角(degree)
    :param bottom_type: 歯元部の形状。"spline"か"line"を指定
    :return: (points, blueprints)
      points: 歯車の形状を表す座標点列
      blueprints: 歯車の諸元情報
    """
    alpha = math.radians(alpha_deg)  # 圧力角(radian)
    p = m * math.pi  # ピッチ
    s = p / 2  # 歯厚

    diameter_pitch = z * m  # 基準円直径(dp)
    diameter_addendum = diameter_pitch + 2 * m  # 歯先円直径(dk)
    diameter_dedendum = diameter_pitch - 2.5 * m  # 歯底円直径(df)
    diameter_base = diameter_pitch * math.cos(alpha)  # 基礎円直径(db)
    radius_pitch = diameter_pitch / 2  # 基準円半径(rp)
    radius_addendum = diameter_addendum / 2  # 歯底円半径(rk)
    radius_dedendum = diameter_dedendum / 2  # 歯先円半径(rf)
    radius_base = diameter_base / 2  # 基礎円半径(rb)

    angle_1 = 2 * math.pi / z  # 歯1つ分の等分角度
    angle_s = s / radius_pitch  # 基準円上、歯厚分の角度
    k_inv = _inv(math.acos(radius_base / radius_pitch))  # インボリュート角
    angle_base = angle_s + k_inv * 2  # 基礎円上、歯厚分の角度
    angle_bottom = angle_1 - angle_base  # 歯底の幅分の角度
    cos_bottom = math.cos(-angle_bottom)  # ※逆方向に回転させる
    sin_bottom = math.sin(-angle_bottom)

    inv_step = 0.5  # インボリュート曲線の点の間隔
    r_diff = radius_addendum - radius_base
    inv_count = math.ceil(r_diff / inv_step)  # 曲線の分割点数

    points = []
    for i in range(z):
        t = angle_1 * i
        cos_t, sin_t = math.cos(t), math.sin(t)
        # 歯底のラインを描く
        xa, ya = radius_base * cos_t, radius_base * sin_t  # 基礎円上の点A
        xb, yb = radius_dedendum * cos_t, radius_dedendum * sin_t  # 歯底円上の点B
        xc = xb * cos_bottom - yb * sin_bottom  # 歯底円上の点C(隣の歯)
        yc = xb * sin_bottom + yb * cos_bottom
        xd = xa * cos_bottom - ya * sin_bottom  # 基礎円上の点D(隣の歯)
        yd = xa * sin_bottom + ya * cos_bottom
        points_bottom = [(xd, yd), (xc, yc), (xb, yb), (xa, ya)]
        if bottom_type == "line":
            _add_bottom_points_line(points, points_bottom)
        else:
            _add_bottom_points_spline(points, points_bottom)

        # インボリュート曲線を描く
        points_inv1 = []
        points_inv2 = []
        cos_inv2, sin_inv2 = math.cos(t + angle_base), math.sin(t + angle_base)
        for r in np.linspace(radius_base, radius_addendum, inv_count):
            inv_alpha = _inv(math.acos(radius_base / r))  # インボリュート角
            x = r * math.cos(inv_alpha)
            y = r * math.sin(inv_alpha)
            # 片側のインボリュート曲線の点を求める
            x1 = x * cos_t - y * sin_t
            y1 = x * sin_t + y * cos_t
            points_inv1.append((x1, y1))
            # 反対側のインボリュート曲線の点を求める(向きを逆に&角度シフト)
            x2 = x * cos_inv2 - (-y) * sin_inv2
            y2 = x * sin_inv2 + (-y) * cos_inv2
            points_inv2.append((x2, y2))
        for p1 in points_inv1:
            points.append(p1)
        for p2 in reversed(points_inv2):  # 点の順番を考慮して逆順につなげる
            points.append(p2)

    # 諸元をまとめる
    blueprints = {
        'diameter_addendum': diameter_addendum,
        'diameter_pitch': diameter_pitch,
        'diameter_base': diameter_base,
        'diameter_dedendum': diameter_dedendum,
        'radius_addendum': radius_addendum,
        'radius_pitch': radius_pitch,
        'radius_base': radius_base,
        'radius_dedendum': radius_dedendum
    }

    return points, blueprints


def draw_circle(draw: ImageDraw, xy: Tuple[float, float],
                radius: float, fill=None, outline=None, width=1) -> None:
    """
    Pillowを使って画像に円を描く(ImageDraw.ellipseの簡易Wrapper)
    :param draw: PIL.ImageDraw
    :param xy: 円の中心座標(x, y)を表すタプル
    :param radius: 円の半径
    :param fill: 円の塗りつぶし色(tuple or string)
    :param outline: 円の輪郭色(tuple or string)
    :param width: 線の太さ
    :return: なし
    """
    x1 = int(xy[0] - radius)
    y1 = int(xy[1] - radius)
    x2 = int(xy[0] + radius)
    y2 = int(xy[1] + radius)
    draw.ellipse((x1, y1, x2, y2), fill, outline, width)


def make_gear_image(output: Union[str, BinaryIO], m: int, z: int,
                    **kwargs) -> Dict[str, float]:
    """
    歯車の画像を生成する
    :param output: 出力するファイル名 または ファイルオブジェクト(BytesIO等)
    :param m: モジュール
    :param z: 歯数
    :param kwargs: オプション。詳細はpython gear.py --helpで参照のこと
    :return: 歯車の諸元情報
    """

    # オプションの確認
    # 圧力角
    alpha = kwargs.get('alpha', _alpha)
    # 主線と塗りつぶし色の設定
    col_r = kwargs.get('r', _col_r)
    col_g = kwargs.get('g', _col_g)
    col_b = kwargs.get('b', _col_b)
    col_a = kwargs.get('a', _col_a)
    col_af = kwargs.get('af', _col_af)
    outline_color = (col_r, col_g, col_b, col_a)  # 歯車の主線の色(RGBA)
    fill_color = (col_r, col_g, col_b, col_af)  # 歯車の描画色(RGBA)
    # [確認用]各種円描画
    draw_circles = kwargs.get('draw_circles', __draw_circles)
    # センターホール
    draw_center_hole = kwargs.get('draw_center_hole', __draw_center_hole)
    center_hole_size = kwargs.get('center_hole_size', __center_hole_size)
    # スーパーサンプリング
    ssaa = kwargs.get("ssaa", _ssaa)
    # 歯底形状
    bottom_type = kwargs.get("bottom_type", __bottom_type)
    # 画像のマージン
    image_margin = kwargs.get("image_margin", __image_margin)

    # スーパーサンプリングの場合、モジュールを定数倍する
    m *= ssaa
    line_width = ssaa

    # 歯車の形状を求める
    points, blueprints = make_gear_figure(m, z, alpha, bottom_type)

    # 画像を生成
    width = round(blueprints['diameter_addendum']) + ssaa * image_margin * 2
    height = width
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx = width / 2
    cy = height / 2

    # 歯車を描く
    points_shift = [(cx + p[0], cy + p[1]) for p in points]  # 中心を(cx,cy)に
    draw.polygon(points_shift, fill=fill_color, outline=outline_color)

    # 歯車に穴を空ける
    r_pitch = blueprints['radius_pitch']
    if draw_center_hole:
        size = r_pitch * center_hole_size
        draw_circle(draw, (cx, cy), size, fill=(0, 0, 0, 0), outline=outline_color)

    # [確認用]各種の円を描く
    if draw_circles:
        rf, cf = blueprints['radius_dedendum'], (64, 64, 64)
        rb, cb = blueprints['radius_base'], (128, 64, 64)
        rp, cp = blueprints['radius_pitch'], (64, 128, 64)
        rk, ck = blueprints['radius_addendum'], (64, 64, 128)
        draw_circle(draw, (cx, cy), rf, outline=cf, width=line_width)  # 歯底円
        draw_circle(draw, (cx, cy), rb, outline=cb, width=line_width)  # 基礎円
        draw_circle(draw, (cx, cy), rp, outline=cp, width=line_width)  # 基準円
        draw_circle(draw, (cx, cy), rk, outline=ck, width=line_width)  # 歯先円

    img = img.resize((width // ssaa, height // ssaa))
    if isinstance(output, str):
        # ファイルに書き出す
        img.save(output)
    else:
        # ファイルオブジェクトに書き出す
        img.save(output, format="PNG")

    blueprints['cx'] = cx
    blueprints['cy'] = cy
    return blueprints


def main():
    # コマンドライン引数を定義する
    # 必須パラメータ
    parser = argparse.ArgumentParser(description='Gear Image Generator.',
                                     argument_default=argparse.SUPPRESS)
    parser.add_argument('output', type=str,
                        help='The path of output image file.')
    parser.add_argument('m', type=int,
                        help='Module number represents the size of the gear-tooth.')
    parser.add_argument('z', type=int,
                        help='The number of teeth.')
    # オプションパラメータ
    parser.add_argument('--alpha', type=float,
                        help=f'Pressure Angle(in degree). default is {_alpha}.')
    parser.add_argument('--ssaa', type=int,
                        help=f'Super Sampling Antialias(SSAA) Rate. default is {_ssaa}.')
    parser.add_argument('--draw-circles', type=bool,
                        help='Draw several circles to describe the shape of gear.')
    parser.add_argument('--draw-center-hole', type=bool,
                        help='Draw center hole.')
    parser.add_argument('--center-hole-size', type=float,
                        help=f'The size of center hole. default is {__center_hole_size}.')
    parser.add_argument('--bottom-type', type=str,
                        help=f'Specify the figure along bottom line. select "spline" or "line". default is "{__bottom_type}".')
    parser.add_argument('--image-margin', type=int,
                        help=f'The margin size on all four-sides of the image. default is {__image_margin}.')
    # 色に関するパラメータ
    parser.add_argument('-r', type=int,
                        help=f'R channel value of gear color. default is {_col_r}.')
    parser.add_argument('-g', type=int,
                        help=f'G channel value of gear color. default is {_col_g}.')
    parser.add_argument('-b', type=int,
                        help=f'B channel value of gear color. default is {_col_b}.')
    parser.add_argument('-a', type=int,
                        help=f'Alpha channel value of gear border color. default is {_col_a}.')
    parser.add_argument('-af', type=int,
                        help=f'Alpha channel value of gear body color.  default is {_col_af}.')

    # コマンドライン引数を解析する
    args = parser.parse_args()
    # 辞書に変換する
    kwargs = vars(args)

    # 歯車を生成・出力
    make_gear_image(**kwargs)


if __name__ == '__main__':
    main()
