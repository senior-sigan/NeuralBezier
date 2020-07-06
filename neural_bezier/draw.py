from typing import Tuple

import cv2
import numpy as np


def normal(points: Tuple[float, ...], width: int) -> Tuple[float, ...]:
    p_ = np.asarray(points)
    p_ = p_ * (width - 1) + 0.5
    return tuple(p_.astype(np.int))


def draw_params(params, *, size: int = 256, debug: bool = False) -> np.ndarray:
    return draw(p0=params[0:2], p1=params[2:4], p2=params[4:6],
                radius=params[6:8], color=params[8:10],
                size=size, debug=debug)


def draw(p0: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float],
         radius: Tuple[float, float], color: Tuple[float, float],
         *, size: int = 256, debug: bool = False) -> np.ndarray:
    r"""Draws a quadratic Bezier curve on a grayscale canvas and returns it as a numpy array.

    It uses circles with the radius to draw the curve.
    Color and Radius is a simple linear gradient.

    :math:`B(t) = (1 - t)^2 P_0 + 2t(1 - t) P_1 + t^2 P_2, t \in [0,1]`

    See: https://en.wikipedia.org/wiki/B%C3%A9zier_curve

    Args:
        p0 (Tuple[float, float]): Point where to start drawing the curve. Should be in [0,1] but could be any.
        p1 (Tuple[float, float]): Middle point to set the curve. Should be in [0,1] but could be any.
        p2 (Tuple[float, float]): Point where to finish drawing the curve. Should be in [0,1] but could be any.
        radius (Tuple[float, float]): Radius of the circle is used to emulate drawing. Must be > 0.
        color (Tuple[float, float]): Transparency value of the circle. Must be in [0, 1]
        size (int, optional): Width and height of the generated canvas in pixels. Defaults to 256.
        debug (bool, optional): Draw or not P0, P1, P2 on the canvas. Defaults to False.

    Returns:
        np.ndarray: grayscale square image with values in [0,255], dims=(size, size), dtype=np.uint8
    """
    x0, y0 = normal(p0, size * 2)
    x1, y1 = normal(p1, size * 2)
    x2, y2 = normal(p2, size * 2)
    r0, r1 = radius
    c0, c1 = color
    r0 = r0 * size // 2 + 1  # avoid zero radius, so add 1
    r1 = r1 * size // 2 + 1  # avoid zero radius, so add 1
    c0 = c0 * 255
    c1 = c1 * 255

    canvas = np.zeros([size * 2, size * 2], dtype=np.uint8)

    if debug:
        # debug draw P0, P1, P2
        cv2.circle(img=canvas, center=(x0, y0), radius=6,
                   color=255, thickness=cv2.FILLED)
        cv2.circle(img=canvas, center=(x1, y1), radius=6,
                   color=255, thickness=cv2.FILLED)
        cv2.circle(img=canvas, center=(x2, y2), radius=6,
                   color=255, thickness=cv2.FILLED)

    ts = np.linspace(start=0, stop=1, num=size)
    xs = (1 - ts) ** 2 * x0 + 2 * (1 - ts) * ts * x1 + ts ** 2 * x2
    ys = (1 - ts) ** 2 * y0 + 2 * (1 - ts) * ts * y1 + ts ** 2 * y2
    rs = (1 - ts) * r0 + ts * r1
    cs = (1 - ts) * c0 + ts * c1

    for x, y, r, c in zip(xs, ys, rs, cs):
        cv2.circle(img=canvas, center=(int(x), int(y)),
                   radius=int(r), color=int(c), thickness=cv2.FILLED)

    return cv2.resize(src=canvas, dsize=(size, size)).astype(np.uint8)
