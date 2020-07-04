import cv2
import numpy as np
from typing import Tuple


def normal(p: Tuple[float, ...], width: int) -> Tuple[float, ...]:
    p = np.asarray(p)
    p = p * (width - 1) + 0.5


def draw(p0: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float],
         radius: Tuple[float, float], color: Tuple[float, float], width: int = 256,
         *, debug: bool = False) -> np.ndarray:
    """Draws a quadratic Bezier curve on a grayscale canvas and returns it as a numpy array.
    It uses circles with the radius to draw the curve.
    Color and Radius is a simple linear gradient.

    $B(t) = (1 - t)^2 P_0 + 2t(1 - t) P_1 + t^2 P_2, \quad  t \in [0,1]$

    See: https://en.wikipedia.org/wiki/B%C3%A9zier_curve

    Args:
        p0 (Tuple[float, float]): Point where to start drawing the curve. Should be in [0,1] but could be any.
        p1 (Tuple[float, float]): Middler to set the curviness. Should be in [0,1] but could be any.
        p2 (Tuple[float, float]): Point where to finish drawing the curve. Should be in [0,1] but could be any.
        radius (Tuple[float, float]): Radius of the circle is used to emulate drawing. Must be > 0.
        color (Tuple[float, float]): Transparency value of the circle. Must be in [0, 1]
        width (int, optional): [description]. Defaults to 256.
        debug (bool, optional): [description]. Defaults to False.

    Returns:
        np.ndarray: [description]
    """
    x0, y0, x1, y1, x2, y2, r0, r1, c0, c1 = (*p0, *p1, *p2, *radius, *color)

    x0, y0, x1, y1, x2, y2 = normal((x0, y0, x1, y1, x2, y2), width*2)
    r0 = r0 * width // 2 + 1  # avoid zero radius, so add 1
    r1 = r1 * width // 2 + 1  # avoid zero radius, so add 1
    c0 = c0 * 255
    c1 = c1 * 255

    canvas = np.zeros([width * 2, width * 2], dtype=np.uint8)

    if debug:
        # debug draw P0, P1, P2
        cv2.circle(img=canvas, center=(x0, y0), radius=6,
                   color=255, thickness=cv2.FILLED)
        cv2.circle(img=canvas, center=(x1, y1), radius=6,
                   color=255, thickness=cv2.FILLED)
        cv2.circle(img=canvas, center=(x2, y2), radius=6,
                   color=255, thickness=cv2.FILLED)

    ts = np.linspace(start=0, stop=1, num=width)
    xs = (1-ts) ** 2 * x0 + 2 * (1-ts) * ts * x1 + ts**2 * x2
    ys = (1-ts) ** 2 * y0 + 2 * (1-ts) * ts * y1 + ts**2 * y2
    rs = (1-ts) * r0 + ts * r1
    cs = (1-ts) * c0 + ts * c1

    for x, y, r, c in zip(xs, ys, rs, cs):
        cv2.circle(img=canvas, center=(int(x), int(y)),
                   radius=int(r), color=int(c), thickness=cv2.FILLED)

    return cv2.resize(src=canvas, dsize=(width, width)).astype(np.uint8)
