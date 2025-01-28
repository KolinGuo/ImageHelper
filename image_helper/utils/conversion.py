from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def bbox_to_anchors(
    bbox: Sequence[float], anchor: Optional[str] = None
) -> np.ndarray | tuple[float, float]:
    """Convert bbox to anchor points
    :param bbox: [left, top, right, bottom]
    :param anchor: If provided, only extract position of that anchor. Choices:
                   'lt', 'mt', 'rt', 'lm', 'mm', 'rm', 'lb', 'mb', 'rb'
                   Otherwise, extract the four corners of the bbox.
    Anchor locations:
        *-------*--------*
        | lt    mt    rt |
        |                |
        * lm    mm    rm *
        |                |
        | lb    mb    rb |
        *-------*--------*

    :return anchor_xy: xy pixel position of anchor points.
                       np.ndarray of shape (4, 2) or tuple of length 2
    """
    left, top, right, bottom = bbox
    if anchor is None:
        return np.array([
            [left, top],
            [right, top],
            [left, bottom],
            [right, bottom],
        ])

    anchor_h, anchor_v = anchor
    if anchor_h == "l":
        x = left
    elif anchor_h == "m":
        x = (left + right) / 2.0
    elif anchor_h == "r":
        x = right
    else:
        raise ValueError(
            f"Unknown anchor[0] '{anchor_h}'. Value choices: 'l', 'm', 'r'"
        )
    if anchor_v == "t":
        y = top
    elif anchor_v == "m":
        y = (top + bottom) / 2.0
    elif anchor_v == "b":
        y = bottom
    else:
        raise ValueError(
            f"Unknown anchor[1] '{anchor_v}'. Value choices: 't', 'm', 'b'"
        )
    return x, y


def anchor_to_bbox(
    anchor_xy: Sequence[float], bbox_size: Sequence[float], anchor: str = "lt"
) -> tuple[float, float, float, float]:
    """Convert anchor and bbox_size to bbox
    :param anchor_xy: xy pixel position of anchor
    :param bbox_size: the size of the bbox, (width, height).
    :param anchor: anchor position. Choices:
                   'lt', 'mt', 'rt', 'lm', 'mm', 'rm', 'lb', 'mb', 'rb'
    Anchor locations:
        *-------*--------*
        | lt    mt    rt |
        |                |
        * lm    mm    rm *
        |                |
        | lb    mb    rb |
        *-------*--------*
    :return bbox: [left, top, right, bottom]
    """
    x, y = anchor_xy
    width, height = bbox_size

    anchor_h, anchor_v = anchor
    if anchor_h == "l":
        left, right = x, x + width
    elif anchor_h == "m":
        half_width = width / 2.0
        left, right = x - half_width, x + half_width
    elif anchor_h == "r":
        left, right = x - width, x
    else:
        raise ValueError(
            f"Unknown anchor[0] '{anchor_h}'. Value choices: 'l', 'm', 'r'"
        )
    if anchor_v == "t":
        top, bottom = y, y + height
    elif anchor_v == "m":
        half_height = height / 2.0
        top, bottom = y - half_height, y + half_height
    elif anchor_v == "b":
        top, bottom = y - height, y
    else:
        raise ValueError(
            f"Unknown anchor[1] '{anchor_v}'. Value choices: 't', 'm', 'b'"
        )
    return (left, top, right, bottom)


def get_bbox_rel_to_bbox(
    anchor: str,
    rel_bbox: Sequence[float],
    bbox_size: Optional[Sequence[float]] = None,
    pad_size: int = 0,
) -> tuple[tuple[float, float], str | tuple[float, float, float, float]]:
    """Get bbox coordinates relative to an existing bbox
        :param anchor: anchor position, composed of:
                       left (l), top (t), right (r), bottom (b), middle (m)
                       inside (i), outside (o), corner (c)
        The anchor position is indicated by '+' below.
    ............. ..........     .....................     .......... .............
    | tlco/ltco | | tl(o)  |     | t(o)/mt(o)/tm(o)  |     |  tr(o) | | trco/rtco |
    ............+ +.........     ..........+..........     .........+ +............
        ........+ *-------------------------------------------------* +........
        | lt(o) | | +...........    .......+.......    ...........+ | | rt(o) |
        ......... | | lti/tli/ |    | ti/mti/tmi  |    | tri/rti/ | | .........
                  | | ltc/tlc  |    ...............    | trc/rtc  | |
                  | ............                       ............ |
      ........... |                                                 | ...........
      |   l(o)/ | | ...............   ...........   ............... | | r(o)/   |
      |  lm(o)/ + | + li/lmi/mli  |   |  m(mi)  |   |  ri/mri/rmi + | + rm(o)/  |
      |  ml(o)  | | ...............   ...........   ............... | | mr(o)   |
      ........... |                                                 | ...........
                  | ............                       ............ |
                  | | lbi/bli/ |    ...............    | bri/rbi/ | |
        ......... | | lbc/blc  |    | bi/mbi/bmi  |    | brc/rbc  | | .........
        | lb(o) | | +...........    .......+.......    ...........+ | | rb(o) |
        ........+ *-------------------------------------------------* +........
    ............+ +.........     ..........+..........     .........+ +............
    | blco/lbco | | bl(o)  |     | b(o)/mb(o)/bm(o)  |     |  br(o) | | brco/rbco |
    ............. ..........     .....................     .......... .............

        :param rel_bbox: existing bbox coordinates: [left, top, right, bottom]
        :param bbox_size: the size of the added bbox, (width, height).
                          If None, only return bbox_anchor string.
        :param pad_size: the padding size between rel_bbox and added bbox.
        :return anchor_xy: xy pixel position (can be float) of anchor
                           relative to the entire image
        :return bbox: the added box, [left, top, right, bottom]
                      If bbox_size = None, return the bbox_anchor string. Choices:
                          'lt', 'mt', 'rt', 'lm', 'mm', 'rm', 'lb', 'mb', 'rb'
    """
    if anchor in ["tlco", "ltco"]:
        rel_bbox_anchor, bbox_anchor = "lt", "rb"
        pad_x, pad_y = -pad_size, -pad_size
    elif anchor in ["tl", "tlo"]:
        rel_bbox_anchor, bbox_anchor = "lt", "lb"
        pad_x, pad_y = 0, -pad_size
    elif anchor in ["lt", "lto"]:
        rel_bbox_anchor, bbox_anchor = "lt", "rt"
        pad_x, pad_y = -pad_size, 0
    elif anchor in ["lti", "tli", "ltc", "tlc"]:
        rel_bbox_anchor, bbox_anchor = "lt", "lt"
        pad_x, pad_y = pad_size, pad_size
    elif anchor in ["tr", "tro"]:
        rel_bbox_anchor, bbox_anchor = "rt", "rb"
        pad_x, pad_y = 0, -pad_size
    elif anchor in ["trco", "rtco"]:
        rel_bbox_anchor, bbox_anchor = "rt", "lb"
        pad_x, pad_y = pad_size, -pad_size
    elif anchor in ["tri", "rti", "trc", "rtc"]:
        rel_bbox_anchor, bbox_anchor = "rt", "rt"
        pad_x, pad_y = -pad_size, pad_size
    elif anchor in ["rt", "rto"]:
        rel_bbox_anchor, bbox_anchor = "rt", "lt"
        pad_x, pad_y = pad_size, 0
    elif anchor in ["lb", "lbo"]:
        rel_bbox_anchor, bbox_anchor = "lb", "rb"
        pad_x, pad_y = -pad_size, 0
    elif anchor in ["lbi", "bli", "lbc", "blc"]:
        rel_bbox_anchor, bbox_anchor = "lb", "lb"
        pad_x, pad_y = pad_size, -pad_size
    elif anchor in ["blco", "lbco"]:
        rel_bbox_anchor, bbox_anchor = "lb", "rt"
        pad_x, pad_y = -pad_size, pad_size
    elif anchor in ["bl", "blo"]:
        rel_bbox_anchor, bbox_anchor = "lb", "lt"
        pad_x, pad_y = 0, pad_size
    elif anchor in ["bri", "rbi", "brc", "rbc"]:
        rel_bbox_anchor, bbox_anchor = "rb", "rb"
        pad_x, pad_y = -pad_size, -pad_size
    elif anchor in ["rb", "rbo"]:
        rel_bbox_anchor, bbox_anchor = "rb", "lb"
        pad_x, pad_y = pad_size, 0
    elif anchor in ["br", "bro"]:
        rel_bbox_anchor, bbox_anchor = "rb", "rt"
        pad_x, pad_y = 0, pad_size
    elif anchor in ["brco", "rbco"]:
        rel_bbox_anchor, bbox_anchor = "rb", "lt"
        pad_x, pad_y = pad_size, pad_size
    elif anchor in ["t", "to", "mt", "mto", "tm", "tmo"]:
        rel_bbox_anchor, bbox_anchor = "mt", "mb"
        pad_x, pad_y = 0, -pad_size
    elif anchor in ["ti", "mti", "tmi"]:
        rel_bbox_anchor, bbox_anchor = "mt", "mt"
        pad_x, pad_y = 0, pad_size
    elif anchor in ["bi", "mbi", "bmi"]:
        rel_bbox_anchor, bbox_anchor = "mb", "mb"
        pad_x, pad_y = 0, -pad_size
    elif anchor in ["b", "bo", "mb", "mbo", "bm", "bmo"]:
        rel_bbox_anchor, bbox_anchor = "mb", "mt"
        pad_x, pad_y = 0, pad_size
    elif anchor in ["l", "lo", "ml", "mlo", "lm", "lmo"]:
        rel_bbox_anchor, bbox_anchor = "lm", "rm"
        pad_x, pad_y = -pad_size, 0
    elif anchor in ["li", "mli", "lmi"]:
        rel_bbox_anchor, bbox_anchor = "lm", "lm"
        pad_x, pad_y = pad_size, 0
    elif anchor in ["ri", "mri", "rmi"]:
        rel_bbox_anchor, bbox_anchor = "rm", "rm"
        pad_x, pad_y = -pad_size, 0
    elif anchor in ["r", "ro", "mr", "mro", "rm", "rmo"]:
        rel_bbox_anchor, bbox_anchor = "rm", "lm"
        pad_x, pad_y = pad_size, 0
    elif anchor in ["m", "mm", "mmi"]:
        rel_bbox_anchor, bbox_anchor = "mm", "mm"
        pad_x, pad_y = 0, 0
    else:
        raise ValueError(f"Unknown anchor '{anchor}'")

    anchor_xy = bbox_to_anchors(rel_bbox, rel_bbox_anchor)
    anchor_xy = (anchor_xy[0] + pad_x, anchor_xy[1] + pad_y)
    if bbox_size is not None:
        return anchor_xy, anchor_to_bbox(anchor_xy, bbox_size, bbox_anchor)
    else:
        return anchor_xy, bbox_anchor
