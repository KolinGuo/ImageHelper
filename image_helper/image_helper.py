"""
ImageHelper for combining images and drawing multiline texts

Written by Kolin Guo
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Optional, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .utils.conversion import anchor_to_bbox, get_bbox_rel_to_bbox

FONT_PATH = str(
    Path(__file__).resolve().parent / "fonts/ubuntu-font-family-0.83/UbuntuMono-R.ttf"
)


class ImageHelper:
    """
    A single ImageHelper for combining images and adding texts

    :ivar _image: actual image np.ndarray. [H, W, C]
                  DO NOT edit it directly (can lead to undefined behavior)
    :ivar image_bboxes: tag sub-image bboxes in _image.
                        [left, top, right, bottom]
    :ivar text_bboxes: text bboxes in _image. [left, top, right, bottom]
                       When drawing the new text and encounter overlaps,
                       shift the new text bbox so that there's no overlap.
    """

    def __init__(
        self,
        bound_pad: int = 20,
        text_bbox_pad: int = 4,
        bg_color: Sequence[np.uint8] = (0, 0, 0),  # type: ignore
    ):
        """
        :param bound_pad: number of pixels to pad
                          (e.g., outside of boundary, between images).
        :param text_bbox_pad: number of pixels to pad around drawn text bboxes.
        :param bg_color: background color used for padding
        """
        self._image = np.zeros((0, 0, 3), dtype=np.uint8)
        self.image_bboxes = {}  # {tag: bbox}
        self.text_bboxes = []
        self.bound_pad = bound_pad
        self.text_bbox_pad = text_bbox_pad
        self.bg_color = list(bg_color)

    def _check_image_format(self, image: np.ndarray):
        assert image.dtype == self._image.dtype, (
            f'image must have dtype "{self._image.dtype}", get {image.dtype}'
        )
        assert image.ndim in [3], f"image must be 3 dimensional, get {image.ndim}"
        assert image.shape[2] in [
            3,
            4,
        ], f"image must have 3 (RGB) or 4 (RGBA) channels, get {image.shape}"

    def _check_out_of_bounds(self, bbox: Sequence[float]) -> bool:
        """Check if the bbox is out of _image bounds (consider bound_pad)"""
        left, top, right, bottom = bbox
        return (
            left < self.bound_pad
            or top < self.bound_pad
            or right > self.width - self.bound_pad
            or bottom > self.height - self.bound_pad
        )

    @staticmethod
    def _update_bbox(
        left: float,
        top: float,
        right: float,
        bottom: float,
        left_pad: float,
        top_pad: float,
    ) -> tuple[float, float, float, float]:
        return (left + left_pad, top + top_pad, right + left_pad, bottom + top_pad)

    def _update_bboxes(self, left_pad: float, top_pad: float) -> None:
        """Update all bboxes due to padding"""
        self.image_bboxes = {
            tag: self._update_bbox(*bbox, left_pad, top_pad)
            for tag, bbox in self.image_bboxes.items()
        }

        self.text_bboxes = [
            self._update_bbox(*bbox, left_pad, top_pad) for bbox in self.text_bboxes
        ]

    def _pad_image(
        self,
        bbox: tuple[float, float, float, float],
        xy: tuple[float, float],
        ret_bbox: bool = False,
    ) -> (
        tuple[float, float]
        | tuple[tuple[float, float], tuple[float, float, float, float]]
    ):
        """Pad image to fit bbox"""
        left, top, right, bottom = bbox
        left_pad, top_pad, right_pad, bottom_pad = 0, 0, 0, 0
        x, y = xy

        if left < self.bound_pad:
            left_pad = int(np.ceil(-left)) + self.bound_pad
            x += left_pad

        if top < self.bound_pad:
            top_pad = int(np.ceil(-top)) + self.bound_pad
            y += top_pad

        if right > self.width - self.bound_pad:
            right_pad = int(np.ceil(right - self.width)) + self.bound_pad

        if bottom > self.height - self.bound_pad:
            bottom_pad = int(np.ceil(bottom - self.height)) + self.bound_pad

        # Update bboxes
        self._update_bboxes(left_pad, top_pad)

        # Pad image with zeros
        pad_width = [[top_pad, bottom_pad], [left_pad, right_pad], [0, 0]]
        self._image = np.pad(self._image, pad_width, constant_values=self.pad_values)
        if ret_bbox:
            return (x, y), self._update_bbox(*bbox, left_pad, top_pad)
        else:
            return (x, y)

    def _get_bbox_from_anchor(
        self,
        anchor_pos: str | Sequence[float],
        rel_tag: str,
        bbox_size: Optional[Sequence[float]] = None,
        pad_size: int = 0,
    ) -> tuple[tuple[float, float], str | tuple[float, float, float, float]]:
        """Get anchor_xy and bbox coordinates relative to an existing bbox
        :param anchor_pos: xy pixel position or anchor string composed of:
                           left (l), top (t), right (r), bottom (b), middle (m)
                           inside (i), outside (o), corner (c)
        Anchor locations:

        tlco/ltco   tl(o)        t(o)/mt(o)/tm(o)          tr(o)   trco/rtco
                  +----------------------------------------------+
            lt(o) | lti/tli/        ti/mti/tmi          tri/rti/ | rt(o)
                  | ltc/tlc                             trc/rtc  |
                  |                                              |
            l(o)/ |                                              | r(o)/
           lm(o)/ | li/lmi/mli         m(mi)          ri/mri/rmi | rm(o)/
           ml(o)  |                                              | mr(o)
                  |                                              |
                  | lbi/bli/                            bri/rbi/ |
            lb(o) | lbc/blc         bi/mbi/bmi          brc/rbc  | rb(o)
                  +----------------------------------------------+
        blco/lbco   bl(o)        b(o)/mb(o)/bm(o)          br(o)   brco/rbco

        :param rel_tag: tag of the relative image to use.
                        Default is the entire image.
                        If xy is pixel position, it is relative to
                            the tag image top-left corner.
        :param bbox_size: the size of the added bbox, (width, height).
                          If None, only return bbox_anchor string.
        :param pad_size: the padding size between rel_bbox and added bbox.
                         Ignored if not in_anchor_mode.
        :return anchor_xy: xy pixel position (can be float) of anchor
                           relative to the entire image
        :return bbox: the added box, [left, top, right, bottom].
                      If bbox_size = None, return the bbox_anchor string.
                      Choices:
                          'lt', 'mt', 'rt', 'lm', 'mm', 'rm', 'lb', 'mb', 'rb'
        """
        in_anchor_mode = isinstance(anchor_pos, str)

        if rel_tag == "__whole__":
            rel_bbox = self.bbox_no_pad if in_anchor_mode else self.bbox
        else:
            assert rel_tag in self.image_bboxes, (
                f"Image {rel_tag} not found. "
                f"Existing tags: {list(self.image_bboxes.keys())}"
            )
            rel_bbox = self.image_bboxes[rel_tag]

        if not in_anchor_mode:
            anchor_xy = (anchor_pos[0] + rel_bbox[0], anchor_pos[1] + rel_bbox[1])
            if bbox_size is not None:
                return anchor_xy, anchor_to_bbox(anchor_xy, bbox_size)
            else:
                return anchor_xy, "lt"

        return get_bbox_rel_to_bbox(anchor_pos, rel_bbox, bbox_size, pad_size)

    def add_image(
        self,
        image: np.ndarray,
        anchor_pos: str | Sequence[float] = "r",
        rel_tag: str = "__whole__",
        tag: Optional[str] = None,
        no_pad=False,
        show_vis=False,
    ) -> np.ndarray:
        """Add an image named tag at anchor_pos
        :param image: image to add.
                      If read by cv2, make sure it's ordered as RGB/RGBA.
        :param anchor_pos: xy pixel position or anchor string composed of:
                           left (l), top (t), right (r), bottom (b), middle (m)
                           inside (i), outside (o), corner (c)
        :param rel_tag: tag of the relative image to use.
                        Default is the entire image.
                        If xy is pixel position, it is relative to
                            the tag image top-left corner.
        :param tag: the image tag for future reference.
        :param no_pad: whether to pad between images.
        :param show_vis: whether to show visualization.
        """
        self._check_image_format(image)
        if tag is None:
            tag = f"image_{len(self.image_bboxes) + 1}"
        assert tag not in self.image_bboxes, f"Image {tag} already exists"
        assert tag != "__whole__", 'Tag "__whole__" is reserved for internal usage'

        height, width, channel = image.shape
        # Get anchor_xy coordinates and bbox coordinates
        anchor_xy, image_bbox = self._get_bbox_from_anchor(
            anchor_pos, rel_tag, (width, height), 0 if no_pad else self.bound_pad
        )
        anchor_xy = tuple(np.floor(anchor_xy).astype(int))
        image_bbox = tuple(np.floor(image_bbox).astype(int))

        self.image_bboxes[tag] = image_bbox

        if self._check_out_of_bounds(image_bbox):
            anchor_xy = self._pad_image(image_bbox, anchor_xy)

        image_bbox = self.image_bboxes[tag]
        if channel == 3:
            self._image[
                image_bbox[1] : image_bbox[3], image_bbox[0] : image_bbox[2]
            ] = image.copy()
        elif channel == 4:
            bg = Image.new("RGBA", self.size, (255, 255, 255, 0))
            bg.paste(Image.fromarray(image), image_bbox)
            self._image = np.asarray(
                Image.alpha_composite(self.pil_image.convert("RGBA"), bg).convert("RGB")
            )

        if show_vis:
            self.show_image(window_name=f'Add image "{tag}"')

        return self.image

    def _pad_image_for_text(
        self, text_bbox: tuple[float, float, float, float], xy: tuple[float, float]
    ) -> tuple[
        tuple[float, float],
        tuple[float, float, float, float],
        Image.Image,
        Image.Image,
        ImageDraw.ImageDraw,
    ]:
        """Pad image for drawing text"""
        xy, text_bbox = self._pad_image(text_bbox, xy, ret_bbox=True)  # type: ignore
        # Update due to change in self._image
        img = self.pil_image.convert("RGBA")
        txt_im = Image.new("RGBA", img.size, (255, 255, 255, 0))
        d = ImageDraw.Draw(txt_im)
        return xy, text_bbox, img, txt_im, d

    def draw_text(
        self,
        text: str,
        anchor_pos: str | Sequence[float] = "r",
        rel_tag: str = "__whole__",
        no_pad: bool = False,
        fill: Optional[Sequence[np.uint8]] = None,
        font_path: str = FONT_PATH,
        font_size: Optional[int] = None,
        anchor: Optional[str] = None,
        spacing: int = 4,
        align: Literal["left", "center", "right"] = "left",
        text_bbox_overlap_shift: Literal["left", "right", "up", "down"] = "right",
        draw_text_bbox: bool = False,
        draw_text_bbox_kwargs: Optional[dict] = None,
        show_vis: bool = False,
        ret_bbox: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, tuple[float, float, float, float]]:
        """Draw a multiline text using PIL. Extend image size when necessary
           When drawing the new text and encounter overlaps,
           shift the new text bbox so that there's no overlap.
        :param text: string to be drawn, can contain multilines.
        :param anchor_pos: xy anchor coordinates or anchor string composed of:
                           left (l), top (t), right (r), bottom (b), middle (m)
                           inside (i), outside (o), corner (c)
        :param rel_tag: tag of the image to draw on.
                        '__whole__' means xy is relative to whole image.
                        Otherwise, xy is relative to the tag image.
        :param no_pad: whether to pad between image and text.
        :param fill: color used for drawing text, (R,G,B) or (R,G,B,A)
        :param font_path: path to a TrueType or OpenType font to use
        :param font_size: the requested font size, in points
        :param anchor: The text anchor alignment.
                       Determines the relative location of
                       the anchor to the text.
                       The default alignment is top left.
                       See :ref:`text-anchors` for valid values.
                       https://pillow.readthedocs.io/en/stable/handbook/text-anchors.html#text-anchors
                       "tb" of anchor[1] are not supported for multiline text
        :param spacing: The number of pixels between lines.
        :param align: "left", "center" or "right".
                      Determines the relative alignment of lines.
                      Use the anchor parameter to specify the alignment to xy.
        :param text_bbox_overlap_shift: "left", "right", "up", "down".
                                        How to shift the new text
                                        when there are overlaps.
        :param draw_text_bbox: whether to draw text bbox.
        :param draw_text_bbox_kwargs: kwargs for drawing text bbox:
                                      {'fill', 'outline', 'width'}.
        :param show_vis: whether to show visualization.
        :param ret_bbox: whether to return text bbox
                         (should only be used internally).
        :return out_image: output image, always RGB format [H, W, 3].
        """
        assert text_bbox_overlap_shift in [
            "left",
            "right",
            "up",
            "down",
        ], f"Unknown text_bbox_overlap_shift {text_bbox_overlap_shift}"

        if draw_text_bbox_kwargs is None:
            draw_text_bbox_kwargs = {}

        # use a truetype font
        font = ImageFont.truetype(font_path, 10 if font_size is None else font_size)

        # Get anchor_xy coordinates and bbox_anchor string
        bbox_anchor: str = ""
        anchor_xy, bbox_anchor = self._get_bbox_from_anchor(  # type: ignore
            anchor_pos, rel_tag, pad_size=0 if no_pad else self.text_bbox_pad
        )
        # Convert bbox_anchor to PIL text-anchors format
        bbox_anchor = bbox_anchor.replace("t", "a").replace("b", "d")
        anchor = anchor if anchor is not None else bbox_anchor

        img = self.pil_image.convert("RGBA")
        # make a blank image for text, initialized to transparent text color
        txt_im = Image.new("RGBA", img.size, (255, 255, 255, 0))
        d = ImageDraw.Draw(txt_im)

        def _pad_bbox(
            bbox: tuple[float, float, float, float], pad: int
        ) -> tuple[float, float, float, float]:
            left, top, right, bottom = bbox
            return (left - pad, top - pad, right + pad, bottom + pad)

        # bbox = (left, top, right, bottom)
        text_bbox = d.textbbox(anchor_xy, text, font, anchor, spacing, align)
        text_bbox = _pad_bbox(text_bbox, self.text_bbox_pad)
        # Check and pad image for text_bbox
        if self._check_out_of_bounds(text_bbox):
            anchor_xy, text_bbox, img, txt_im, d = self._pad_image_for_text(
                text_bbox, anchor_xy
            )

        # Check if there's overlap with text_bboxes
        if self.text_bboxes:
            left, top, right, bottom = text_bbox
            anchor_x, anchor_y = anchor_xy
            text_bboxes = np.array(self.text_bboxes)
            overlap_idx = ~(
                (text_bboxes[:, 0] > right)
                | (text_bboxes[:, 2] < left)
                | (text_bboxes[:, 1] > bottom)
                | (text_bboxes[:, 3] < top)
            )
            if np.any(overlap_idx):  # overlaps exist
                if text_bbox_overlap_shift == "right":
                    max_right = np.max(text_bboxes[overlap_idx, 2])
                    anchor_x += max_right - left + 1
                else:
                    raise NotImplementedError(
                        "text_bbox_overlap_shift other than "
                        '"right" is not yet implemented'
                    )

                anchor_xy = (anchor_x, anchor_y)
                text_bbox = d.textbbox(anchor_xy, text, font, anchor, spacing, align)
                text_bbox = _pad_bbox(text_bbox, self.text_bbox_pad)
                # Check and pad image for text_bbox
                if self._check_out_of_bounds(text_bbox):
                    anchor_xy, text_bbox, img, txt_im, d = self._pad_image_for_text(
                        text_bbox, anchor_xy
                    )

        # Actual drawing code
        if draw_text_bbox:
            d.rectangle(text_bbox, **draw_text_bbox_kwargs)
        # Draw the text
        d.text(anchor_xy, text, fill, font, anchor, spacing, align)
        out_im = Image.alpha_composite(img, txt_im).convert("RGB")

        if show_vis:
            short_text = re.sub("[^ a-zA-Z0-9_]", "", text)[:25]
            self.show_image(out_im, window_name=f'Draw text "{short_text}"')

        if ret_bbox:
            return np.asarray(out_im).copy(), text_bbox
        else:
            return np.asarray(out_im).copy()

    def add_text(
        self, text: str, anchor_pos: str | Sequence[float] = "r", **kwargs
    ) -> np.ndarray:
        """Draw and add multiline text
        :param kwargs: kwargs for drawing image bbox:
                       {'fill', 'outline', 'width'}.
                       Can also contain boolean show_vis
        """
        self._image, text_bbox = self.draw_text(
            text, anchor_pos, ret_bbox=True, **kwargs
        )
        self.text_bboxes.append(text_bbox)
        return self.image

    def draw_image_bboxes(
        self, image: Optional[np.ndarray] = None, show_vis: bool = False, **kwargs
    ) -> np.ndarray:
        """Draw all image bboxes
        :param kwargs: kwargs for drawing image bbox:
                       {'fill', 'outline', 'width'}.
        """
        if image is None:
            image = self._image

        img = Image.fromarray(image).convert("RGBA")  # type: ignore
        # make a blank image for bbox, initialized to transparent color
        bbox_im = Image.new("RGBA", img.size, (255, 255, 255, 0))
        d = ImageDraw.Draw(bbox_im)

        for image_bbox in self.image_bboxes.values():
            d.rectangle(image_bbox, **kwargs)
        out_im = Image.alpha_composite(img, bbox_im).convert("RGB")

        if show_vis:
            self.show_image(out_im, window_name="Draw image bboxes")

        return np.asarray(out_im)

    def add_image_bboxes(self, **kwargs) -> np.ndarray:
        """Draw and add all image bboxes
        :param kwargs: kwargs for drawing image bbox:
                       {'fill', 'outline', 'width'}.
                       Can also contain boolean show_vis
        """
        assert "image" not in kwargs, (
            "image found in kwargs when calling add_image_bboxes()"
        )
        self._image = self.draw_image_bboxes(**kwargs)
        return self.image

    def show_image(
        self,
        img: Optional[Image.Image] = None,
        use_cv2: bool = True,
        window_name: str = "Image",
    ) -> None:
        if img is None:
            img = self.pil_image

        if not use_cv2:
            img.show(window_name)
        else:
            cv2.imshow(window_name, np.asarray(img)[..., ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def save_image(self, save_path: str | Path) -> None:
        self.pil_image.save(save_path)

    @property
    def image(self) -> np.ndarray:
        """Read-only view of _image"""
        image = self._image.view()
        image.flags.writeable = False
        return image

    @property
    def pil_image(self) -> Image.Image:
        """PIL image"""
        return Image.fromarray(self._image)

    @property
    def cv2_image(self) -> np.ndarray:
        """cv2 BGR image copy of _image"""
        return self._image[..., ::-1]  # RGB to BGR

    @property
    def image_no_pad(self) -> np.ndarray:
        """Read-only view of _image without boundary padding"""
        image = self._image[
            self.bound_pad : -self.bound_pad, self.bound_pad : -self.bound_pad
        ].view()
        image.flags.writeable = False
        return image

    @property
    def pil_image_no_pad(self) -> Image.Image:
        """PIL image"""
        return Image.fromarray(self.image_no_pad)

    @property
    def cv2_image_no_pad(self) -> np.ndarray:
        """cv2 BGR image copy of _image without boundary padding"""
        return self.image_no_pad[..., ::-1]  # RGB to BGR

    @property
    def height(self) -> int:
        return self._image.shape[0]

    @property
    def width(self) -> int:
        return self._image.shape[1]

    @property
    def channel(self) -> int:
        return self._image.shape[2]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._image.shape  # type: ignore

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def dtype(self) -> np.dtype:
        return self._image.dtype

    @property
    def color_format(self) -> str:
        if self.channel == 3:
            return "RGB"
        elif self.channel == 4:
            return "RGBA"
        else:
            raise NotImplementedError(
                f"Image channel {self.channel} is not yet implemented"
            )

    @property
    def pad_values(self) -> np.ndarray:
        # TODO: check if dtype=object causes issues
        if self.color_format == "RGB":
            return np.asarray([[self.bg_color] * 2] * 2 + [[0, 0]], dtype=object)
        elif self.color_format == "RGBA":
            return np.asarray(
                [[self.bg_color + [255]] * 2] * 2 + [[0, 0]], dtype=object
            )
        else:
            raise NotImplementedError(
                f"Image channel {self.channel} is not yet implemented"
            )

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Image bbox: (left, top, right, bottom)"""
        return (0, 0, self.width, self.height)

    @property
    def bbox_no_pad(self) -> tuple[int, int, int, int]:
        """Image bbox without boundary padding: (left, top, right, bottom)"""
        if self._image.size > 0:
            return (
                self.bound_pad,
                self.bound_pad,
                self.width - self.bound_pad,
                self.height - self.bound_pad,
            )
        else:
            return self.bbox
