"""Generate QR Code from given data"""

from pathlib import Path
from typing import Optional

import qrcode
from PIL.Image import Image
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.colormasks import RadialGradiantColorMask
from qrcode.image.styles.moduledrawers.pil import RoundedModuleDrawer


def generate_qr_code(
    data,
    save_path: Optional[str | Path] = None,
    embeded_image_path: Optional[str | Path] = None,
    back_color: tuple[int, int, int] = (255, 255, 255),
    center_color: tuple[int, int, int] = (0, 0, 0),
    edge_color: tuple[int, int, int] = (0, 0, 0),
    error_correction=qrcode.constants.ERROR_CORRECT_H,  # type: ignore
    box_size: int = 10,
    border: int = 4,
) -> Image:
    """
    Generate a QR code from data with given parameters

    :param data: text data to be encoded
    :param save_path: if not None, path to save the image
    :param embeded_image_path: path to the image file
    :param back_color: background color in RadialGradiantColorMask, tuple of RGB colors
    :param center_color: center color in RadialGradiantColorMask, tuple of RGB colors
    :param edge_color: edge color in RadialGradiantColorMask, tuple of RGB colors
    :param error_correction: error correction level, one of
        ERROR_CORRECT_L (<7% error),
        ERROR_CORRECT_M (<15% error),
        ERROR_CORRECT_Q (<25% error),
        ERROR_CORRECT_H (<30% error, default)
    :param box_size: how many pixels each "box" of the QR code is
    :param border: how many boxes thick the border should be
        (the default of 4 boxes is the minimum according to the specs)
    """

    qr = qrcode.QRCode(
        error_correction=error_correction, box_size=box_size, border=border
    )
    qr.add_data(data)

    qr_img = qr.make_image(
        image_factory=StyledPilImage,
        embeded_image_path=embeded_image_path,
        module_drawer=RoundedModuleDrawer(),
        color_mask=RadialGradiantColorMask(
            back_color=back_color,
            center_color=center_color,
            edge_color=edge_color,
        ),
    )
    if save_path is not None:
        qr_img.save(save_path)

    return qr_img._img


if __name__ == "__main__":
    img = generate_qr_code("""Hello world! This is a sample QR code!""")
    img.show()
