import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from image_helper import ImageHelper
import numpy as np
from PIL import Image

from utils import get_func_name

IMAGE_PATH = Path(__file__).resolve().parent / "images"


class TestRGBAImage:
    @property
    def image_dir(self):
        return IMAGE_PATH / self.__class__.__name__

    def test_add_rgba(self):
        h = 100
        image_helper = ImageHelper()
        image_helper.add_image(
            np.tile([0, 255, 0], [h, h, 1]).astype(np.uint8), (0, 0), tag="green"
        )
        image_helper.add_image(
            np.tile([255, 0, 0], [h, h, 1]).astype(np.uint8), [200, 200], tag="red"
        )
        image_helper.add_multiline_text(
            "green zasqssl\nred\n tt", [20-4, 20],
            fill=(255, 255, 255, 128), font_size=10,
            anchor='ra', draw_text_bbox=True
        )
        image_helper.add_image(
            np.asarray(Image.open(self.image_dir / "img_rgba.png")),
            [0, 0], tag="rgba"
        )

        np.testing.assert_allclose(
            image_helper.image,
            np.asarray(Image.open(self.image_dir / f"{get_func_name()}.png"))
        )
