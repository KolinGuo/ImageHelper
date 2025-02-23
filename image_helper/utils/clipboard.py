import io
import shutil
import subprocess

from PIL.Image import Image


def copy_to_clipboard(img: Image):
    """Copy image to system clipboard in PNG format using xclip"""
    if shutil.which("xclip") is None:
        raise RuntimeError(
            "xclip is not installed. Install it using `sudo apt-get install xclip`"
        )

    buffer = io.BytesIO()
    img.save(buffer, "PNG")
    buffer.seek(0)

    process = subprocess.Popen(
        "xclip -selection clipboard -t image/png -i", shell=True, stdin=subprocess.PIPE
    )
    process.communicate(input=buffer.getvalue())  # type: ignore
