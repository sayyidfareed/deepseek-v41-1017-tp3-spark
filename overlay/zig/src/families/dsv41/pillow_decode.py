# Modified for the DeepSeek 1017 TP3 integration; upstream copyrights retained. See CREDITS.md and provenance.json.
"""Embedded ordinary-image decoder. Never receives a secure request."""
import io
import struct
import warnings
from PIL import Image

Image.MAX_IMAGE_PIXELS = 67108864
# Resolve installed plugins at startup, before any request or secure phase.
Image.init()
VERSION = Image.__version__.encode()


def decode(data):
    # Same first-frame/convert("RGB") semantics as the existing child decoder.
    # No request data is placed in module globals, files or error messages.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with io.BytesIO(data) as source_bytes:
                with Image.open(source_bytes) as source:
                    with source.convert("RGB") as image:
                        pixels = image.tobytes()
                        return struct.pack("<BQQQ", 0, image.width, image.height,
                                           len(pixels)) + pixels
    except Image.DecompressionBombError:
        return b"\x01"
    except MemoryError:
        raise
    except Exception:
        return b"\x02"
