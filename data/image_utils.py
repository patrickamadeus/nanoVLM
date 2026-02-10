import io
import os
from typing import Any

import numpy as np
from PIL import Image


def _normalize_numpy_image(image: np.ndarray) -> np.ndarray:
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported numpy image shape: {image.shape}.")

    arr = image
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            max_value = float(np.nanmax(arr)) if arr.size > 0 else 0.0
            if max_value <= 1.0:
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            else:
                arr = np.clip(arr, 0.0, 255.0)
        else:
            arr = np.clip(arr, 0, 255)
        arr = arr.astype(np.uint8)
    return arr


def coerce_image_to_pil(image: Any) -> Image.Image:
    """
    Convert common image payload formats into an RGB PIL image.

    Supported inputs:
    - PIL Image
    - numpy.ndarray
    - path string / os.PathLike
    - bytes / bytearray / memoryview
    - dict payload with keys such as `bytes`, `path`, `array`, or `image`
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, np.ndarray):
        return Image.fromarray(_normalize_numpy_image(image)).convert("RGB")

    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as img:
            return img.convert("RGB")

    if isinstance(image, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(image))) as img:
            return img.convert("RGB")

    if isinstance(image, dict):
        if image.get("image") is not None:
            return coerce_image_to_pil(image["image"])
        if image.get("array") is not None:
            return coerce_image_to_pil(image["array"])
        if image.get("bytes") is not None:
            return coerce_image_to_pil(image["bytes"])
        if image.get("path"):
            return coerce_image_to_pil(image["path"])
        raise ValueError(
            "Unsupported image dict payload. Expected one of keys: "
            "`image`, `array`, `bytes`, `path`."
        )

    raise ValueError(
        f"Unsupported image type: {type(image)}. Expected PIL Image, numpy array, path, bytes, or image dict payload."
    )
