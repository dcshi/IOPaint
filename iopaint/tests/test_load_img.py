import numpy as np

from iopaint.helper import load_img
from iopaint.helper import (
    alpha_channel_requires_inpaint,
    alpha_channel_is_binary_like,
    alpha_channel_to_rgb,
    rgb_to_alpha_channel,
    postprocess_alpha_channel,
    inpaint_binary_like_alpha,
)
from iopaint.tests.utils import current_dir

png_img_p = current_dir / "image.png"
jpg_img_p = current_dir / "bunny.jpeg"


def test_load_png_image():
    with open(png_img_p, "rb") as f:
        np_img, alpha_channel = load_img(f.read())
    assert np_img.shape == (256, 256, 3)
    assert alpha_channel.shape == (256, 256)


def test_load_jpg_image():
    with open(jpg_img_p, "rb") as f:
        np_img, alpha_channel = load_img(f.read())
    assert np_img.shape == (394, 448, 3)
    assert alpha_channel is None


def test_alpha_channel_requires_inpaint():
    assert alpha_channel_requires_inpaint(None) is False
    assert alpha_channel_requires_inpaint(np.full((2, 2), 255, dtype=np.uint8)) is False
    assert alpha_channel_requires_inpaint(np.array([[255, 128]], dtype=np.uint8)) is True


def test_alpha_channel_roundtrip_helpers():
    alpha_channel = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    alpha_rgb = alpha_channel_to_rgb(alpha_channel)
    assert alpha_rgb.shape == (2, 2, 3)
    restored_alpha = rgb_to_alpha_channel(alpha_rgb)
    assert np.array_equal(restored_alpha, alpha_channel)


def test_alpha_channel_is_binary_like():
    assert alpha_channel_is_binary_like(np.array([[0, 255]], dtype=np.uint8)) is True
    assert alpha_channel_is_binary_like(np.array([[3, 252]], dtype=np.uint8)) is True
    assert alpha_channel_is_binary_like(np.array([[32, 255]], dtype=np.uint8)) is False


def test_postprocess_alpha_channel():
    alpha_channel = np.array([[10, 100, 240]], dtype=np.uint8)
    mask = np.array([[255, 255, 255]], dtype=np.uint8)
    processed = postprocess_alpha_channel(alpha_channel, mask)
    assert np.array_equal(processed, np.array([[0, 100, 255]], dtype=np.uint8))


def test_inpaint_binary_like_alpha():
    alpha_channel = np.array(
        [
            [0, 0, 0],
            [0, 255, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    mask = np.array(
        [
            [0, 0, 0],
            [0, 255, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    processed = inpaint_binary_like_alpha(alpha_channel, mask)
    assert processed[1, 1] == 0
