"""Architecture parameter maths - verified without TensorFlow installed.

TensorFlow cannot be imported in CI here, so the builders are not exercised.
These tests guard the layer arithmetic that determines the published model size.
"""

import pytest

from medsam_seg.models import ARCHITECTURES, BUILDERS, count_params_analytic


def test_published_unet_size():
    """The report states 1.95M parameters; the architecture must match it."""
    assert count_params_analytic("unet") == 1_952_513
    assert count_params_analytic("unet") == pytest.approx(1_950_000, rel=0.002)


def test_unet_is_the_largest_of_the_three():
    """U-Net concatenates skip features, so it carries the most parameters."""
    sizes = {a: count_params_analytic(a) for a in ARCHITECTURES}
    assert sizes["unet"] > sizes["deeplabv3"] > sizes["segnet"]


def test_all_architectures_are_registered():
    assert set(BUILDERS) == set(ARCHITECTURES)


def test_narrower_filters_mean_fewer_parameters():
    big = count_params_analytic("unet", filters=(32, 64, 128, 256))
    small = count_params_analytic("unet", filters=(16, 32, 64, 128))
    assert small < big
    # A quarter of the filters is roughly a quarter of the parameters.
    assert small == pytest.approx(big / 4, rel=0.1)


def test_shallower_network_is_smaller():
    assert count_params_analytic("unet", depth=2) < count_params_analytic("unet", depth=3)


def test_unknown_architecture_raises():
    with pytest.raises(ValueError, match="unknown architecture"):
        count_params_analytic("maskrcnn")


def test_hand_computed_two_stage_unet():
    """Independent check: 16 filters, depth 2, 1 input channel."""

    def conv(k, cin, cout):
        return k * k * cin * cout + cout

    def bn(c):
        return 4 * c

    def block(cin, cout):
        return conv(3, cin, cout) + bn(cout) + conv(3, cout, cout) + bn(cout)

    # enc(1->16), enc(16->32), bottleneck(32->64), dec(64+32->32), dec(32+16->16), out(16->1)
    expected = (
        block(1, 16)
        + block(16, 32)
        + block(32, 64)
        + block(64 + 32, 32)
        + block(32 + 16, 16)
        + conv(1, 16, 1)
    )
    assert (
        count_params_analytic("unet", filters=(16, 32, 64, 64), depth=2, input_channels=1)
        == expected
    )
