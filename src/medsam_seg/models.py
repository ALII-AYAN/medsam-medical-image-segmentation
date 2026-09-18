"""Model architectures.

TensorFlow is imported lazily inside the builders so that the rest of the
package (data loading, metrics, config) stays importable - and testable -
without a TensorFlow install.
"""

from __future__ import annotations

from typing import Callable

ARCHITECTURES = ("unet", "deeplabv3", "segnet")


def build_unet(input_shape=(256, 256, 3), filters=(32, 64, 128, 256),
               dropout: float = 0.2, depth: int = 3):
    """Enhanced U-Net: encoder, bottleneck, decoder with skip connections.

    Batch normalisation after every convolution and dropout on the down-sampling
    path. With the default filters this is ~1.95M parameters.
    """
    from tensorflow.keras.layers import (
        BatchNormalization, Concatenate, Conv2D, Dropout, Input, MaxPooling2D,
        UpSampling2D,
    )
    from tensorflow.keras.models import Model

    f1, f2, f3, f4 = filters
    inputs = Input(input_shape)

    def block(tensor, channels, name):
        tensor = Conv2D(channels, (3, 3), activation="relu", padding="same",
                        name=f"{name}_conv1")(tensor)
        tensor = BatchNormalization(name=f"{name}_bn1")(tensor)
        tensor = Conv2D(channels, (3, 3), activation="relu", padding="same",
                        name=f"{name}_conv2")(tensor)
        tensor = BatchNormalization(name=f"{name}_bn2")(tensor)
        return tensor

    # Encoder
    x = inputs
    skips = []
    levels = [f1, f2, f3][:depth]
    for index, channels in enumerate(levels, start=1):
        x = block(x, channels, f"enc{index}")
        skips.append(x)
        x = MaxPooling2D((2, 2), name=f"pool{index}")(x)
        x = Dropout(min(dropout, 0.1) if index == 1 else dropout,
                    name=f"drop{index}")(x)

    # Bottleneck
    x = block(x, f4, "bottleneck")

    # Decoder
    for index, (channels, skip) in enumerate(zip(reversed(levels), reversed(skips)), start=1):
        x = UpSampling2D((2, 2), name=f"up{index}")(x)
        x = Concatenate(name=f"concat{index}")([x, skip])
        x = block(x, channels, f"dec{index}")
        if index < len(levels):
            x = Dropout(dropout, name=f"ddrop{index}")(x)

    from tensorflow.keras.layers import Conv2D as _Conv2D
    outputs = _Conv2D(1, (1, 1), activation="sigmoid", name="mask")(x)
    return Model(inputs=inputs, outputs=outputs, name="unet")


def build_deeplabv3(input_shape=(256, 256, 3), filters=(32, 64, 128, 256),
                    dropout: float = 0.2, depth: int = 3):
    """DeepLabV3-style head: atrous spatial pyramid pooling on the bottleneck.

    Parallel dilated convolutions capture context at several scales without
    losing resolution, which is what ASPP is for.
    """
    from tensorflow.keras.layers import (
        BatchNormalization, Concatenate, Conv2D, Dropout, Input, MaxPooling2D,
        UpSampling2D,
    )
    from tensorflow.keras.models import Model

    f1, f2, f3, f4 = filters
    inputs = Input(input_shape)

    def block(tensor, channels, name, dilation=1):
        tensor = Conv2D(channels, (3, 3), activation="relu", padding="same",
                        dilation_rate=dilation, name=f"{name}_conv")(tensor)
        tensor = BatchNormalization(name=f"{name}_bn")(tensor)
        return tensor

    x = block(inputs, f1, "enc1")
    x = MaxPooling2D((2, 2))(block(x, f1, "enc1b"))
    x = Dropout(dropout)(x)
    x = MaxPooling2D((2, 2))(block(x, f2, "enc2"))
    x = Dropout(dropout)(x)
    x = block(x, f3, "enc3")

    # ASPP: same features, several receptive-field sizes.
    tower1 = block(x, f4, "aspp1", dilation=1)
    tower2 = block(x, f4, "aspp2", dilation=2)
    tower3 = block(x, f4, "aspp3", dilation=4)
    tower4 = block(x, f4, "aspp4", dilation=8)
    x = Concatenate(name="aspp_concat")([tower1, tower2, tower3, tower4])
    x = Conv2D(f4, (1, 1), activation="relu", padding="same", name="aspp_reduce")(x)
    x = BatchNormalization(name="aspp_reduce_bn")(x)

    x = UpSampling2D((4, 4), name="up_to_input")(x)
    x = block(x, f1, "dec")
    outputs = Conv2D(1, (1, 1), activation="sigmoid", name="mask")(x)
    return Model(inputs=inputs, outputs=outputs, name="deeplabv3")


def build_segnet(input_shape=(256, 256, 3), filters=(32, 64, 128, 256),
                 dropout: float = 0.2, depth: int = 3):
    """SegNet-style encoder-decoder.

    The decoder upsamples and convolves without concatenating encoder features,
    which is the memory-saving trade-off SegNet makes.
    """
    from tensorflow.keras.layers import (
        BatchNormalization, Conv2D, Dropout, Input, MaxPooling2D, UpSampling2D,
    )
    from tensorflow.keras.models import Model

    f1, f2, f3, f4 = filters
    inputs = Input(input_shape)

    def block(tensor, channels, name):
        tensor = Conv2D(channels, (3, 3), activation="relu", padding="same",
                        name=f"{name}_conv1")(tensor)
        tensor = BatchNormalization(name=f"{name}_bn1")(tensor)
        tensor = Conv2D(channels, (3, 3), activation="relu", padding="same",
                        name=f"{name}_conv2")(tensor)
        tensor = BatchNormalization(name=f"{name}_bn2")(tensor)
        return tensor

    x = inputs
    levels = [f1, f2, f3][:depth]
    for index, channels in enumerate(levels, start=1):
        x = block(x, channels, f"enc{index}")
        x = MaxPooling2D((2, 2), name=f"pool{index}")(x)
        x = Dropout(dropout, name=f"drop{index}")(x)

    x = block(x, f4, "bottleneck")

    for index, channels in enumerate(reversed(levels), start=1):
        x = UpSampling2D((2, 2), name=f"up{index}")(x)
        x = block(x, channels, f"dec{index}")

    outputs = Conv2D(1, (1, 1), activation="sigmoid", name="mask")(x)
    return Model(inputs=inputs, outputs=outputs, name="segnet")


BUILDERS: dict[str, Callable] = {
    "unet": build_unet,
    "deeplabv3": build_deeplabv3,
    "segnet": build_segnet,
}


def build_model(arch: str = "unet", input_shape=(256, 256, 3),
                filters=(32, 64, 128, 256), dropout: float = 0.2, depth: int = 3):
    """Build one of the supported architectures by name."""
    if arch not in BUILDERS:
        raise ValueError(f"unknown architecture '{arch}'; choose from {list(BUILDERS)}")
    return BUILDERS[arch](input_shape=input_shape, filters=filters,
                          dropout=dropout, depth=depth)


# ---------------------------------------------------------------- param maths


def _conv_params(k: int, cin: int, cout: int) -> int:
    return k * k * cin * cout + cout


def _bn_params(channels: int) -> int:
    return 4 * channels  # gamma + beta + 2 moving statistics


def count_params_analytic(arch: str = "unet", filters=(32, 64, 128, 256),
                          depth: int = 3, input_channels: int = 3) -> int:
    """Parameter count from the layer maths, without loading TensorFlow.

    Useful as a fast sanity check in tests: the published U-Net configuration
    is 1,952,513 parameters, so a refactor that changes the shape silently will
    fail here rather than at training time.
    """
    f1, f2, f3, f4 = filters
    levels = [f1, f2, f3][:depth]

    def block(cin, cout, name):  # noqa: ARG001 - name kept for symmetry
        return (_conv_params(3, cin, cout) + _bn_params(cout)
                + _conv_params(3, cout, cout) + _bn_params(cout))

    if arch in ("unet", "segnet"):
        total = 0
        previous = input_channels
        skips = []
        for channels in levels:
            total += block(previous, channels, "enc")
            skips.append(channels)
            previous = channels
        total += block(previous, f4, "bottleneck")
        previous = f4
        for index, channels in enumerate(reversed(levels)):
            if arch == "unet":
                cin = previous + skips[-(index + 1)]  # skip concatenation
            else:
                cin = previous
            total += block(cin, channels, "dec")
            previous = channels
        return total + _conv_params(1, previous, 1)

    if arch == "deeplabv3":
        total = block(input_channels, f1, "enc1")
        total += block(f1, f1, "enc1b")
        total += block(f1, f2, "enc2")
        total += block(f2, f3, "enc3")
        # Four parallel ASPP towers at different dilation rates.
        total += 4 * (_conv_params(3, f3, f4) + _bn_params(f4))
        total += _conv_params(1, 4 * f4, f4) + _bn_params(f4)
        total += _conv_params(3, f4, f1) + _bn_params(f1)
        return total + _conv_params(1, f1, 1)

    raise ValueError(f"unknown architecture '{arch}'")
