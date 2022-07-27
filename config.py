#!/usr/bin/env python3


"""
Tuple     : ConvLayer    (out_channels, kernel_size, stride)
Int       : ResBlock     num_repeats
String(S) : prediction   "P"
String(U) : Upsampling   "U"
"""

model_config = [
    (32, 3, 1),
    (64, 3, 2),
    1,
    (128, 3, 2),
    2,
    (256, 3, 2),
    8,
    (512, 3, 2),
    8,
    (1024, 3, 2),
    4, # end of darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "P",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 1, 1),
    "P",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "P"
]
