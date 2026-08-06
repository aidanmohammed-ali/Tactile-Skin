from __future__ import annotations

import sys
from typing import Any

from sensor_msgs.msg import Image


def bgr_to_image_msg(frame: Any, stamp: Any, frame_id: str) -> Image:
    image = Image()
    image.header.stamp = stamp
    image.header.frame_id = frame_id
    image.height, image.width = frame.shape[:2]
    image.encoding = "bgr8"
    image.is_bigendian = sys.byteorder == "big"
    image.step = int(frame.strides[0])
    image.data = frame.tobytes()
    return image
