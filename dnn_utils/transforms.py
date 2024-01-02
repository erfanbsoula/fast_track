import numpy as np
import cv2

class LetterBox:
    """Resize image and padding for detection"""

    def __init__(self, shape, new_shape=(640, 640), fill_value=(0, 0, 0),
                 scaleFill=False, scaleup=True, center=True):
        """Initialize LetterBox object with specific parameters."""

        self.shape = shape
        self.new_shape = new_shape
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.center = center  # Put the image in the middle or top-left
        self.fill_value = fill_value

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
        dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding

        if self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[0], new_shape[1])

        if self.center: # divide padding into 2 sides
            dw, dh =  dw/2, dh/2

        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))

        self.new_unpad = new_unpad
        self.top, self.bottom = top, bottom
        self.left, self.right = left, right

        self.unboxing_bias = np.array([0, top, 0, 0])
        self.unboxing_ratio = np.array([
            shape[0] / new_unpad[0],
            shape[1] / new_unpad[1],
            shape[0] / new_unpad[0],
            shape[1] / new_unpad[1],
        ])

    def __call__(self, img):
        """Return transformed image"""

        if self.shape != self.new_unpad:  # resize
            img = cv2.resize(img, self.new_unpad, interpolation=cv2.INTER_LINEAR)

        img = cv2.copyMakeBorder(
            img, self.top, self.bottom, self.left, self.right,
            cv2.BORDER_CONSTANT, value=self.fill_value
        )

        return img

    def unbox(self, tlwh):
        """Convert detections to positions of original image"""
    
        if len(tlwh) == 0: # there are no detections
            return tlwh

        return (tlwh - self.unboxing_bias) * self.unboxing_ratio
