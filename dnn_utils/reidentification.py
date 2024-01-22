from deepsparse import Engine
import numpy as np
import cv2


class FeatureExtractor:

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path, batch_size=2,
                 model_input_size=(128, 256)):

        self.engine = Engine(
            model=model_path,
            num_cores=2,
            batch_size=batch_size
        )

        self.batch_size = batch_size
        self.model_input_size = model_input_size

    def __call__(self, frames):
        
        if len(frames) == 0:
            return []

        imgs = []

        for img in frames:
            imgs.append(
                cv2.resize(
                    img, self.model_input_size,
                    interpolation=cv2.INTER_LINEAR
                )
            )

        imgs = np.array(imgs, dtype=np.float32)
        imgs = imgs / 255.0
        imgs = (imgs - self.mean) / self.std
        imgs = imgs.transpose((0,3,1,2)) # bwhc -> bcwh

        initial_batch_size = imgs.shape[0]
        if initial_batch_size % self.batch_size != 0:
            pad_size = self.batch_size - initial_batch_size % self.batch_size
            imgs = np.pad(imgs, ((0, pad_size), 0, 0, 0), constant_values=0)
    
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)

        outputs = []

        for i in range(0, imgs.shape[0], self.batch_size):
            outputs.extend(
                self.engine([imgs[i:i+self.batch_size]])[0]
            )

        return outputs[:initial_batch_size]