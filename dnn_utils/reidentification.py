import numpy as np
import cv2

class FeatureExtractor:

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, engine, model_name, model_path,
                 model_image_size=(256, 128), # (Height, Width)
                 device='cpu', batch_size=2):

        assert isinstance(engine, str)
        assert engine in ['torchreid', 'deepsparse']
        assert device in ['cpu', 'cuda']

        self.model_name = model_name
        self.model_path = model_path
        # convert model_image_size to (Width, Height) format
        self.model_image_size = model_image_size[::-1]
        self.device = device
        self.batch_size = batch_size

        if engine == 'torchreid':
            self.init_torchreid()

        elif engine == 'deepsparse':
            if device != 'cpu':
                raise Exception('deepsparse only aupports cpu.')

            self.init_deepsparse()

        else: raise Exception("unknown engine!")

        self.engine_type = engine


    def __call__(self, frames):

        return self.process(frames)


    def init_torchreid(self):
        
        from torchreid.utils import FeatureExtractor as TorchreidFE

        self.engine = TorchreidFE(
            self.model_name, self.model_path,
            image_size = self.model_image_size,
            device = self.device
        )

        self.process = self.run_torchreid


    def run_torchreid(self, frames):

        return self.engine(frames).numpy()


    def init_deepsparse(self):
        
        from deepsparse import Engine

        self.engine = Engine(
            model=self.model_path,
            num_cores=2,
            batch_size=self.batch_size
        )

        self.process = self.run_deepsparse


    def run_deepsparse(self, frames):
        
        if len(frames) == 0:
            return []

        imgs = []

        for img in frames:
            imgs.append(
                cv2.resize(
                    img, self.model_image_size,
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
            pad_size = ((0, pad_size), (0,0), (0,0), (0,0))
            imgs = np.pad(imgs, pad_size, constant_values=0)
    
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)

        outputs = []

        for i in range(0, imgs.shape[0], self.batch_size):
            outputs.extend(
                self.engine([imgs[i:i+self.batch_size]])[0]
            )

        return outputs[:initial_batch_size]