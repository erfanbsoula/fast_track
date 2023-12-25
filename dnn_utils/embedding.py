import torch
from deepsparse import Engine
from onnx2torch import convert
import numpy as np
import cv2


class FeatureExtractor:

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path, use_pytorch,
                 model_input_size=(128, 256)):

        self.use_pytorch = use_pytorch

        if use_pytorch: # convert onnx to pytorch model
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.device = torch.device(self.device)
            engine = convert(model_path).to(self.device)
            self.engine = engine.eval()

        else: # use the default deepsparse engine
            self.engine = Engine(model=model_path, num_cores=1)

        self.model_input_size = model_input_size

    def __call__(self, img):

        img = cv2.resize(img, self.model_input_size,
                         interpolation=cv2.INTER_LINEAR)

        img = img / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose((2,0,1)) # whc -> cwh
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img, dtype=np.float32)

        if self.use_pytorch:
            with torch.no_grad():
                outputs = self.engine(torch.tensor(img, device=self.device))
                outputs = outputs.cpu().numpy()[0]

        else:
            outputs = self.engine([img])[0][0]

        return outputs