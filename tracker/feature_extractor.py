import numpy as np
import onnxruntime as rt
import cv2


def preprocess(img):
    img = cv2.resize(img, (64, 128))
    img = np.float32(img)
    img = img / 255.0
    img = img.transpose(2, 1, 0)
    img = np.expand_dims(img, axis=0)

    return img


class Extractor:
    def __init__(self, model_path) -> None:
        self.onnx_model = rt.InferenceSession(model_path)
        self.input_names = ["input_1"]
        self.output_names = ["output_1"]

    def __call__(self, im_crops):
        embs = []
        for im in im_crops:
            inp = preprocess(im)
            emb = self.onnx_model.run(self.output_names, {self.input_names[0]: inp})[0]
            embs.append(emb.squeeze())
        embs = np.array(np.stack(embs), dtype=np.float32)
        return embs