import numpy as np
import onnxruntime as ort
import cv2

class LivenessONNX:
    def __init__(self, onnx_path: str):
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def preprocess(self, face_bgr):
        face = cv2.resize(face_bgr, (80, 80))
        x = face.astype(np.float32) / 255.0
        x = np.transpose(x, (2,0,1))[None, ...]  # NCHW
        return x

    def score(self, face_bgr) -> float:
        x = self.preprocess(face_bgr)
        out = self.sess.run(None, {self.input_name: x})[0]
        # assume output is [N,2] => [spoof_prob, live_prob] or similar
        live_prob = float(out[0][-1])
        return live_prob
