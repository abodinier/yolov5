import os
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import (
    check_img_size,
    non_max_suppression, 
    scale_coords,
)
from utils.torch_utils import select_device

class Detector:
    def __init__(self, config, weights):
        """Init method that store the config dict and loads the model

        Args:
            config (dict): Configuration dict.
            weights (pathlib.Path): Path to the torch model weights (.pt)
        """
        self.config = config
        self.weights_path = weights
        
        # Load model
        self.device = select_device('cpu')
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False)
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
    
    @torch.no_grad()
    def predict(self, img):
        """Get prediction from the object detection model

        Args:
            img (str): Path to the image on which to perform the detection.

        Returns:
            list: List of the detected bounding boxes in dict format:
            [{'class': 0, 'confidence': .8, 'x': 0.3, 'y': 0.2', 'w': 0.1, 'h': 0.1}, ...]
        """
        imgsz = check_img_size(self.config["imgsz"], s=self.stride)  # check image size

        # Dataloader
        dataset = LoadImages(img, img_size=imgsz, stride=self.stride, auto=self.pt)
        
        # Run inference
        self.model.warmup(imgsz=(1, 3, *imgsz), half=False)  # warmup
        for path, im, im0s, _, _ in dataset:
            im = torch.from_numpy(im).to(self.device)
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            pred = self.model(im, augment=False, visualize=False)

            # NMS
            pred = non_max_suppression(pred, self.config["conf_thres"], self.config["iou_thres"], self.config["classes"], self.config["agnostic_nms"], max_det=self.config["max_det"])

            # Process predictions
            for i, det in enumerate(pred):  # per image
                results = list()
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        results.append(
                            {
                                "left": xyxy[0].item(),
                                "bottom": xyxy[1].item(),
                                "right": xyxy[2].item(),
                                "top": xyxy[3].item(),
                                "class": cls.item(),
                                "confidence": conf.item()
                            }
                        )
                return results
