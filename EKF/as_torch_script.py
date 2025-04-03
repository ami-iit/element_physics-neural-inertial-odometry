import numpy as np
import torch
from utils.logging import logging

class NetworkTorchScript:
    """
    Load a torchscript file to reconstruct the network with trained weights.
    """
    def __init__(self, model_path, force_cpu=False):
        # load the pretrained network
        logging.info(f"Loading pre-trained network from {model_path}...")
        if not torch.cuda.is_available() or force_cpu:
            torch.init_num_threads()
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            self.device = torch.device("cpu")
            self.net = torch.jit.load(model_path, map_location="cpu")
        else:
            self.device = torch.device("cuda:0")
            self.net = torch.jit.load(model_path, map_location=self.device)
        self.net.to(self.device)
        self.net.eval()
        logging.info(f"Load model {model_path} to device {self.device}.")

    def get_displacement_meas_from_nn(self):
        return
    