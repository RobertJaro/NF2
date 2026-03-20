from __future__ import annotations

import os

import torch


class NF2Checkpoint:
    def __init__(self, path, device=None):
        self.path = path
        self.device = device
        self.state = torch.load(path, map_location=device, weights_only=False)

    @property
    def geometry(self):
        return self.state["data"]["type"]

    @property
    def metadata(self):
        return {
            "checkpoint_path": self.path,
            "checkpoint_name": os.path.basename(self.path),
            "geometry": self.geometry,
            "config": self.state.get("config", {}),
            "data": self.state.get("data", {}),
        }
