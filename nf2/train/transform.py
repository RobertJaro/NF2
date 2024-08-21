import torch
from torch import nn

from nf2.train.model import GenericModel


class BaseTransformModel(nn.Module):

    def __init__(self, ds_id, tensor_ids):
        super().__init__()
        self.ds_id = ds_id if isinstance(ds_id, list) else [ds_id]
        self.tensor_ids = tensor_ids if isinstance(tensor_ids, list) else [tensor_ids]


class AzimuthTransformModel(BaseTransformModel):

    def __init__(self, **kwargs):
        super().__init__(tensor_ids=['coords', 'b_true'], **kwargs)
        self.model = GenericModel(in_coords=3, out_coords=1, n_layers=8, dim=128, encoding='gaussian')

    def forward(self, batch, **kwargs):
        coords = batch['coords']

        # flip probability between 0 and 1
        flip = torch.sigmoid(self.model(coords))

        return {'flip': flip}


class HeightTransformModel(BaseTransformModel):

    def __init__(self, **kwargs):
        super().__init__(tensor_ids=['coords', 'height_range', 'b_true'], **kwargs)
        self.mapping_module = GenericModel(in_coords=3, out_coords=1, n_layers=4, dim=128, encoding='gaussian')

    def forward(self, batch):
        coords = batch['coords']
        height_range = batch['height_range']

        z_coords = self.mapping_module(coords)
        z_coords = torch.sigmoid(z_coords) * (height_range[:, 1:2] - height_range[:, 0:1]) + height_range[:, 0:1]
        transformed_coords = torch.cat([coords[:, :2], z_coords], -1)

        return {'coords': transformed_coords, 'original_coords': coords}
