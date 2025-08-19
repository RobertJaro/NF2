import torch
from torch import nn

from nf2.train.model import GenericModel, jacobian
from nf2.train.siren import SirenModel


class BaseTransformModel(nn.Module):

    def __init__(self, ds_id, tensor_ids):
        super().__init__()
        self.ds_id = ds_id if isinstance(ds_id, list) else [ds_id]
        self.tensor_ids = tensor_ids if isinstance(tensor_ids, list) else [tensor_ids]


class AzimuthTransformModel(BaseTransformModel):

    def __init__(self, **kwargs):
        super().__init__(tensor_ids=['coords', 'b_true'], **kwargs)
        encoding_config = {'type': 'gaussian' }
        self.model = GenericModel(in_coords=3, out_coords=1, n_layers=4, dim=64,
                                  encoding_config=encoding_config)

    def forward(self, batch, **kwargs):
        coords = batch['coords']

        # flip probability between 0 and 1
        flip = torch.sigmoid(self.model(coords))

        return {'flip': flip}


class HeightTransformModel(BaseTransformModel):

    def __init__(self, **kwargs):
        super().__init__(tensor_ids=['coords', 'height_range', 'b_true'], **kwargs)
        encoding_config = {'type': 'gaussian'}
        self.mapping_module = GenericModel(in_coords=3, out_coords=1, n_layers=4, dim=64,
                                           encoding_config=encoding_config)

    def forward(self, batch):
        coords = batch['coords']
        height_range = batch['height_range']

        z_coords = self.mapping_module(coords)
        z_coords = torch.sigmoid(z_coords) * (height_range[:, 1:2] - height_range[:, 0:1]) + height_range[:, 0:1]
        transformed_coords = torch.cat([coords[:, :2], z_coords], -1)

        return {'coords': transformed_coords, 'original_coords': coords}

class NLTEHeightTransformModel(BaseTransformModel):

    def __init__(self, height_range, Mm_per_ds, **kwargs):
        super().__init__(tensor_ids=['coords'], **kwargs)
        self.mapping_module = SirenModel(in_dim=2, out_dim=1, n_layers=4, dim=64)
        self.height_range = nn.Parameter(torch.tensor(height_range, dtype=torch.float32) / Mm_per_ds, requires_grad=False)

    def forward(self, batch):
        coords = batch['coords']
        z_coords = coords[..., 2:3]  # Extract z-coordinates
        xy_coords = coords[:, :2]  # Extract x and y coordinates
        scaling = self.mapping_module(xy_coords)
        z_coords = z_coords * 10 ** scaling
        # min_height = self.height_range[None, 0:1]
        # max_height = self.height_range[None, 1:2]
        # z_coords = torch.sigmoid(z_coords) * (max_height - min_height) + min_height
        transformed_coords = torch.cat([xy_coords, z_coords], -1)

        return {'coords': transformed_coords, 'original_coords': coords}

class OpticalDepthTransformModel(BaseTransformModel):

    def __init__(self, Mm_per_ds, max_height, max_log_optical_depth=-5,  **kwargs):
        assert max_log_optical_depth <= 0, "max_log_optical_depth must be negative"
        super().__init__(tensor_ids=['coords', 'b_true'], **kwargs)
        encoding_config = {'type': 'gaussian'}
        self.model = GenericModel(in_coords=3, out_coords=1, n_layers=8, dim=128,
                                  encoding_config=encoding_config)
        max_scaling = max_height / Mm_per_ds
        self.max_scaling = nn.Parameter(torch.tensor(max_scaling, dtype=torch.float32), requires_grad=False)

    def forward(self, batch):
        coords = batch['coords']

        z = torch.sigmoid(self.model(coords)) * self.max_scaling

        xy_coords = coords[:, :2]
        transformed_coords = torch.cat([xy_coords, z], dim=-1)

        z_jacobian = jacobian(z, coords)
        dz_dtau = z_jacobian[..., :, 2:3]

        return {'coords': transformed_coords, 'original_coords': coords, 'dz_dtau': dz_dtau}