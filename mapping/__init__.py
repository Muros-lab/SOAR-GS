from .mapper import IncrementalMapper
from .voxel_map import VoxelMap
from .gaussian_map import GaussianMap


def get_mapper(cfg, device):
    # Create mapper with full config access (not just cfg.mapper)
    mapper = IncrementalMapper(cfg.mapper, device)
    # Store full config so detector can be accessed
    mapper.full_cfg = cfg
    return mapper

