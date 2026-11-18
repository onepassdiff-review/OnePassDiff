"""Terrain kind of each env under the curriculum layout of envs/common/terrain.py (numpy + torch only).

With curriculum=True, terrain.py assigns terrain kinds to columns in proportion to
terrainProportions (cumulative, rounded to whole columns); every env keeps its column
(task.terrain_types). Used by the OnePass agent (per-kind eta) and by legged_terrain.py
(per-kind curriculum level, flag terrain.logLevelPerKind).
"""
import numpy as np
import torch

KIND_NAMES = ('rough_up', 'rough_down', 'rough_flat', 'stair_up', 'stair_down',
              'smooth_up', 'smooth_down', 'discrete', 'stepping_stone')


def kind_of_envs(cum_proportions, env_cols, terrain_types):
    """Kind index (0..8, terrain.py order) of every env, as a numpy int array."""
    rounded = np.round(np.asarray(cum_proportions) * env_cols).astype(np.int64)
    cols = torch.as_tensor(terrain_types).detach().cpu().numpy()
    return np.clip(np.searchsorted(rounded, cols, side='right'), 0, len(KIND_NAMES) - 1)
