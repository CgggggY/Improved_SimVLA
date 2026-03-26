from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import torch


def sample_wrong_instruction(
    instruction: str,
    wrong_map: Dict[str, List[str]],
    all_wrong_pool: Sequence[str],
) -> str:
    """
    Sample one wrong-target instruction for contrastive training.

    Priority:
    1) use wrong_map[instruction] if available
    2) fallback to any non-identical instruction in global pool
    3) fallback to original instruction (no-op)
    """
    candidates = wrong_map.get(instruction, [])
    if candidates:
        return random.choice(candidates)

    pool = [x for x in all_wrong_pool if x != instruction]
    if pool:
        return random.choice(pool)
    return instruction


def sample_wrong_batch(
    instructions: Sequence[str],
    wrong_map: Dict[str, List[str]],
) -> List[str]:
    all_wrong_pool: List[str] = []
    for vals in wrong_map.values():
        all_wrong_pool.extend(vals)
    return [sample_wrong_instruction(ins, wrong_map, all_wrong_pool) for ins in instructions]


def instruction_contrastive_hinge(
    l_pos: torch.Tensor,
    l_neg: torch.Tensor,
    margin: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    L_total = L_pos + lam * max(0, margin - (L_neg - L_pos))
    """
    contrast = torch.relu(torch.as_tensor(margin, device=l_pos.device, dtype=l_pos.dtype) - (l_neg - l_pos))
    total = l_pos + torch.as_tensor(lam, device=l_pos.device, dtype=l_pos.dtype) * contrast
    return total, contrast

