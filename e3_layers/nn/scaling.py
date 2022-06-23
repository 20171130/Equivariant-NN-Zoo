from typing import List, Optional

import torch
from .sequential import Module

RESCALE_THRESHOLD = 1e-6

import torch


class PerTypeScaleShift(Module):
    def __init__(
        self,
        num_types: int,
        shifts: Optional[List[float]],
        scales: Optional[List[float]],
        scales_trainable: bool = False,
        shifts_trainable: bool = False,
        irreps_in="1x0e",
        irreps_out="1x0e",
        species="1x0e",
    ):
        """shifts are applied after scaling"""
        super().__init__()
        self.num_types = num_types

        self.init_irreps(
            input=irreps_in, output=irreps_out, species=species, output_keys=["output"]
        )

        self.has_shifts = shifts is not None
        if shifts is not None:
            shifts = torch.as_tensor(shifts, dtype=torch.get_default_dtype())
            if len(shifts.reshape([-1])) == 1:
                shifts = torch.ones(num_types) * shifts
            assert shifts.shape == (num_types,), f"Invalid shape of shifts {shifts}"
            self.shifts_trainable = shifts_trainable
            if shifts_trainable:
                self.shifts = torch.nn.Parameter(shifts)
            else:
                self.register_buffer("shifts", shifts)

        self.has_scales = scales is not None
        if scales is not None:
            scales = torch.as_tensor(scales, dtype=torch.get_default_dtype())
            if len(scales.reshape([-1])) == 1:
                scales = torch.ones(num_types) * scales
            assert scales.shape == (num_types,), f"Invalid shape of scales {scales}"
            self.scales_trainable = scales_trainable
            if scales_trainable:
                self.scales = torch.nn.Parameter(scales)
            else:
                self.register_buffer("scales", scales)

    def forward(self, data):
        input = self.inputKeyMap(data)
        species, input = input["species"], input["input"]

        if self.has_scales:
            input = self.scales[species].view(-1, 1) * input
        if self.has_shifts:
            input = self.shifts[species].view(-1, 1) + input

        is_per = self.inputKeyMap(data.attrs)["input"][0]
        data.attrs.update(
            self.outputKeyMap({"output": (is_per, self.irreps_out["output"])})
        )
        data.update(self.outputKeyMap({"output": input}))
        return data
