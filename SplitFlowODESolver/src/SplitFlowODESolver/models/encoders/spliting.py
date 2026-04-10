from __future__ import annotations
from typing import Callable, List, Optional, Tuple
from dataclasses import dataclass


import torch
import torch.nn as nn




@dataclass(frozen=True)
class FlowSpec:
    name:str
    field:Optional(nn.Module)
    t_0: float
    t_1: float
    steps: int
    enabled: bool = True

class SplitingODEBlock(nn.Module):
    """
    Split-flow ODE block to tune models for transformer-style token dynamics.


    Modes
    -------
    1 ) Lie splitting
        - pre : fric > attn > mlp
        - mid : attn > fric > mlp
        - post : attn > mlp >fric
    2 ) Strang splitting
        - pre : fric > attn(h/2) > mlp(h) > attn(h/2)
        - mid : attn(h/2) > fric > mlp(h) > attn(h/2)
        - post : attn(h/2) > mlp(h) > attn(h/2) > fric
        - symmetric : fric(h/2) > attn(h/2) > mlp(h) > attn(h/2) > fric(h/2)
    Notes
    -------
    - residual connections : x + block(x, grid_shape), let the setting be true
    """

    def __init__(
        self,
        attn_field: nn.Module,
        mlp_field: nn.Module,
        fric_field: nn.Module,
        ode_mode: str = "strang",
        *,
        steps_attn: int = 2,
        steps_mlp: int = 1,
        steps_fric: int = 1,
        t_0: float = 0.0,
        t_1: float = 1.0,
        use_friction: bool = True,
        friction_position: str = "pre",
        residual: bool = False,
        retrun_debug: bool = True,
        integrator: Optional[Callable[..., torch.Tensor]] = None,
    ):
        super().__init__()

        valid_modes = {"lie", "strang"}
        valid_friction_positions = {"pre","mod","post","symmetric"}

        if mode not in valid_modes:
            raise ValueError(f"mode : {valid_modes}, got = {mode}")
        if friction_position not in valide_friction_positions:
            raise ValueError(
                f"friction_position = {valid_friction_positions}, got = {friction_position}"
            )
        if steps_attn < 1:
            raise ValueError(f"steps_attn >= 1, got {steps_attn}")
        if steps_mlp < 1:
            raise ValueError(f"steps_mlp >= 1, got {steps_mlp}")
        if steps_fric < 1:
            raise ValueError(f"steps_fric >= 1, got {steps_fric}")

        self.attn = attn_field
        self.mlp = mlp_field
        self.fric = fric_field
        self.ode_mode = ode_mode
        self.steps_attn = steps_attn
        self.steps_mlp = steps_mlp
        self.steps_fric = steps_fric
        self.t_0 = t_0
        self.t_1 = t_1
        self.use_friction = use_friction
        self.friction_position = friction_position
        self.residual = residual
        self.return_debug = return_debug

        if integrator is None:
            self.integrator = ode_integrator
        else:
            self.integrator = integrator

    def _flow(self, field: nn.Module, x: torch.Tensor, grid_shape: Tuple[int, int, int], t_0: float, t_1: float, steps: int) -> torch.Tensor:
        return self.integrator(
            field, x, grid_shape, t_0, t_1, steps,
        )

    def extra_debug(self) -> str:
        return(
            f"mode = {self.mode},"
            f"steps_attn = {self.steps_attn},"
            f"steps_mlp = {self.steps_mlp},"
            f"steps_fric = {self.steps_fric},"
            f"residual = {self.residual}"
        )

    def _split_steps(self, total_steps: int) -> Tuple[int, int]:
        left = total_steps // 2
        right = total_steps - left
        left = max(1, left)
        right = max(1, right)

        return left, right

    def _run_spec(self, spec: FlowSpec, x: torch.Tensor, grid_shape: Tuple[int, int, int], debug_log: List[str]) -> torch.Tensor:
        if not spec.enabled or spec.field is None:
            return x
        
        debug_log.append(
            f"{spec.name}: t=({spec.t_0:.3f}, {spec.t_1:.3f}, steps={spec.steps})"
        )

        return self._flow(spec.field, x, grid_shape, spec.t_0, spec.t_1, spec.steps)
    
    def _build_lie_schdule(self) -> List[FlowSpec]:
        t_0, t_1 = self.t_0, self.t_1
        schedule: List[FlowSpec] = []

        fric_spec = FlowSpec(
            name="fric",
            field=self.fric,
            t_0 = t_0,
            t_1 = t_1,
            steps = self.steps_fric,
            enabled = self.use_friction,
        )

        attn_spec = FlowSpec(
            name="attn",
            field=self.attnm
            t_0 = t_0,
            t_1 = t_1,
            steps = self.steps_attn,
            enabled=True,
        )

        mlp_spec = FlowSpec(
            name="mlp",
            field=self.mlp,
            t_0=t_0,
            t_1=t_1,
            steps=self.steps_mlp,
            enabled=True,
        )

        if self.friction_position == "pre":
            schedule = [fric_spec, attn_spec, mlp_spec]
        elif self.friction_position == "mid":
            schedule = [attn_spec, fric_spec, mlp_spec]
        elif self.friction_position == "post":
            schedule = [attn_spec, mlp_spec, fric_spec]
        elif self.friction_position == "symmetric":
            schedule = [fric_spec, attn_spec, mlp_spec, fric_spec]
        else:
            raise RuntimeError("Unexpected friction mode")

        return schedule
    
    def _build_strang_schdule(self) -> List[FlowSpec]:
        t_0, t_1 = self.t_0, self.t_1
        schedule: List[FlowSpec] = []

        mid = 0.5 * (t_0 + t_1)

        left_attn_steps, right_attn_steps = self._split_steps

        fric_full_spec = FlowSpec(
            name="fric",
            field=self.fric,
            t_0 = t_0,
            t_1 = t_1,
            steps = self.steps_fric,
            enabled = self.use_friction,
        )

        fric_right = FlowSpec(
            name="fric_right",
            field=self.fric,
            t_0 = t_0,
            t_1 = t_1,
            steps = max(1, self.steps_fric - (self.steps_fric // 2)),
            enabled = self.use_friction,
        )

        fric_left = FlowSpec(
            name="fric_left",
            field=self.fric,
            t_0 = t_0,
            t_1 = t_1,
            steps = max(1, self.steps_fric // 2),
            enabled = self.use_friction,
        )

        attn_right = FlowSpec(
            name="attn_right",
            field=self.attn,
            t_0 = t_0,
            t_1 = t_1,
            steps = right_attn_steps,
            enabled=True,
        )

        attn_left = FlowSpec(
            name="attn_left",
            field=self.attn,
            t_0 = t_0,
            t_1 = t_1,
            steps = left_attn_steps,
            enabled=True,
        )

        mlp_spec = FlowSpec(
            name="mlp",
            field=self.mlp,
            t_0=t_0,
            t_1=t_1,
            steps=self.steps_mlp,
            enabled=True,
        )

        if self.friction_position == "pre":
            schedule = [fric_full_spec, attn_left, mlp_spec, attn_right]
        elif self.friction_position == "mid":
            schedule = [attn_left, fric_full_spec, mlp_spec, attn_right]
        elif self.friction_position == "post":
            schedule = [attn_left, mlp_spec, attn_right, fric_full_spec]
        elif self.friction_position == "symmetric":
            schedule = [fric_left, attn_left, mlp_spec, attn_right, fric_right]
        else:
            raise RuntimeError("Unexpected friction mode")

        return schedule

    def _build_schedule(self) -> List[FlowSpec]:
        if self.ode_mode == "lie":
            return self._build_lie_schedule()
        if self.ode_mode == "strang":
            return self._build_strang_schedule()
        raise RuntimeError(f"Unexpected ode mode: {self.ode_mode}")

    def forward(
        self,
        t: Optional[torch.Tensor],
        x: torch.Tensor,
        grid_shape: Tuple[int, int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
       
       x_input = x

       debug_log: List[str] = []

       schedule = self._build_schedule()

       zx = x
       for spec in schedule:
        zx = self._run_spec(spec, zx, grid_shape, debug_log)

       if self.residual:
        x_input + zx

       if self.return_debug:
        return zx, {
            "ode_mode": self.ode_mode,
            "friction_position":self.friction_position,
            "schedule": debug_log,
        } 

        return zx, t

