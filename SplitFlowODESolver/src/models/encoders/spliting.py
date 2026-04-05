from typing import Optional, Tuple



import torch
import torch.nn as nn



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
    - residual connections : x + block(x, grid_shape)
    """

    def __init__(
        self,
        attn_field: nn.Module,
        mlp_field: nn.Module,
        fric_field: nn.Module,
        ode_mode: str = "strang",
        steps_attn: int = 2,
        steps_mlp: int = 1,
        steps_fric: int = 1,
        t_0: float = 0.0,
        t_1: float = 1.0,
        use_friction: bool = True,
        friction_position: str = "pre",
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

    def _flow(self, field: nn.Module, x: torch.Tensor, grid_shape: Tuple[int, int, int], t_0: float, t_1: float, steps: int) -> torch.Tensor:
        return ode_rk(
            field, x, grid_shape, t_0, t_1, steps,
        )

    def _split_steps(self, total_steps: int) -> Tuple[int, int]:
        left = total_steps // 2
        right = total_steps - left
        left = max(1, left)
        right = max(1, right)

        return left, right

    def forward(
        self,
        t: Optional[torch.Tensor],
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.time_invariant:
            t = None
        dx = self.net(x)
        return dx, t

