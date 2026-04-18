import torch
import torch.nn as nn


from AnomalyDetection.src.solvers.rk import ode_integrate_rk



@dataclass
class FlowSpec:
    name:str
    field: Optional[nn.Module]
    t_0: float
    t_1: float
    steps: int
    enabled: bool = True

class SplitODEBlock(nn.Module):
    """
    residual block
    Compose sub-flows:
     - Lie-Trotter: f_fric^h > f_attn^h > f_mlp^(h)
     - Strang : f_attn^(h/2) > [fric] > mlp > [fric] > f_attn^(h/2)
    """
    def __init__(
        self,
        attn_field: nn.Module,
        mlp_field: nn.Module,
        fric_field: nn.Module,
        mode: str = 'strang',
        steps_attn: int = 2,
        steps_mlp: int = 1,
        steps_fric: int = 1,
        t_0: float = 0.0,
        t_1: float = 1.0,
        use_friction: bool = True,
        friction_position: str = 'mid',
        residual: bool = False,
        return_debug: bool = False,
        integrator: Optional[Callable[..., torch.Tensor]] = None,
    ):
        super().__init__()
        assert mode in ['lie', 'strang']
        assert friction_position in ['pre', 'mid', 'post', 'symmetric']

        self.attn = attn_field
        self.mlp = mlp_field
        self.fric = fric_field
        self.mode = mode

        self.steps_attn = steps_attn
        self.steps_mlp = steps_mlp
        self.steps_fric = steps_fric

        self.t_0, self.t_1 = t_0, t_1
        self.use_friction = use_friction and (fric_field is not None)
        self.friction_position = friction_position
        self.residual = residual
        self.return_debug = return_debug
        self.integrator = integrator

        if step_attn < 1:
            raise ValueError(f"steps attention must be >= 1, got {steps_attn}")
        
        if steps_min < 1:
            raise ValueERror(f"steps mlp must be >= 1, got {steps_mlp}")
        
        if steps_fric < 1:
            raise ValueError(f"steps friction must be >= 1, got {steps_fric}")

        if t_1 <= t_0:
            raise ValueError(f"[value inspection] t_0 < t_1, got {t_0}, {t_1}")
    

    def extra_rep(self) -> str:
        return (
            f"mode = {self.mode}, "
            f"steps_attn = {self.steps_attn}, "
            f"steps_mlp = {self.steps_mlp}, "
            f"steps_fric = {self.steps_fric}, "
            f"use_friction = {self.use_friction}, "
            f"friction_position = {self.friction_position}, "
            f"residual = {self.residual}"
        )
    
    def _flow(self, field: nn.Module, x: torch.Tensor, grid_shape: Tuple[int, int, int], t_0: float, t_1: float, steps: int) -> torch.Tensor:
        
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        
        return self.integrator(field, x, grid_shape ,t_0 = t_0, t_1 = t_1, steps = steps)

    def _run_spec(self, spec: FlowSpec, x: torch.Tensor, grid_shape: Tuple[int, int, int], debug_log: List[str]) -> torch.Tensor:

        if not spec.enabled or spec.field is None:
            return x

        debug_log.append(f"{spec.name}: t= ({spec.t_0:..3f}), ({spec.t_1:.3f}), ({spec.steps})")

        return self._flow(spec.field, x, grid_shape, spec.t_0, spec.t_1, spec.steps)

    def _split_attn_steps(self) -> Tuple[int, int]:

        left = self.steps_attn // 2
        right = self.steps_attn - left

        left = max(left, 1)
        right = max(right, 1)

        return left, right
    
    def _build_lie(self) -> List[FlowSpec]:
        t_0, t_1 = self.t_0, self.t_1
        schedule: List[FlowSpec] = []

        fric_spec = FlowSpec(
            name = "fric",
            field = self.fric,
            t_0 = t_0,
            t_1 = t_1,
            steps = self.steps_fric,
            enabled = self.use_friction,
        )

        attn_spec = FlowSpec(
            name = "attn",
            field = self.attn,
            t_0 = t_0,
            t_1 = t_1,
            steps = self.steps_attn,
            enabled = True,
        )

        mlp_spec = FlowSpec(
            name = "mlp",
            field = self.mlp,
            t_0 = t_0,
            t_1 = t_1,
            steps = self.steps_mlp,
            enabled = True,
        )

        if self.frition_position == "pre":
            schedule = [fric_spec, attn_spec, mlp_spec]
        elif self.friction_position == "mid":
            schedule = [attn_spec, fric_spec, mlp_spec]
        elif self.friction_position == "post":
            schedule = [attn_spec, mlp_spec, fric_spec]
        elif self.friction_position == "symmetric":
            schedule = [fric_spec, attn_spec, mlp_spec, fric_spec]
        else:
            raise RuntimeError(f"unexpected friction position >> {self.friction_position}")

        
        return schedule

    
    def _build_strang(self) -> List[FlowSpec]:

        t_0, t_1 = self.t_0, self.t_1
        mid = 0.5 * (t_0 + t_1)

        left_attn_steps, right_attn_steps = self._split_attn_steps()

        attn_left = FlowSpec(
            name = "attn_left",
            field = self.attn,
            t_0 = t_0,
            t_1 = t_1,
            steps = left_attn_steps,
            enabled = True,
        )

        attn_right = FlowSpec(
            name = "attn_right",
            field = self.attn,
            t_0 = t_0,
            t_1 = t_1,
            steps = right_attn_steps,
            enabled = True,
        )

        fric_full = FlowSpec(
            name = "fric_full",
            field = self.fric,
            t_0 = t_0,
            t_1 = t_1,
            steps = self.steps_fric,
            enabled = self.use_friction,
        )

        # fric(l / 2)
        fric_left = FlowSpec(
            name = "fric_left",
            field = self.fric,
            t_0 = t_0,
            t_1 = t_1,
            steps = max(1, self.steps_fric // 2),
            enabled = self.use_friction,
        )

        # fric(l / 2)
        fric_right = FlowSpec(
            name = "fric_right",
            field = self.fric,
            t_0 = t_0,
            t_1 = t_1,
            steps = max(1, self.steps_fric - (self.steps_fric // 2)),
            enabled = self.use_friction,
        )

        mlp_full = FlowSpec(
            name = "mlp",
            field = self.mlp,
            t_0 = t_0,
            t_1 = t_1,
            steps = self.steps_mlp,
            enabled = True,
        )

        if self.frition_position == "pre":
            schedule = [fric_full, attn_left, mlp_full, attn_right]
        elif self.friction_position == "mid":
            schedule = [attn_left, fric_full, mlp_full, attn_right]
        elif self.friction_position == "post":
            schedule = [attn_left, mlp_full, attn_right, fric_full]
        elif self.friction_position == "symmetric":
            schedule = [fric_left, attn_left, mlp_full, attn_right, fric_right]
        else:
            raise RuntimeError(f"unexpected friction position >> {self.friction_position}")

        return schedule

    def forward(self, x: torch.Tensor, grid_shape: Tuple[int, int, int]):
        input_x = x
        debug_log: List[str] =[]

        if mode == "lie":
            schedule = self._build_lie()
        elif mode == "strang":
            schedule = self._build_strang()
        else:
            raise ValueError(f"unsupported mode : {self.mode}")

        def apply_fric(z, grid_shape):
            if self.use_friction:
                return self._flow(self.fric, z, grid_shape, t_0, t_1, self.steps_fric)
            return z

        z = x
        for spec in schedule:
            z = self._run_spec(spec, z, grid_shape, debug_log)
        
        
        if self.residual:
            z = input_x + z

        if self.return_debug:
            return z, {
                "mode": self.mode,
                "friction_position": self.friction_position,
                "schedule": debug_log,
            }

        return z