from  __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Union


import torch.nn as nn
import torch



GridShape = Tuple[int, int, int]
StepMode = Literal['lie', 'strang', 'adaptive']
SubSolver = Literal['euler', 'rk4', 'rk45']
TimePolicy = Literal['start', 'midpoint']

def _as_time_tensor(value: Union[float, torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(dtype=ref.dtype, device=ref.device)
    return torch.tensor(value, dtype=ref.dtype, device=ref.device)

def _check_grid_shape(shape: GridShape) -> GridShape:
    if not isinstance(shape, (tuple, list)) or len(shape) != 3:
        raise ValueError(f"Grid shape must be a tuple of integers (nx, ny, nz), got {shape!r}")
    
    if any(int(v) <= 0 for v in shape):
        raise ValueError(f"Grid shape entries must > 0, got {shape!r}")
    return int(shape[0]), int(shape[1]), int(shape[2])

def _check_steps(steps: int) -> int:
    steps = int(steps)
    if steps <= 0:
        raise ValueError(f"Steps must be a positive integer, got {steps}") 
    
    return steps

def _rms_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x.pow(2)))

class RK4Steps(nn.Module):
    """
    dt == float or scalar tensor
    vf returns (vx, vy, vz) of shape (B, 3, nx, ny, nz)
    """

    def __init__(self, vf: nn.Module):
        super().__init__()
        self.vf = vf
    
    def forward(self, t: Union[float, torch.Tensor], x: torch.Tensor, dt: Union[float, torch.Tensor], grid_shape: GridShape) -> torch.Tensor:
        grid_shape = _check_grid_shape(grid_shape)
        t = _as_time_tensor(t, x)
        dt = _as_time_tensor(dt, x)

        half_dt = 0.5 * dt

        k1 = self.vf(t, x, grid_shape)
        k2 = self.vf(t+half_dt, x+half_dt*k1, grid_shape)
        k3 = self.vf(t+half_dt, x+half_dt*k2, grid_shape)
        k4 = self.vf(t+dt, x+dt*k3, grid_shape)

        k = x + (dt/6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return k

@dataclass
class AdaptiveRK45Config:
    rtol: float = 1e-5
    atol: float = 1e-8
    max_steps: int = 300
    min_factor: float = 0.2
    max_factor: float = 2.0

@dataclass
class RK45Steps(nn.Module):

    def __init__(self, vf: nn.Module, config: Optional[AdaptiveRK45Config] = None):
        super().__init__()
        self.vf = vf
        self.config = config or AdaptiveRK45Config()

    def forward(self, t: Union[float, torch.Tensor], x: torch.Tensor, dt: Union[float, torch.Tensor], grid_shape: GridShape, min_dt: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grid_shape = _check_grid_shape(grid_shape)
        t = _as_time_tensor(t, x)
        dt = _as_time_tensor(dt, x)

        if torch.abs(dt).item() == 0.0:
            return x

        direction = torch.sign(dt)
        t_end = t + dt
        h = dt
        atol = self.config.atol
        rtol = self.config.rtol
        max_steps = self.config.max_steps
        
        steps = 0
        rejects = 0

        # Fehlberg coefficients b4 >> 4th order, b5 >> 5th order
        a21 = 1.0 /4.0
        a31, a32 = 3.0 / 32.0, 9.0 / 32.0
        a41, a42, a32 = 1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0
        a51, a52, a53, a54 = 439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0
        a61, a62, a63, a64, a65 = -8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0

        b = (25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, -1.0/5.0, 0.0)
        b_hat = (16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0)
        c2, c3, c4, c5, c6 = 1.0 /4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0

        k1 = self.vf(t, x, grid_shape)
        k2 = self.vf(t + c2 * h, x + h * (a21 * k1), grid_shape)
        k3 = self.vf(t + c3 * h, x + h * (a31 * k1 + a32 * k2), grid_shape)
        k4 = self.vf(t + c4 * h, x + h * (a41 * k1 + a42 * k2 + a43 * k3), grid_shape)
        k5 = self.vf(t + c5 * h, x + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), grid_shape)
        k6 = self.vf(t + c6 * h, x + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5), grid_shape)

        x4 = x + h * (b[0] * k1 + b[2] * k3 + b[3] * k4 + b[4] * k5)
        x5 = x + h * (b_hat[0] * k1 + b_hat[2] * k3 + b_hat[3] * k4 + b_hat[4] * k5 + b_hat[5] * k6)

        err = _rms_norm(x5 -x4) / (atol + rtol * _rms_norm(x5))

        while ((t_end - t) * direction).item() > 0.0:
            if steps >= max_steps:
                raise RuntimeError(f"RK45steps exceeds max steps. {max_steps}")

            remaining = t_end - t

            if (torch.abs(h) > torch.abs(remaining)).item():
                h = remaining
            
            if torch.abs(h).item() < min_dt:
                raise RuntimeError(f"RK45steps below minimum dt, {h} < {min_dt}")
            
            err_value = float(err.detach().cpu().item())
            
            if err_value <= 1.0 or torch.abs(h).item() < min_dt:
                x = x5
                t = t + h
                steps += 1

                if err_value == 0.0:
                    factor = self.config.max_factor
                
                else:
                    factor = 0.9 * (1.0 / err_value) ** 0.2
                    factor = min(self.config.max_factor, max(self.config.min_factor, factor))
                h = h * factor
                print(f"h = {h}")
            
            else:
                rejects += 1
                if rejects > max_steps:
                    raise RuntimeError(f"RK45steps exceeds rejection budget = {max_steps}, last error = {err_value:.3g}")
                factor = 0.9 * (1.0 / err_value) ** 0.2
                factor = min(1.0, max(self.config.min_factor, factor))
                h = h * factor
                print(f"h = {h}")
        
        return x
                
class EulerSteps(nn.Module):
    """
    vf(t, x, grid_shape)
    """
    def __init__(self, vf: nn.Module):
        super().__init__()
        self.vf = vf
    
    def forward(self, t: Union[float, torch.Tensor], x: torch.Tensor, dt: Union[float, torch.Tensor], grid_shape: GridShape) ->  torch.Tensor:
        grid_shape = _check_grid_shape(grid_shape)
        t = _as_time_tensor(t, x)
        dt = _as_time_tensor(dt, x)

    
        euler_velocity = x + dt * self.vf(t, x, grid_shape)

        return euler_velocity
    
class ODEIntegrate(nn.Module):
    """
    monolithic vector field integrator, supports both fixed and adaptive step sizes, as well as Lie-Trotter and Strang splitting.

    corresponding to the Exp-B 'Monolithic ODE Block':
        u' = f_mono(t, u)
    """

    def __init__(self, vf: nn.Module):
        super().__init__()
        self.stepper = RK4Steps(vf)

    def forward(self, x_0: torch.Tensor, t_0: Union[float, torch.Tensor] = 0.0, t_1: Union[float, torch.Tensor] = 1.0, grid_shape: GridShape = None, steps: int = 10) -> torch.Tensor:
        grid_shape = _check_grid_shape(grid_shape)
        steps = _check_steps(steps)

        x = x_0
        t_0 = _as_time_tensor(t_0, x)
        t_1 = _as_time_tensor(t_1, x)

        dt = (t_1 - t_0) / float(steps + 1e-5)

        t = t_0

        for _ in range(steps):
            x = self.stepper(t, x, dt, grid_shape)
            t = t + dt

        return x

def ode_integrate_rk(vf: nn.Module, x_0: torch.Tensor, t_0: Union[float, torch.Tensor], t_1: Union[float, torch.Tensor], grid_shape: GridShape, steps: int = 10) -> torch.Tensor:

    fn = ODEIntegrate(vf)
    t_0 = 0.0
    t_1 = 1.0
    solved = fn(x_0, t_0, t_1, grid_shape, steps=steps)

    return solved

class DiagonalExponentialFrictionFlow(nn.Module):
    """
    Diagonal and channel-wise exponential friction flow

    if coeff_net estimates nonnegative P(t, u):
        u_{n+1} = exp(-h * P(t_n, u_n)) * u_n
    
    coeff_net:
        - input: (t, u, grid_shape)
        - tensor broadcastable to x
    """
    def __init__(self, coeff_net: nn.Module, min_coeff: float = 0.0, max_coeff: float = 10.0):
        super().__init__()
        self.coeff_net = coeff_net
        self.min_coeff = min_coeff
        self.max_coeff = max_coeff
        
    def forward(self, t: Union[float, torch.Tensor], x: torch.Tensor, h: Union[float, torch.Tensor], grid_shape: GridShape) -> torch.Tensor:
        grid_shape = _check_grid_shape(grid_shape)

        t = _as_time_tensor(t, x)
        h = _as_time_tensor(h, x)

        p = self.coeff_net(t, x, grid_shape)
        # P_hat
        p = torch.clamp(p, min = self.min_coeff, max = self.max_coeff)

        return torch.exp(-h * p) * x

class VectorFieldFrictionFlow(nn.Module):
    """
    using when no closed-form exponential flow

    Integrating f_fric(t, x) as an ordinary vector field using Euler or RK4
    """

    def __init__(self, friction_vf: nn.Module, sub_solver: SubSolver = "euler"):
        super().__init__()
        self.sub_solver = sub_solver
        self.euler_stepper = EulerSteps(friction_vf)
        self.rk4_stepper = RK4Steps(friction_vf)
        self.rk45_stepper = RK45Steps(friction_vf)

    def forward(self, t: Union[float, torch.Tensor], x: torch.Tensor, h: Union[float, torch.Tensor], grid_shape: GridShape) -> torch.Tensor:
        if self.sub_solver == "euler":
            return self.euler_stepper(t, x, h, grid_shape)
        elif self.sub_solver == "rk4":
            return self.rk4_stepper(t, x, h, grid_shape)
        elif self.sub_solver == "rk45":
            return self.rk45_stepper(t, x, h, grid_shape)
        else:
            raise ValueError(f"unsupported subsolver {self.sub_solver}, expected 'euler' or 'rk4'")
        

@dataclass
class AdaptiveConfig:
    accepted: int
    rejected: int
    steps: int
    f: float
    mean_error: float
    max_error: float


class SplitFlowODESolver(nn.Module):
    """
    theorem:
        u' = f_attn(t, u) + f_mlp(t, u) + f_fric(t, u)
    
    supports:
        - Lie-Trotter:
            phi_mlp(h) o phi_attn(h) o phi_fric(h)
        - Strang
            phi_mlp(h/2) o phi_attn(h/2) o phi_fric(h) o phi_attn(h/2) o phi_mlp(h/2)
        - Adaptive
            compare Lie and Strang local outputs and accept / reject
        
    parameters:
    - attn_vf: attention-based vector field module
    - mlp_vf: MLP-based token wise reaction vector field module
    - friction_flow : closed-form or numerical friction flow module
        input: (t, x, h, grid_shape)
    - subsolver: for attention and MLP subflows
    - time policy: 
        - start: eval subflows at the left endpoint of t
        - midpoint: eval subflows at t + h/2, time-conditioned nueral blocks
    """
    def __init__(
        self,
        attn_vf: nn.Module,
        mlp_vf: nn.Module,
        friction_flow: nn.Module,
        *,
        sub_solver: SubSolver = "euler",
        mode: StepMode = "lie",
        time_policy: TimePolicy = "start",
        atol: float = 1e-4,
        rtol: float = 1e-3,
        s_eps: float = 0.8,
        min_factor: float = 0.2,
        max_factor: float = 2.0,
        min_dt: float = 1e-5,
        max_rejection: int = 64
    ):
        super().__init__()
        if sub_solver not in ["euler", "rk4"]:
            raise ValueError(f"unsupported sub_solver {sub_solver}, expected 'euler' or 'rk4'")
        if mode not in ["lie", "strang", "adaptive"]:
            raise ValueError(f"unsupported mode {mode}, expected 'lie', 'strang', 'adaptive'")
        
        self.attn_vf = attn_vf
        self.mlp_vf = mlp_vf
        self.friction_flow = friction_flow
        self.attn_euler = EulerSteps(attn_vf)
        self.attn_rk4 = RK4Steps(attn_vf)
        self.attn_rk45 = RK45Steps(attn_vf)
        self.mlp_euler = EulerSteps(mlp_vf)
        self.mlp_rk4 = RK4Steps(mlp_vf)
        self.mlp_rk45 = RK45Steps(mlp_vf)
        self.sub_solver = sub_solver
        self.mode = mode
        self.time_policy = time_policy
        self.atol = atol
        self.rtol = rtol
        self.s_eps = s_eps
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.min_dt = min_dt
        self.max_rejection = max_rejection

    def _eval_time(self, t: torch.Tensor, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t = _as_time_tensor(t, x)
        h = _as_time_tensor(h, x)

        if self.time_policy == "midpoint":
            return t + 0.5 * h
        
        return t

    def _attn_flow(self, t: torch.Tensor, h: torch.Tensor, x: torch.Tensor, grid_shape: GridShape) -> torch.Tensor:
        if self.sub_solver == 'euler':
            return self.attn_euler(t, x, h, grid_shape)
        if self.sub_solver == 'rk45':
            return self.attn_rk45(t, x, h, grid_shape)
        
        return self.attn_rk4(t, x, h, grid_shape)

    def _mlp_flow(self, t: torch.Tensor, h: torch.Tensor, x: torch.Tensor, grid_shape: GridShape) ->  torch.Tensor:

        if self.sub_solver == 'euler':
            return self.mlp_euler(t, x, h, grid_shape)
        if self.sub_solver == 'rk45':
            return self.mlp_rk45(t, x, h, grid_shape)
        
        return self.mlp_rk4(t, x, h, grid_shape)
    
    def lie(self, t: Union[float, torch.Tensor], h: Union[float, torch.Tensor], x: torch.Tensor, grid_shape: GridShape) -> torch.Tensor:
        grid_shape = _check_grid_shape(grid_shape)
        t = _as_time_tensor(t, x)
        h = _as_time_tensor(h, x)

        t_eval = self._eval_time(t, h, x)

        x = self.friction_flow(t_eval, x, h, grid_shape)
        x = self._attn_flow(t_eval, h, x, grid_shape)
        x = self._mlp_flow(t_eval, h, x, grid_shape)

        return x
    
    def strang(self, t: Union[float, torch.Tensor], h: Union[float, torch.Tensor], x: torch.Tensor, grid_shape: GridShape) -> torch.Tensor:
        grid_shape = _check_grid_shape(grid_shape)
        t = _as_time_tensor(t, x)
        h = _as_time_tensor(h, x)

        t_eval = self._eval_time(t, h, x)
        half = 0.5 * h

        x = self.friction_flow(t_eval, x, half, grid_shape)
        x = self._attn_flow(t_eval, half, x, grid_shape)
        x = self._mlp_flow(t_eval, h, x, grid_shape)
        x = self._attn_flow(t_eval, half, x, grid_shape)
        x = self.friction_flow(t_eval, x, half, grid_shape)

        return x
    
    def _normalized_error(self, x_max: torch.Tensor, x_min: torch.Tensor) -> torch.Tensor:

        diff = x_max - x_min

        if diff.ndim >= 2:
            r_dims = tuple(range(1, diff.ndim))
            norm_diff = torch.sqrt(torch.mean(diff.pow(2), dim = r_dims))
            norm_high = torch.sqrt(torch.mean(x_max.pow(2), dim = r_dims))
            err = norm_diff / (self.atol + self.rtol * norm_high)

            return torch.max(err)

        norm_diff = torch.sqrt(torch.mean(diff.pow(2)))
        norm_high = torch.sqrt(torch.mean(x_max.pow(2)))

        return norm_diff / (self.atol + self.rtol * norm_high)

    def fixed_integrate(self, x_0: torch.Tensor, grid_shape: GridShape, t_0 : Union[float, torch.Tensor] = 0.0, t_1: Union[float, torch.Tensor] = 1.0, steps: int = 1, method: Literal['lie', 'strang'] = 'lie') -> torch.Tensor:
        grid_shape = _check_grid_shape(grid_shape)
        steps = _check_steps(steps)

        x = x_0
        t_0 = _as_time_tensor(t_0, x)
        t_1 = _as_time_tensor(t_1, x)
        h = (t_1 - t_0) / float(steps + 1e-5)
        t = t_0

        for _ in range(steps):
            if method == 'lie':
                x = self.lie(t, h, x, grid_shape)
            elif method == 'strang':
                x = self.strang(t, h, x, grid_shape)
            else:
                raise ValueError(f"unsupported method {method}, expected 'lie', 'strang'")        
            t = t + h

        return x
    
    def adaptive_integrate(self, x_0: torch.Tensor, grid_shape: GridShape, t_0 : Union[float, torch.Tensor] = 0.0, t_1: Union[float, torch.Tensor] = 1.0, steps: int = 1, max_steps: int = 128, debug: bool = True):
        """
        ||strang - lie|| / (atol + rtol * ||strang||) < 1 : accept
        else: reject and reduce step size
        """
        grid_shape = _check_grid_shape(grid_shape)
        steps = _check_steps(steps)
        max_steps = _check_steps(max_steps)

        x = x_0
        t_0 = _as_time_tensor(t_0, x)
        t_1 = _as_time_tensor(t_1, x)

        dir = torch.sign(t_1 - t_0)

        if torch.abs(t_1 - t_0).item() == 0.0:
            if debug:
                stats = AdaptiveConfig(accepted=0, rejected=0, steps=0, f=1.0, mean_error=0.0, max_error=0.0)
                return x, stats
            return x
        
        h = (t_1 - t_0) / float(steps + 1e-5)
        t =  t_0
        accepted = 0
        rejected = 0
        errors = []

        while ((t_1 - t) * dir).item() > 0.0 and accepted < max_steps:
            
            s = t_1 - t

            if (torch.abs(h) > torch.abs(s)).item():
                h = s
            
            if torch.abs(h).item() < self.min_dt:
                raise RuntimeError(f"adaptive steps below minimum dt = {self.min_dt}, at t = {t.item()}, t_1 = {t_1.item()}, h = {torch.abs(h).item()}")
            
            x_lie = self.lie(t, h, x, grid_shape)
            x_strang = self.strang(t, h, x, grid_shape)
            err = self._normalized_error(x_strang, x_lie)
            error = float(err.detach().cpu().item())
            errors.append(error)

            if error <= 1.0 or torch.abs(h).item() <= self.min_dt:
                x = x_strang
                t = t + h
                accepted += 1

                if error == 0.0:
                    factor = self.max_factor
                else:
                    factor = self.s_eps * (1.0 / error) ** 0.5
                    factor = min(self.max_factor, max(self.min_factor, factor))
                h = h * factor
            else:
                rejected += 1
                if rejected > self.max_rejection:
                    raise RuntimeError(f"adpative solver exceeded maximum rejections ({self.max_rejection}) at last error = {error:.5g}")
                factor = self.s_eps * (1.0 / error) ** 0.5
                factor = min(1.0, max(self.min_factor, factor))
                h = h * factor
            
        if debug:
            stats = AdaptiveConfig(accepted=accepted, rejected=rejected, steps=accepted+rejected, f=torch.abs(h).detach().cpu().item(), mean_error=float(sum(errors) / max(1, len(errors))), max_error = float(max(errors)  if errors else 0.0))
            return x, stats
            
        return x

    def forward(self, x_0: torch.Tensor, t_0: Union[float, torch.Tensor], t_1: Union[float, torch.Tensor], grid_shape: GridShape, steps: int = 10, debug: bool = True) -> torch.Tensor:
        if self.mode == 'lie':
            return self.fixed_integrate(x_0, grid_shape, t_0, t_1, steps = steps, method='lie')
        elif self.mode == 'strang':
            return self.fixed_integrate(x_0, grid_shape, t_0, t_1, steps = steps, method='strang')
        elif self.mode == 'adaptive':
            return self.adaptive_integrate(x_0, grid_shape, t_0, t_1, steps = steps, debug = debug)
        else:
            raise ValueError(f"unsupported mode {self.mode}, expected 'lie', 'strang', 'adaptive'")
