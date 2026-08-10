import torch
import torch.nn as nn


class ThermalInformedLoss(nn.Module):
    """
    Physics-Informed Loss Function (PILF) for lithium-ion cell surface temperature prediction.
    
    This loss function combines a standard empirical loss (Mean Squared Error) with two 
    thermodynamic penalty terms:
    
    1. Thermal Energy Balance Penalty (L_ODE): Enforces a lumped-capacitance energy balance 
       by penalizing deviations in the temporal derivative of the predicted temperature 
       against the theoretical heat generation (Joule and entropic heating) and dissipation.
    2. Rest Phase Constraint (L_rest): Ensures that when the cell is at rest (Current ~ 0), 
       the temperature strictly decays toward the environmental temperature, penalizing any 
       spontaneous and non-physical temperature increases or divergences.
       
    Note: 
        This loss is strictly designed for sequence-to-sequence tasks. The input tensors 
        must contain a time sequence dimension (e.g., `(batch, sequence_length)` or 
        `(batch, sequence_length, features)`) to compute the finite-difference temporal 
        gradients correctly.
    """
    def __init__(self, lambda_1=1.0, lambda_2=1.0, delta_t=600, epsilon=1e-3):
        """
        Args:
            lambda_1 (float): Weighting hyperparameter for the ODE residual penalty.
            lambda_2 (float): Weighting hyperparameter for the rest phase constraint. 
            delta_t (float): The time step duration between consecutive sequence points. 600 seconds according to the raw dataset.
            epsilon (float): Threshold to determine if the current is effectively zero to account 
                            for sensor noise during resting periods. 
        """
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.delta_t = delta_t
        self.epsilon = epsilon
        
        # Learnable physical coefficients
        self.alpha = nn.Parameter(torch.tensor(1e-4))
        self.beta = nn.Parameter(torch.tensor(1e-4))
        self.gamma = nn.Parameter(torch.tensor(1e-4))
        
        self.mse_criterion = nn.MSELoss()

    def forward(self, T_pred, T_true, current, T_env):
        # 1. Empirical Loss (MSE)
        loss_mse = self.mse_criterion(T_pred, T_true)
        
        # Slicing to align t and t-1
        # Assuming tensors are of shape (batch_size, sequence_length) or (sequence_length,)
        T_pred_t = T_pred[..., 1:]
        T_pred_t_minus_1 = T_pred[..., :-1]
        I_t = current[..., 1:]
        T_env_t = T_env[..., 1:]
        
        # Temporal derivative
        dT = T_pred_t - T_pred_t_minus_1
        dT_dt = dT / self.delta_t
        
        # 2. ODE Residual Penalty (L_ODE)
        ode_rhs = self.alpha * (I_t ** 2) + self.beta * I_t - self.gamma * (T_pred_t - T_env_t)
        loss_ode = torch.mean((dT_dt - ode_rhs) ** 2)
        
        # 3. Rest Phase Constraint (L_rest)
        rest_indicator = (torch.abs(I_t) < self.epsilon).float()
        
        sign_diff = torch.sign(T_pred_t_minus_1 - T_env_t)
        cooling_violation = torch.relu(dT * sign_diff)
        
        loss_rest = torch.mean(rest_indicator * (cooling_violation ** 2))
        
        # Total Loss formulation
        loss_total = loss_mse + self.lambda_1 * loss_ode + self.lambda_2 * loss_rest
        
        return loss_total
