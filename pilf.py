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


class ThermalInformedLossV2(nn.Module):
    """
    Physics-Informed Loss Function (PILF) utilizing Global Thermodynamic Boundaries 
    and a Thermal Inertia Penalty. 
    
    Note: This loss function is strictly designed for sequence-to-sequence models, 
    as it requires a contiguous temporal dimension to compute the discrete second 
    derivative (curvature) of the temperature trajectory.
    
    The total loss combines the empirical error with three physical constraints:
    
    1. Empirical Loss (MSE): Standard Mean Squared Error.
    2. Environmental Floor Penalty (L_floor): Penalizes predictions falling below 
       the minimum environmental baseline.
       Formulation: L_floor = mean(max(0, T_env_min - T_pred)^2)
    3. Thermodynamic Ceiling Penalty (L_ceil): Penalizes non-physical temperature spikes.
       Formulation: L_ceil = mean(max(0, T_pred - T_max_limit)^2)
    4. Thermal Inertia Penalty (L_smooth): Treats the cell as a thermal mass (low-pass filter).
       It penalizes high-frequency oscillations in the predicted temperature sequence by 
       minimizing the discrete second derivative, ensuring physically smooth transitions.
       Formulation: L_smooth = mean((T_{i+2} - 2T_{i+1} + T_{i})^2)
    """
    def __init__(self, T_env_min, T_max_limit, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0):
        """
        Args:
            T_env_min (float): Minimum absolute environmental temperature (Kelvin) (must be scaled if data is scaled).
            T_max_limit (float): Maximum physical or experimental cell temperature (Kelvin) (must be scaled if data is scaled).
            lambda_1 (float): Weight for the environmental floor penalty.
            lambda_2 (float): Weight for the thermodynamic ceiling penalty.
            lambda_3 (float): Weight for the thermal inertia (smoothness) penalty.
        """
        super().__init__()
        self.T_env_min = T_env_min
        self.T_max_limit = T_max_limit
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        
        self.mse_criterion = nn.MSELoss()

    def forward(self, T_pred, T_true):
        # 1. Empirical Loss (MSE)
        loss_mse = self.mse_criterion(T_pred, T_true)
        
        # 2. Environmental Floor Penalty (L_floor)
        floor_violation = torch.relu(self.T_env_min - T_pred)
        loss_floor = torch.mean(floor_violation ** 2)
        
        # 3. Thermodynamic Ceiling Penalty (L_ceil)
        ceil_violation = torch.relu(T_pred - self.T_max_limit)
        loss_ceil = torch.mean(ceil_violation ** 2)
        
        # 4. Thermal Inertia Penalty (L_smooth)
        # Calculates the discrete second derivative across the sequence
        second_derivative = T_pred[..., 2:] - 2 * T_pred[..., 1:-1] + T_pred[..., :-2]
        loss_smooth = torch.mean(second_derivative ** 2)
        
        # Total Loss formulation
        loss_total = loss_mse + self.lambda_1 * loss_floor + self.lambda_2 * loss_ceil + self.lambda_3 * loss_smooth
        
        return loss_total


class ThermalInformedLossV3(nn.Module):
    """
    Physics-Informed Loss Function (PILF) utilizing Global Thermodynamic Boundaries.
    
    This loss function is designed to be agnostic to the model architecture, fully 
    supporting both sequence-to-value and sequence-to-sequence predictions. It enforces 
    macroscopic thermodynamic constraints without requiring synchronous exogenous variables 
    (e.g., current, ambient temperature) during the forward pass.
    
    The total loss is a weighted sum of the empirical error and two boundary penalties:
    
    1. Empirical Loss (MSE): Standard Mean Squared Error between predictions and ground truth.
    2. Environmental Floor Penalty (L_floor): Enforces the physical constraint that a cell 
       under normal operation cannot cool below the minimum ambient environmental temperature.
       Formulation: L_floor = mean(max(0, T_env_min - T_pred)^2)
    3. Thermodynamic Ceiling Penalty (L_ceil): Regularizes the model against predicting 
       extreme, non-physical temperature spikes exceeding the known safety or operational 
       limits of the specific cell/experimental setup.
       Formulation: L_ceil = mean(max(0, T_pred - T_max_limit)^2)
    """
    def __init__(self, T_env_min, T_max_limit, lambda_1=1.0, lambda_2=1.0):
        """
        Args:
            T_env_min (float): Minimum absolute environmental temperature (Kelvin) (must be scaled if data is scaled).
            T_max_limit (float): Maximum physical or experimental cell temperature (Kelvin) (must be scaled if data is scaled).
            lambda_1 (float): Weighting hyperparameter for the environmental floor penalty.
            lambda_2 (float): Weighting hyperparameter for the thermodynamic ceiling penalty.
        """
        super().__init__()
        self.T_env_min = T_env_min
        self.T_max_limit = T_max_limit
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        
        self.mse_criterion = nn.MSELoss()

    def forward(self, T_pred, T_true):
        # 1. Empirical Loss (MSE)
        loss_mse = self.mse_criterion(T_pred, T_true)
        
        # 2. Environmental Floor Penalty (L_floor)
        floor_violation = torch.relu(self.T_env_min - T_pred)
        loss_floor = torch.mean(floor_violation ** 2)
        
        # 3. Thermodynamic Ceiling Penalty (L_ceil)
        ceil_violation = torch.relu(T_pred - self.T_max_limit)
        loss_ceil = torch.mean(ceil_violation ** 2)
        
        # Total Loss formulation
        loss_total = loss_mse + self.lambda_1 * loss_floor + self.lambda_2 * loss_ceil
        
        return loss_total


def get_scaled_thermal_boundaries(dataset, t_env_min=296.0, t_max_limit=310.0, target_name="Cell Temperature (K)"):
    """
    Scales physical temperature boundaries using the dataset's fitted target scaler.
    
    Used for ThermalInformedLossV2 and ThermalInformedLossV3 to ensure that the loss function operates in the same scaled space as the model's predictions.
    
    Returns:
        tuple: A tuple containing the scaled environmental minimum temperature and the scaled maximum cell temperature.
            (scaled_t_env_min, scaled_t_max_limit)
    """
    target_scaler = dataset.target_scaler
    
    # Bypass if no scaling was applied to the targets
    if target_scaler is None or target_scaler.mean_ is None:
        return t_env_min, t_max_limit
        
    target_idx = dataset.target_names.index(target_name)
    
    # Create a dummy tensor matching the target feature space
    dummy_tensor = torch.zeros((2, dataset.number_of_targets), dtype=torch.float32)
    
    # Inject the physical values into the designated target column
    dummy_tensor[0, target_idx] = t_env_min
    dummy_tensor[1, target_idx] = t_max_limit
    
    # Apply the standardization transformation
    scaled_tensor = target_scaler.transform(dummy_tensor)
    
    # Extract the scaled boundary values
    scaled_t_env_min = scaled_tensor[0, target_idx].item()
    scaled_t_max_limit = scaled_tensor[1, target_idx].item()
    
    return scaled_t_env_min, scaled_t_max_limit
