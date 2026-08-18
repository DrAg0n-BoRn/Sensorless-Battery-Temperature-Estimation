import torch
from dragon.ML_trainer import DragonSequenceTrainer
from dragon.keys._keys import PyTorchLogKeys
from dragon.ML_scaler import DragonScaler
from dragon.schema import FeatureSchema
from constants import EngineeredColumns


class SequenceThermalInformedTrainer(DragonSequenceTrainer):
    """
    Custom trainer wrapper that overrides the loss computation to pass 
    physical feature sequences (Current and Environment Temperature) 
    to the ThermalInformedLoss.
    
    Requires sequence-to-sequence datasets.
    """
    def __init__(self, schema: FeatureSchema, *args, **kwargs):
        # Initialize the base DragonSequenceTrainer
        # `criterion` passed in kwargs must be an instance of ThermalInformedLoss
        
        # pop schema from kwargs to avoid passing it to the base class
        # self.schema = schema
        kwargs.pop('schema', None)
        super().__init__(*args, **kwargs)
        
        # Dynamically find the indices of the required physical features
        try:
            # --- 1. Find relative indices for tensor slicing during training ---
            # self.train_dataset.feature_names only contains exogenous features
            exogenous_list = self.train_dataset.feature_names
            self.current_idx_in_tensor = exogenous_list.index(EngineeredColumns.CURRENT)
            self.t_env_idx_in_tensor = exogenous_list.index(EngineeredColumns.ENVIRONMENT_TEMPERATURE)
            
            # --- 2. Find absolute indices to extract scaling parameters safely ---
            # The scaler was fitted on the original schema's full feature list
            abs_current_idx = schema.feature_names.index(EngineeredColumns.CURRENT)
            abs_env_idx = schema.feature_names.index(EngineeredColumns.ENVIRONMENT_TEMPERATURE)
            
            f_scaler: DragonScaler = self.train_dataset.feature_scaler
            t_scaler: DragonScaler = self.train_dataset.target_scaler
            
            # Extract Feature Means and Stds
            if f_scaler and f_scaler.mean_ is not None:
                # Look up the scaler's internal array index using the absolute original index
                curr_scaler_idx = f_scaler.continuous_feature_indices.index(abs_current_idx)
                env_scaler_idx = f_scaler.continuous_feature_indices.index(abs_env_idx)
                
                self.curr_mean = torch.as_tensor(f_scaler.mean_[curr_scaler_idx], dtype=torch.float32, device=self.device)
                self.curr_std = torch.as_tensor(f_scaler.std_[curr_scaler_idx], dtype=torch.float32, device=self.device)
                self.env_mean = torch.as_tensor(f_scaler.mean_[env_scaler_idx], dtype=torch.float32, device=self.device)
                self.env_std = torch.as_tensor(f_scaler.std_[env_scaler_idx], dtype=torch.float32, device=self.device)
            else:
                raise ValueError("Feature scaler is not fitted or missing. Ensure that the training dataset has a valid feature scaler.")
                
            # Extract Target Means and Stds
            self.t_cell_target_idx = self.target_names.index(EngineeredColumns.CELL_TEMPERATURE)
            if t_scaler and t_scaler.mean_ is not None:
                # Use the relative target index to find the scaling parameters
                t_scaler_idx = t_scaler.continuous_feature_indices.index(self.t_cell_target_idx)
                self.target_mean = torch.as_tensor(t_scaler.mean_[t_scaler_idx], dtype=torch.float32, device=self.device)
                self.target_std = torch.as_tensor(t_scaler.std_[t_scaler_idx], dtype=torch.float32, device=self.device)
            else:
                raise ValueError("Target scaler is not fitted or missing. Ensure that the training dataset has a valid target scaler.")

        except ValueError as e:
            raise ValueError(f"Physical features missing from dataset features.") from e

    def _compute_loss(self, outputs, targets, features: torch.Tensor) -> torch.Tensor:
        """
        Overridden to unpack physical features and pass them to the PILF.
        Expects outputs and targets to have a sequence dimension (sequence-to-sequence).
        Handles multi-head dictionary outputs.
        """
        # 1. Extract scaled tensors using relative indices
        current_scaled = features[..., self.current_idx_in_tensor]
        T_env_scaled = features[..., self.t_env_idx_in_tensor]
        
        # --- Handle Dictionary Outputs ---
        if isinstance(outputs, dict):
            T_pred_scaled = outputs[EngineeredColumns.CELL_TEMPERATURE]
        else:
            T_pred_scaled = outputs
            
        # --- Handle Dictionary Targets ---
        if isinstance(targets, dict):
            T_true_scaled = targets[EngineeredColumns.CELL_TEMPERATURE]
        else:
            # If target is a tensor, extract the specific column using the index
            if targets.ndim > 1 and targets.shape[-1] > 1:
                T_true_scaled = targets[..., self.t_cell_target_idx]
            else:
                T_true_scaled = targets
        
        T_pred_scaled = T_pred_scaled.squeeze(-1) if T_pred_scaled.shape[-1] == 1 else T_pred_scaled
        T_true_scaled = T_true_scaled.float().squeeze(-1) if T_true_scaled.shape[-1] == 1 else T_true_scaled.float()
        
        # 2. Mathematically unscale back to physical units (Amperes, Kelvins)
        # Adding 1e-8 to std to perfectly mirror DragonScaler's inverse_transform behavior
        # Ensure scaling parameters match the active device of the features tensor
        active_device = features.device
        
        current_physical = (current_scaled * (self.curr_std.to(active_device) + 1e-8)) + self.curr_mean.to(active_device)
        T_env_physical = (T_env_scaled * (self.env_std.to(active_device) + 1e-8)) + self.env_mean.to(active_device)
        T_pred_physical = (T_pred_scaled * (self.target_std.to(active_device) + 1e-8)) + self.target_mean.to(active_device)
        T_true_physical = (T_true_scaled * (self.target_std.to(active_device) + 1e-8)) + self.target_mean.to(active_device)
        
        # 3. Compute loss using the custom criterion
        assert isinstance(self.criterion, torch.nn.Module), "Criterion must be an instance of nn.Module (ThermalInformedLoss)."
        loss = self.criterion(
            T_pred=T_pred_physical,
            T_true=T_true_physical,
            current=current_physical,
            T_env=T_env_physical
        )
        return loss

    def _train_step(self) -> dict[str, float]:
        """Overridden to pass `features` into `_compute_loss`."""
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        
        for batch_idx, (features, target) in enumerate(self.train_loader): # type: ignore
            batch_logs = {
                PyTorchLogKeys.BATCH_INDEX: batch_idx, 
                PyTorchLogKeys.BATCH_SIZE: features.size(0)
            }
            self._callbacks_hook('on_batch_begin', batch_idx, logs=batch_logs)

            features = features.to(self.device)
            if isinstance(target, torch.Tensor):
                target = target.to(self.device)
            elif isinstance(target, dict):
                target = {k: v.to(self.device) for k, v in target.items()}

            self.optimizer.zero_grad()
            outputs = self.model(features)
            
            # Pass features to the overridden compute_loss
            loss = self._compute_loss(outputs, target, features) # type: ignore
            
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item()
            batch_size = features.size(0)
            running_loss += batch_loss * batch_size
            total_samples += batch_size
            
            batch_logs[PyTorchLogKeys.BATCH_LOSS] = batch_loss
            self._callbacks_hook('on_batch_end', batch_idx, logs=batch_logs)
        
        if total_samples == 0:
            return {PyTorchLogKeys.TRAIN_LOSS: 0.0}

        return {PyTorchLogKeys.TRAIN_LOSS: running_loss / total_samples}

    def _validation_step(self) -> dict[str, float]:
        """Overridden to pass `features` into `_compute_loss`."""
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for features, target in self.validation_loader: # type: ignore
                features = features.to(self.device)
                if isinstance(target, torch.Tensor):
                    target = target.to(self.device)
                elif isinstance(target, dict):
                    target = {k: v.to(self.device) for k, v in target.items()}

                outputs = self.model(features)
                
                # Pass features to the overridden compute_loss
                loss = self._compute_loss(outputs, target, features) # type: ignore
                
                batch_size = features.size(0)
                running_loss += loss.item() * batch_size
                total_samples += batch_size
                
        if total_samples == 0:
            return {PyTorchLogKeys.VAL_LOSS: 0.0}
        
        return {PyTorchLogKeys.VAL_LOSS: running_loss / total_samples}
