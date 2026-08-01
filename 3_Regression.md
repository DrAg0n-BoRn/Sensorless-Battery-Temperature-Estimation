---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: sensorless-battery-temperature-estimation (3.12.13)
    language: python
    name: python3
---

```python
from ml_tools.ML_datasetmaster import DragonDataset as ChosenDataset
from ml_tools.ML_trainer import DragonTrainer as ChosenTrainer
from ml_tools.ML_models import DragonNodeModel as ChosenModel
from ml_tools.ML_configuration import (
    FormatRegressionMetrics as ChosenMetricsConfig, 
    FinalizeRegression as ChosenFinalizer, 
    DragonNodeParams as ChosenModelParams,    
)

from ml_tools.ML_configuration import DragonTrainingConfig
from ml_tools.ML_callbacks import DragonModelCheckpoint, DragonPatienceEarlyStopping, DragonPlateauScheduler
from ml_tools.ML_utilities import build_optimizer_params, inspect_model_architecture
from ml_tools.IO_tools import train_logger
from ml_tools.schema import FeatureSchema
from ml_tools.keys import TaskKeys
from torch.optim import AdamW

from paths import PM
from constants import TARGET
```

```python
!yes | plotly_get_chrome
```

```python
SCHEMA_PATH = PM.engineering
TRAIN_DATASET_PATH = PM.train_dataset
TEST_DATASET_PATH = PM.test_dataset
TRAIN_ARTIFACTS_DIR = PM.regression
```

## 1. Load Schema and Dataset

```python
schema = FeatureSchema.from_json(SCHEMA_PATH)
```

```python
dataset_train = ChosenDataset.from_bundle(TRAIN_DATASET_PATH)
```

```python
dataset_test = ChosenDataset.from_bundle(TEST_DATASET_PATH)
```

## 2. Train Config

```python
train_config = DragonTrainingConfig(
    validation_size=dataset_train.validation_split,
    test_size=0, # a complete independent test set
    initial_learning_rate=0.001,
    batch_size=64,
    task = TaskKeys.REGRESSION,
    device = "cuda:0",
    finalized_filename = "node",
    
    targets=TARGET,
    weight_decay=0.005,
    early_stop_patience=25,
    scheduler_patience=6,
    scheduler_lr_factor=0.8,
    monitor_metric="Validation Loss"
)
```

## 3. Model and Trainer

```python
model_params = ChosenModelParams(
    schema=schema,
    out_targets=dataset_train.number_of_targets,
    embedding_dim=24,
    num_trees=512,
    num_layers=2,
    tree_depth=6,
    additional_tree_output_dim=2,
    input_dropout=0.1,
    embedding_dropout=0.1,
    choice_function='entmax',
    bin_function='entmoid',
    batch_norm_continuous=False
)

model = ChosenModel(**model_params)

# Initialize decision thresholds before training.
if hasattr(model, "data_aware_initialization"):
    model.data_aware_initialization(train_dataset=dataset_train.train_dataset, num_samples=2000)

# optimizer
optim_params = build_optimizer_params(model=model, weight_decay=train_config.weight_decay)
optimizer = AdamW(params=optim_params, lr=train_config.initial_learning_rate)

trainer = ChosenTrainer(model=model,
                        train_dataset=dataset_train.train_dataset,
                        validation_dataset=dataset_train.validation_dataset,
                        save_dir=TRAIN_ARTIFACTS_DIR,
                        kind=train_config.task,
                        optimizer=optimizer,
                        device=train_config.device,
                        checkpoint_callback=DragonModelCheckpoint(monitor=train_config.monitor_metric),
                        early_stopping_callback=DragonPatienceEarlyStopping(patience=train_config.early_stop_patience, 
                                                                            monitor=train_config.monitor_metric),
                        lr_scheduler_callback=DragonPlateauScheduler(monitor=train_config.monitor_metric,
                                                                     patience=train_config.scheduler_patience,
                                                                     factor=train_config.scheduler_lr_factor),  
                        )
```

## 4. Training

```python
history = trainer.fit(epochs=1000, batch_size=train_config.batch_size)
```

## 5. Evaluation

```python
trainer.evaluate(model_checkpoint="best",
                test_data=dataset_test.train_dataset, # all test data is in the train_dataset of the test dataset bundle
                val_format_configuration=ChosenMetricsConfig(scatter_color="orange"),
                test_format_configuration=ChosenMetricsConfig(scatter_color="fuchsia"),
                )
```

## Explanation

```python
trainer.explain_captum(explain_dataset=dataset_test.train_dataset,
                    n_samples=1000,
                    n_steps=200)
```

```python
trainer.explain_shap(explain_dataset=dataset_test.train_dataset,
                    n_samples=1000,
                    explainer_type="deep")
```

## 6. Save artifacts

```python
# Dataset artifacts
dataset_train.save_artifacts(TRAIN_ARTIFACTS_DIR)

# Model artifacts
model.save_architecture(TRAIN_ARTIFACTS_DIR)
inspect_model_architecture(model=model, save_dir=TRAIN_ARTIFACTS_DIR)

# FeatureSchema
schema.to_json(TRAIN_ARTIFACTS_DIR)

# Train log
train_logger(train_config=train_config,
             model_parameters=model_params,
             train_history=history,
             save_directory=TRAIN_ARTIFACTS_DIR)
```

## 7. Finalize Deep Learning

```python
trainer.finalize_model_training(finalize_config=ChosenFinalizer(filename=train_config.finalized_filename,
                                                                target_name=train_config.targets))
```
