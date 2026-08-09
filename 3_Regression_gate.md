---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: sensorless-battery-temperature-estimation (3.12.12)
    language: python
    name: python3
---

```python
from ml_tools.ML_datasetmaster import DragonDataset as ChosenDataset
from ml_tools.ML_trainer import DragonTrainer as ChosenTrainer
from ml_tools.ML_models import DragonGateModel as ChosenModel
from ml_tools.ML_configuration import (
    FormatRegressionMetrics as ChosenMetricsConfig, 
    FinalizeRegression as ChosenFinalizer  
)

from ml_tools.ML_configuration import DragonTrainingConfig
from ml_tools.ML_callbacks import DragonPatienceEarlyStopping, DragonPlateauScheduler
from ml_tools.ML_utilities import build_optimizer_params
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
SCHEMA_PATH = PM.engineering_regression
TRAIN_DATASET_PATH = PM.train_dataset
TEST_DATASET_PATH = PM.test_dataset
TRAIN_ARTIFACTS_DIR = PM.regression_gate
```

## 1. Load Schema and Dataset

```python
schema = FeatureSchema.from_json(SCHEMA_PATH)
```

```python
dataset_train = ChosenDataset.from_bundle(TRAIN_DATASET_PATH, verbose=True)
```

```python
dataset_test = ChosenDataset.from_bundle(TEST_DATASET_PATH, verbose=True)
```

## 2. Train Config

```python
train_config = DragonTrainingConfig(
    initial_learning_rate=0.003,
    batch_size=1024,
    task = TaskKeys.REGRESSION,
    device = "cuda:0",
    finalized_filename = "gate",
    
    targets=TARGET,
    weight_decay=0.01,
    early_stop_patience=25,
    scheduler_patience=4,
    scheduler_lr_factor=0.6,
    monitor_metric="Validation Loss"
)
```

## 3. Model and Trainer

```python
model = ChosenModel(
    schema=schema,
    out_targets=dataset_train.number_of_targets,
    embedding_dim=64,
    gflu_stages = 6,
    gflu_dropout = 0.1,
    num_trees = 10,
    tree_depth = 4,
    tree_dropout = 0.1,
    chain_trees = True,
    tree_wise_attention = True,
    tree_wise_attention_dropout = 0.1,
    binning_activation = "entmoid",
    feature_mask_function = "entmax",
    share_head_weights = True,
    batch_norm_continuous = True
)

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
                        early_stopping_callback=DragonPatienceEarlyStopping(patience=train_config.early_stop_patience, 
                                                                            monitor=train_config.monitor_metric),
                        lr_scheduler_callback=DragonPlateauScheduler(monitor=train_config.monitor_metric,
                                                                     patience=train_config.scheduler_patience,
                                                                     factor=train_config.scheduler_lr_factor),  
                        checkpoint_config="default")
```

## 4. Training

```python
trainer.fit(epochs=1000, batch_size=train_config.batch_size)
```

## 5. Evaluation

```python
trainer.evaluate(model_checkpoint="best",
                test_data=dataset_test.train_dataset, # all test data is in the train_dataset of the test dataset bundle
                val_format_configuration=ChosenMetricsConfig(scatter_color="orange", boxplot_palette="husl"),
                test_format_configuration=ChosenMetricsConfig(scatter_color="fuchsia", boxplot_palette="Set2")
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
                    n_samples=300,
                    explainer_type="kernel")
```

## 6. Save artifacts

```python
# Dataset artifacts
dataset_train.save_artifacts(TRAIN_ARTIFACTS_DIR)

# Model artifacts
model.save_architecture(TRAIN_ARTIFACTS_DIR)

# Train log
trainer.save_training_log(train_config=train_config)
```

## 7. Finalize Deep Learning

```python
trainer.finalize_model_training(finalize_config=ChosenFinalizer(filename=train_config.finalized_filename,
                                                                target_name=train_config.targets))
```
