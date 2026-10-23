---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.6
  kernelspec:
    display_name: sensorless-battery-temperature-estimation (3.14.9)
    language: python
    name: python3
---

```python
from dragon.datasetmaster import DragonDatasetSequenceExogenous as ChosenDataset
from dragon.trainer import DragonSequenceTrainer as ChosenTrainer
from dragon.models_sequence import DragonSequenceTransformer as ChosenModel
from dragon.configuration import (
    FormatExogenousSequenceValueMetrics as ChosenMetricsConfig, 
    FinalizeExogenousSequenceValue as ChosenFinalizer,
    FormatRegressionMetrics
)

from dragon.configuration import DragonTrainingConfig
from dragon.callbacks import DragonPatienceEarlyStopping, DragonPlateauScheduler
from dragon.ML_utilities import build_optimizer_params
from dragon.schema import FeatureSchema
from dragon.keys import TaskKeys
from torch.optim import AdamW

from paths import PM
from constants import TARGET, SEQUENCE_LENGTH
```

```python
!yes | plotly_get_chrome
```

```python
SCHEMA_PATH = PM.engineering_sequence
TRAIN_DATASET_PATH = PM.sequence_train_dataset
TEST_DATASET_PATH = PM.sequence_test_dataset
TRAIN_ARTIFACTS_DIR = PM.sequence_transformer
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
    finalized_filename = "Transformer_exo_seq_to_val",
    initial_learning_rate=0.005,
    batch_size=256,
    device = "cuda:0",
    task = TaskKeys.EXOGENOUS_SEQUENCE_VALUE,
    
    target=TARGET,
    weight_decay=0.5,
    early_stop_patience=25,
    scheduler_patience=6,
    scheduler_lr_factor=0.6,
    monitor_metric="Validation Loss"
)
```

## 3. Model and Trainer

```python
model = ChosenModel(
    schema=schema,
    targets=[TARGET],
    prediction_mode=train_config.task,
    sequence_length=SEQUENCE_LENGTH,
    embedding_dim=128,
    heads=2,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.5,
)


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
val_metrics = FormatRegressionMetrics(scatter_color="mediumturquoise", boxplot_palette="Accent")
test_metrics = FormatRegressionMetrics(scatter_color="deeppink", boxplot_palette="Dark2")

trainer.evaluate(model_checkpoint="best",
                test_data=dataset_test.train_dataset, # all test data is in the 'train_dataset' of the test dataset bundle
                val_format_configuration=ChosenMetricsConfig(regression_config=val_metrics),
                test_format_configuration=ChosenMetricsConfig(regression_config=test_metrics)
                )
```

## Explanation

```python
trainer.explain_captum(explain_dataset=dataset_test.train_dataset,
                    n_samples=1000,
                    n_steps=200)
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
                                                                target_types=dataset_train.target_types,
                                                                last_dataset_sequence=dataset_train.last_dataset_sequence,
                                                                target=train_config.target)
                                )
```
