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
from dragon.datasetmaster import DragonDataset as ChosenDataset
from dragon.utilities import load_dataframe_with_schema
from dragon.schema import FeatureSchema
from dragon.keys import TaskKeys

from paths import PM
```

```python
SCHEMA_PATH = PM.engineering_regression

TRAIN_DATASET_INPUT_FILE = PM.engineering_train_file
TRAIN_DATASET_OUTPUT_DIR = PM.train_dataset

TEST_DATASET_INPUT_FILE = PM.engineering_test_file
TEST_DATASET_OUTPUT_DIR = PM.test_dataset
```

## 1. Load Schema and Dataframe

```python
schema = FeatureSchema.from_json(SCHEMA_PATH)
```

```python
df_train, _ = load_dataframe_with_schema(df_path=TRAIN_DATASET_INPUT_FILE, 
                                        schema=schema)
```

```python
df_test, _ = load_dataframe_with_schema(df_path=TEST_DATASET_INPUT_FILE,
                                        schema=schema)
```

## 2. Make Dataset Class

```python
dataset_train = ChosenDataset(pandas_df=df_train,
                        schema=schema,
                        kind=TaskKeys.REGRESSION,
                        feature_scaler="fit",
                        target_scaler="fit",
                        validation_size=0.2,
                        test_size=0, # no test set is needed since we already have a separate test set
                        random_state=42
                        )
```

```python
assert dataset_train.feature_scaler is not None, "Feature scaler is not fitted."
assert dataset_train.target_scaler is not None, "Target scaler is not fitted."

dataset_test = ChosenDataset(pandas_df=df_test,
                        schema=schema,
                        kind=TaskKeys.REGRESSION,
                        feature_scaler=dataset_train.feature_scaler, # use the same scaler as the training set
                        target_scaler=dataset_train.target_scaler, # use the same scaler as the training set
                        validation_size=0, # not needed for the test set
                        test_size=0, # not needed for the test set
                        random_state=42
                        )
```

```python
# inspect the dataset
dataset_train
```

```python
dataset_test
```

## 3. Save Dataset Class

```python
dataset_train.save_dataset_bundle(directory=TRAIN_DATASET_OUTPUT_DIR)
```

```python
dataset_test.save_dataset_bundle(directory=TEST_DATASET_OUTPUT_DIR)
```
