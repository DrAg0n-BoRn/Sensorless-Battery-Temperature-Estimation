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
from ml_tools.ML_datasetmaster import DragonDatasetSequenceExogenous as ChosenDataset
from ml_tools.utilities import load_dataframe_with_schema
from ml_tools.schema import FeatureSchema
from ml_tools.keys import TaskKeys

from paths import PM
from constants import TARGET, SEQUENCE_LENGTH
```

```python
SCHEMA_PATH = PM.engineering_sequence

TRAIN_DATASET_INPUT_FILE = PM.engineering_sequence_train_file
TRAIN_DATASET_OUTPUT_DIR = PM.sequence_train_dataset
TRAIN_DATASET_OUTPUT_DIR_SEQSEQ = PM.seq_seq_train_dataset

TEST_DATASET_INPUT_FILE = PM.engineering_sequence_test_file
TEST_DATASET_OUTPUT_DIR = PM.sequence_test_dataset
TEST_DATASET_OUTPUT_DIR_SEQSEQ = PM.seq_seq_test_dataset
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

## 2. Make Dataset Class (Sequence-to-Value)

```python
dataset_train = ChosenDataset(pandas_df=df_train,
                            targets=[TARGET],
                            schema=schema,
                            prediction_mode=TaskKeys.EXOGENOUS_SEQUENCE_VALUE,
                            sequence_length=SEQUENCE_LENGTH,
                            feature_scaler="fit",
                            validation_size=0.2,
                            test_size=0, # no test set is needed since we already have a separate test set
                            verbose=3
                        )
```

```python
assert dataset_train.feature_scaler is not None, "Feature scaler is not fitted."

dataset_test = ChosenDataset(pandas_df=df_test,
                            targets=[TARGET],
                            schema=schema,
                            prediction_mode=TaskKeys.EXOGENOUS_SEQUENCE_VALUE,
                            sequence_length=SEQUENCE_LENGTH,
                            feature_scaler=dataset_train.feature_scaler, # use the same scaler as the training set
                            validation_size=0, # not needed for the test set
                            test_size=0, # not needed for the test set
                            verbose=3
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

## 4. Make Dataset Class (Sequence-to-Sequence)

```python
dataset_train_seqseq = ChosenDataset(pandas_df=df_train,
                                    targets=[TARGET],
                                    schema=schema,
                                    prediction_mode=TaskKeys.EXOGENOUS_SEQUENCE_SEQUENCE,
                                    sequence_length=SEQUENCE_LENGTH,
                                    feature_scaler="fit",
                                    validation_size=0.2,
                                    test_size=0, # no test set is needed since we already have a separate test set
                                    verbose=3
                                )
```

```python
assert dataset_train_seqseq.feature_scaler is not None, "Feature scaler is not fitted."

dataset_test_seqseq = ChosenDataset(pandas_df=df_test,
                                    targets=[TARGET],
                                    schema=schema,
                                    prediction_mode=TaskKeys.EXOGENOUS_SEQUENCE_SEQUENCE,
                                    sequence_length=SEQUENCE_LENGTH,
                                    feature_scaler=dataset_train_seqseq.feature_scaler, # use the same scaler as the training set
                                    validation_size=0, # not needed for the test set
                                    test_size=0, # not needed for the test set
                                    verbose=3
                                )
```

## 5. Save Dataset Class

```python
dataset_train_seqseq.save_dataset_bundle(directory=TRAIN_DATASET_OUTPUT_DIR_SEQSEQ)
```

```python
dataset_test_seqseq.save_dataset_bundle(directory=TEST_DATASET_OUTPUT_DIR_SEQSEQ)
```
