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

# Feature Engineering

```python
from dragon.data_exploration import (summarize_dataframe,
                                       split_features_targets,
                                       plot_value_distributions_multi,
                                       plot_numeric_overview_boxplot_macro,
                                       plot_continuous_vs_target,
                                       plot_correlation_heatmap,
                                       plot_pairgrid_continuous_vs_target,
                                       plot_target_temporal_analysis,
                                       show_null_columns)
from dragon.outlier_detection import drop_outliers_rule
from dragon.utilities import load_dataframe, save_dataframe_with_schema, merge_dataframes_horizontal
from dragon.schema import finalize_feature_schema

from paths import PM
from constants import TARGET, RawColumns, EngineeredColumns, TEST_SUBSET_STRATEGY
```

## 1. Load Data

```python
df_raw, _ = load_dataframe(df_path=PM.raw_file, kind="pandas")
```

```python
summarize_dataframe(df_raw)
```

## 2. Data Cleaning

```python
# drop ID column
df_clean = df_raw.drop(columns=[RawColumns.CELL_ID])

# forward fill missing values for the environment temperature (AC controlled)
df_clean[RawColumns.ENVIRONMENT_TEMPERATURE] = df_clean[RawColumns.ENVIRONMENT_TEMPERATURE].ffill()
```

```python
# Select subset where cell temperature is not null
df_clean = df_clean[df_clean[RawColumns.CELL_TEMPERATURE].notnull()]
```

```python
summarize_dataframe(df_clean)
```

```python
# Transform centigrade to Kelvin
df_clean[RawColumns.ENVIRONMENT_TEMPERATURE] = df_clean[RawColumns.ENVIRONMENT_TEMPERATURE] + 273.15
df_clean[RawColumns.CELL_TEMPERATURE] = df_clean[RawColumns.CELL_TEMPERATURE] + 273.15
```

```python
# Order dataset by time
df_clean.sort_values(by=[RawColumns.TIME], inplace=True)
df_clean.head(15)
```

```python
# Drop time column and round up to 6 decimal places
df_clean = df_clean.drop(columns=[RawColumns.TIME]).round(6)
```

```python
# rename columns from RawColumns to EngineeredColumns
df_clean.rename(columns={
    RawColumns.CURRENT: EngineeredColumns.CURRENT,
    RawColumns.VOLTAGE: EngineeredColumns.VOLTAGE,
    RawColumns.ENVIRONMENT_TEMPERATURE: EngineeredColumns.ENVIRONMENT_TEMPERATURE,
    RawColumns.CELL_TEMPERATURE: EngineeredColumns.CELL_TEMPERATURE,
}, inplace=True)
```

```python
summarize_dataframe(df_clean)
```

```python
# Identify faulty entries where cell temperature > sensible threshold
temperature_threshold = 312  # Kelvin
suspect_entries = df_clean[df_clean[EngineeredColumns.CELL_TEMPERATURE] >= temperature_threshold]
len(suspect_entries)
```

```python
suspect_entries
```

```python
df_clean = drop_outliers_rule(df=df_clean,
                              bounds_dict={EngineeredColumns.CELL_TEMPERATURE: (0, temperature_threshold)},
                              reset_index=True)
```

```python
summarize_dataframe(df_clean)
```

```python
# minimum environment temperature
df_clean[EngineeredColumns.ENVIRONMENT_TEMPERATURE].min()
```

```python
# maximum cell temperature
df_clean[EngineeredColumns.CELL_TEMPERATURE].max()
```

Store minimum and maximum temperature values in `constants.py` for later use in custom PILF.


## 3. Split train and test sets

```python
# Leave-One-Group-Out Cross-Validation
df_test = df_clean[df_clean[EngineeredColumns.STRATEGY] == TEST_SUBSET_STRATEGY]
df_train = df_clean[df_clean[EngineeredColumns.STRATEGY] != TEST_SUBSET_STRATEGY]
```

```python
# Drop Strategy column after spliting
df_train.drop(columns=[EngineeredColumns.STRATEGY], inplace=True)
df_test.drop(columns=[EngineeredColumns.STRATEGY], inplace=True)
```

```python
summarize_dataframe(df_train)
```

```python
summarize_dataframe(df_test)
```

## 4. Split Features and Targets

```python
df_train_features, df_train_targets = split_features_targets(df_train, targets=[TARGET])
```

```python
df_test_features, df_test_targets = split_features_targets(df_test, targets=[TARGET])
```

```python
# assert that the train and test subsets have the same columns and order
assert list(df_train_features.columns) == list(df_test_features.columns), "Train and test feature subsets have different columns"
```

## 5. Plots

```python
_named_datasets = {
    "Train-Validation Subset": df_train,
    "Test Subset": df_test
}

plot_value_distributions_multi(named_dataframes=_named_datasets,
                               save_dir=PM.engineering_plots_train,
                               mode="percentage")
```

```python
plot_numeric_overview_boxplot_macro(df=df_train,
                                    save_dir=PM.engineering_plots_train,
                                    plot_title="Train-Validation Subset",
                                    handle_zero_variance="constant",
                                    palette="Set2")
```

```python
plot_numeric_overview_boxplot_macro(df=df_test,
                                    save_dir=PM.engineering_plots_test,
                                    plot_title="Test Subset",
                                    handle_zero_variance="constant",
                                    palette="tab10")
```

```python
plot_correlation_heatmap(df=df_train, 
                         plot_title="Train-Validation Subset", 
                         save_dir=PM.engineering_plots_train,
                         cmap="coolwarm")
```

```python
plot_correlation_heatmap(df=df_test, 
                         plot_title="Test Subset", 
                         save_dir=PM.engineering_plots_test,
                         cmap="afmhot")
```

```python
plot_continuous_vs_target(df_continuous=df_train_features, 
                          df_targets=df_train_targets, 
                          save_dir=PM.engineering_plots_train,
                          color="tab:brown")
```

```python
plot_continuous_vs_target(df_continuous=df_test_features, 
                          df_targets=df_test_targets, 
                          save_dir=PM.engineering_plots_test,
                          color="tab:purple")
```

```python
plot_pairgrid_continuous_vs_target(df_continuous=df_train_features,
                                   df_targets=df_train_targets,
                                   save_dir=PM.engineering_plots_train,
                                   color="tab:orange",
                                   hexbin_lower=True,
                                   log_diagonal=True)
```

```python
plot_pairgrid_continuous_vs_target(df_continuous=df_test_features,
                                   df_targets=df_test_targets,
                                   save_dir=PM.engineering_plots_test,
                                   color="tab:purple",
                                   hexbin_lower=True,
                                   log_diagonal=True)
```

```python
# sequence to value analysis (testing several lag lengths to visually determine the optimal value)
plot_target_temporal_analysis(df=df_train,
                              continuous_targets=[TARGET],
                              categorical_targets=None,
                              save_dir=PM.engineering_plots_train,
                              max_lag=50)
```

Optimal sequence length for the sequence-to-value added to `constants.py`. 

This is based on the analysis of the temporal dependencies in the data.


## 6. Merge dataframes

```python
df_train_final = merge_dataframes_horizontal([df_train_features, df_train_targets])
```

```python
df_test_final = merge_dataframes_horizontal([df_test_features, df_test_targets])
```

```python
# assert that the train and test subsets have the same columns and order
assert list(df_train_final.columns) == list(df_test_final.columns), "Train and test subsets have different columns"
```

```python
show_null_columns(df_train_final)
```

```python
show_null_columns(df_test_final)
```

## 7. Create Feature Schema

```python
schema_regression = finalize_feature_schema(features=df_test_features, categorical_mappings=None)
```

```python
# target must be included for sequence-to-value pipeline
schema_sequence = finalize_feature_schema(features=df_test_final, categorical_mappings=None)
```

## 8. Save Artifacts


Regression

```python
save_dataframe_with_schema(df=df_train_final, full_path=PM.engineering_train_file, schema=schema_regression)
```

```python
save_dataframe_with_schema(df=df_test_final, full_path=PM.engineering_test_file, schema=schema_regression)
```

```python
schema_regression.to_json(PM.engineering_regression)
schema_regression.save_artifacts(PM.engineering_regression)
```

Sequence

```python
save_dataframe_with_schema(df=df_train_final, full_path=PM.engineering_sequence_train_file, schema=schema_sequence)
```

```python
save_dataframe_with_schema(df=df_test_final, full_path=PM.engineering_sequence_test_file, schema=schema_sequence)
```

```python
schema_sequence.to_json(PM.engineering_sequence)
schema_sequence.save_artifacts(PM.engineering_sequence)
```
