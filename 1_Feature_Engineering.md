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

# Feature Engineering

```python
from ml_tools.data_exploration import (summarize_dataframe,
                                       split_features_targets,
                                       plot_value_distributions_multi,
                                       plot_numeric_overview_boxplot_macro,
                                       plot_continuous_vs_target,
                                       plot_correlation_heatmap,
                                       plot_pairgrid_continuous_vs_target,
                                       finalize_feature_schema,
                                       show_null_columns)

from ml_tools.utilities import load_dataframe, save_dataframe_with_schema, merge_dataframes

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
summarize_dataframe(df_clean)
```

```python
# Select subset where cell temperature is not null
df_clean = df_clean[df_clean[RawColumns.CELL_TEMPERATURE].notnull()]
```

```python
# Transform time to hours, centigrade to Kelvin
df_clean[RawColumns.TIME] = df_clean[RawColumns.TIME] / 60 / 60
df_clean[RawColumns.ENVIRONMENT_TEMPERATURE] = df_clean[RawColumns.ENVIRONMENT_TEMPERATURE] + 273.15
df_clean[RawColumns.CELL_TEMPERATURE] = df_clean[RawColumns.CELL_TEMPERATURE] + 273.15
```

```python
# rename columns from RawColumns to EngineeredColumns
df_clean.rename(columns={
    RawColumns.TIME: EngineeredColumns.TIME,
    RawColumns.CURRENT: EngineeredColumns.CURRENT,
    RawColumns.VOLTAGE: EngineeredColumns.VOLTAGE,
    RawColumns.ENVIRONMENT_TEMPERATURE: EngineeredColumns.ENVIRONMENT_TEMPERATURE,
    RawColumns.CELL_TEMPERATURE: EngineeredColumns.CELL_TEMPERATURE,
}, inplace=True)
```

```python
summarize_dataframe(df_clean)
```

## 3. Split train and test sets

```python
# Leave-One-Group-Out Cross-Validation
df_test = df_clean[df_clean[EngineeredColumns.STRATEGY] == TEST_SUBSET_STRATEGY]
df_train = df_clean[df_clean[EngineeredColumns.STRATEGY] != TEST_SUBSET_STRATEGY]
```

```python
summarize_dataframe(df_train)
```

```python
summarize_dataframe(df_test)
```

## 4. Split Features and Targets

```python
# Drop Strategy column from train and test subsets
df_train.drop(columns=[EngineeredColumns.STRATEGY], inplace=True)
df_test.drop(columns=[EngineeredColumns.STRATEGY], inplace=True)
```

```python
# use float values with maximum 6 decimal places
df_train = df_train.round(6)
df_test = df_test.round(6)
```

```python
df_train_features, df_train_targets = split_features_targets(df_train, targets=[TARGET])
```

```python
df_test_features, df_test_targets = split_features_targets(df_test, targets=[TARGET])
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

## 6. Create Feature Schema

```python
# assert that the train and test subsets have the same columns
assert set(df_train_features.columns) == set(df_test_features.columns), "Train and test subsets have different columns"
```

```python
schema = finalize_feature_schema(df_features=df_train_features, categorical_mappings=None)
```

## 7. Merge dataframes in expected column order

```python
df_train_final = merge_dataframes(df_train_features, df_train_targets, direction="horizontal")
```

```python
df_test_final = merge_dataframes(df_test_features, df_test_targets, direction="horizontal")
```

```python
show_null_columns(df_train_final)
```

```python
show_null_columns(df_test_final)
```

## 7. Save Artifacts

```python
save_dataframe_with_schema(df=df_train_final, full_path=PM.engineering_train_file, schema=schema)
```

```python
save_dataframe_with_schema(df=df_test_final, full_path=PM.engineering_test_file, schema=schema)
```

```python
schema.to_json(PM.engineering)
schema.save_artifacts(PM.engineering)
```
