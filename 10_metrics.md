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
from dragon.utilities import load_dataframe, merge_dataframes_vertical, save_dataframe_filename
from dragon.path_manager import list_subdirectories

from pathlib import Path
from paths import PM
```

```python
# base metric paths
metric_paths = [PM.pilf_v2, PM.pilf_v3]
```

```python
# get subdirectories for each metric path (models)
all_models: dict[str, Path] = dict()
for root_path in metric_paths:
    subdirs_dict = list_subdirectories(root_path)
    all_models.update(subdirs_dict)
```

```python
# pop empty generic directory if it exists
all_models.pop(PM.pilf_v2_lstm.name, None)
all_models.pop(PM.pilf_v3_lstm.name, None)
```

```python
# get metrics csv from each model subdirectory
all_test_metrics_df = []
all_val_metrics_df = []

for model_name, model_path in all_models.items():
    test_metrics_path = model_path / "Evaluation_Test" / "Cell_Temperature_K" / "regression_report.csv"
    val_metrics_path = model_path / "Evaluation_Validation" / "Cell_Temperature_K" / "regression_report.csv"
    
    # load dataframe and append model name to dataframe
    test_metrics_df, _ = load_dataframe(test_metrics_path, verbose=False)
    test_metrics_df['Model'] = model_name
    all_test_metrics_df.append(test_metrics_df)
    
    val_metrics_df, _ = load_dataframe(val_metrics_path, verbose=False)
    val_metrics_df['Model'] = model_name
    all_val_metrics_df.append(val_metrics_df)
    
print(len(all_test_metrics_df), len(all_val_metrics_df))
```

```python
# merge all test metrics dataframes into one
merged_test_metrics_df = merge_dataframes_vertical(all_test_metrics_df, reset_index=True)

# reorder columns to have Model first
cols = merged_test_metrics_df.columns.tolist()
cols.remove('Model')
merged_test_metrics_df = merged_test_metrics_df[['Model'] + cols]
```

```python
merged_test_metrics_df
```

```python
# merge all validation metrics dataframes into one
merged_val_metrics_df = merge_dataframes_vertical(all_val_metrics_df, reset_index=True)

# reorder columns to have Model first
cols = merged_val_metrics_df.columns.tolist()
cols.remove('Model')
merged_val_metrics_df = merged_val_metrics_df[['Model'] + cols]
```

```python
merged_val_metrics_df
```

```python
# save metrics dataframes to csv
save_dataframe_filename(df=merged_test_metrics_df, save_dir=PM.metrics, filename="merged_test_metrics.csv")
```

```python
save_dataframe_filename(df=merged_val_metrics_df, save_dir=PM.metrics, filename="merged_val_metrics.csv")
```
