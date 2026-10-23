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
from dragon.path_manager import list_files_by_extension_global, list_subdirectories

from paths import PM
```

```python
all_model_dirs = list_subdirectories(PM.pilf_v3)
```

```python
test_metrics_dfs = list()
validation_metrics_dfs = list()

for model_name, subpath in all_model_dirs.items():
    metrics_tuples = list_files_by_extension_global(directory=subpath, 
                                                    extension="csv",
                                                    verbose=1,
                                                    raise_on_empty=False)
    if len(metrics_tuples) > 0:
        for filename, filepath in metrics_tuples:
            if "report" in filename:
                df, _ = load_dataframe(filepath, verbose=False)
                df["Model"] = model_name
                if "Test" in str(filepath):
                    test_metrics_dfs.append(df)
                elif "Validation" in str(filepath):
                    validation_metrics_dfs.append(df)
```

```python
test_df = merge_dataframes_vertical(test_metrics_dfs, reset_index=True)
```

```python
validation_df = merge_dataframes_vertical(validation_metrics_dfs, reset_index=True)
```

```python
save_dataframe_filename(df=test_df, save_dir=PM.metrics_chosen, filename="test_metrics.csv")
```

```python
save_dataframe_filename(df=validation_df, save_dir=PM.metrics_chosen, filename="validation_metrics.csv")
```
