from ml_tools.path_manager import DragonPathManager


# 1. Initialize the PathManager using this file as the anchor, adding base directories.
PM = DragonPathManager(
    anchor_file=__file__,
    base_directories=["start_data", "results", "backups"]
)

# 2. Define directories and files.
### Base files
PM.raw_file = PM.start_data / "all_profile_data.csv"

### Feature Engineering
PM.engineering = PM.results / "1 Feature Engineering"
PM.engineering_plots_train = PM.engineering / "Engineering Plots Train-Validation"
PM.engineering_plots_test = PM.engineering / "Engineering Plots Test"

PM.engineering_regression = PM.engineering / "Engineering Regression"
PM.engineering_train_file = PM.engineering_regression / "engineered_train.csv"
PM.engineering_test_file = PM.engineering_regression / "engineered_test.csv"

PM.engineering_sequence = PM.engineering / "Engineering Sequence"
PM.engineering_sequence_train_file = PM.engineering_sequence / "engineered_sequence_train.csv"
PM.engineering_sequence_test_file = PM.engineering_sequence / "engineered_sequence_test.csv"

### Train Data
PM.train = PM.results / "2 Train Data"
PM.train_dataset = PM.train / "Train-Validation Dataset"
PM.test_dataset = PM.train / "Test Dataset"

PM.sequence_train_dataset = PM.train / "Sequence-to-Value Train-Validation Dataset"
PM.sequence_test_dataset = PM.train / "Sequence-to-Value Test Dataset"

PM.seq_seq_train_dataset = PM.train / "Sequence-to-Sequence Train-Validation Dataset"
PM.seq_seq_test_dataset = PM.train / "Sequence-to-Sequence Test Dataset"

### Regression
PM.regression = PM.results / "3 Regression"
PM.regression_node = PM.regression / "NODE"
PM.regression_autoint = PM.regression / "AutoInt"
PM.regression_tabnet = PM.regression / "TabNet"
PM.regression_gate = PM.regression / "GATE"

### Sequence-to-Value
PM.sequence = PM.results / "4 Sequence-to-Value"
PM.sequence_lstm = PM.sequence / "LSTM"
PM.sequence_transformer = PM.sequence / "Transformer"

### Sequence-to-Sequence
PM.seq_seq = PM.results / "5 Sequence-to-Sequence"
PM.seq_seq_lstm = PM.seq_seq / "LSTM"
PM.seq_seq_transformer = PM.seq_seq / "Transformer"

# physics informed ML
PM.pilf = PM.results / "6 Physics-Informed ML"
PM.pilf_lstm = PM.pilf / "LSTM-PILF"

# physics informed ML V2
PM.pilf_v2 = PM.results / "7 Physics-Informed ML V2"
PM.pilf_v2_lstm = PM.pilf_v2 / "LSTM-PILF-V2"

# physics informed ML V3 - Seq-Val
PM.pilf_v3 = PM.results / "8 Physics-Informed ML V3 Seq-Val"
PM.pilf_v3_lstm = PM.pilf_v3 / "LSTM-PILF-V3"

# Metrics DataFrames
PM.metrics = PM.results / "9 Metrics DataFrames"

# 3. Make directories and check status
PM.make_dirs()

if __name__ == "__main__":
    PM.status()
