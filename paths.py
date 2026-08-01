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
PM.engineering_train_file = PM.engineering / "engineered_train.csv"
PM.engineering_test_file = PM.engineering / "engineered_test.csv"

### Train Data
PM.train = PM.results / "2 Train Data"
PM.train_dataset = PM.train / "Train-Validation Dataset"
PM.test_dataset = PM.train / "Test Dataset"

### Regression
PM.regression = PM.results / "3 Regression"


# 3. Make directories and check status
PM.make_dirs()

if __name__ == "__main__":
    PM.status()
