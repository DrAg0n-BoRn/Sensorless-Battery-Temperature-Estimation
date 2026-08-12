# Raw data columns
class RawColumns:
    STRATEGY = "Strategy"
    CELL_ID = "Cell_ID"
    TIME = "time_s"
    CURRENT = "current_A"
    VOLTAGE = "voltage_V"
    CELL_TEMPERATURE = "cell_temperature_C"
    ENVIRONMENT_TEMPERATURE = "environment_temperature_C"

# Engineered data columns
class EngineeredColumns:
    STRATEGY = "Strategy"
    # TIME = "Time (h)"
    CURRENT = "Current (A)"
    VOLTAGE = "Voltage (V)"
    CELL_TEMPERATURE = "Cell Temperature (K)"
    ENVIRONMENT_TEMPERATURE = "Environment Temperature (K)"
    
TARGET = EngineeredColumns.CELL_TEMPERATURE

# Strategies: BMP, BMR, SPM
TEST_SUBSET_STRATEGY = "SPM"

# Sequence
SEQUENCE_LENGTH = 28  # From feature engineering plots (temporal analysis)

# Minimum and maximum temperature values for PILF
MIN_ENVIRONMENT_TEMPERATURE = 295.25  # Minimum environment temperature in the whole dataset in Kelvin
MAX_CELL_TEMPERATURE = 311.39  # Maximum cell temperature in the whole dataset in Kelvin
