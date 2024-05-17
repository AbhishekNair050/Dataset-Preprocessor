# DatasetProcessor

DatasetProcessor is a Python library for preprocessing datasets for machine learning. It provides functionalities to handle missing values, encode categorical features, normalize numerical features, select important features, and more.

## Features

- **Read Data**: Load data from a CSV file.
- **Handle Missing Values**: Fill or drop missing values.
- **Encode Categorical Features**: Encode categorical features using label encoding.
- **Normalize Numerical Features**: Normalize numerical features using standard scaling or min-max scaling.
- **Select Features**: Perform feature selection based on correlation with the target variable.
- **Drop Useless Columns**: Drop common useless columns from the dataset.
- **Save and Load Preprocessed Data**: Save preprocessed data to a CSV file and load it back.
- **Generate Report**: Generate a report summarizing the preprocessing steps.

## Installation

You can install DatasetProcessor using pip:

```bash
pip install DatasetProcessor
```

## Usage

Here is an example of how to use the DatasetProcessor library:

```python
from my_preprocessor.preprocessor import DatasetProcessor

# Initialize the preprocessor
preprocessor = DatasetProcessor()

# Read data
data = preprocessor.read_data("data.csv")

# Preprocess data
processed_data = preprocessor.preprocess_data(
    file_path="data.csv",
    target_column="target",
    handle_missing="fill",
    encode_categorical=True,
    normalize_numerical="standard",
    feature_selection_method="correlation",
    n_features_to_select=5,
    variance_ratio_threshold=0.1
)

# Save preprocessed data
preprocessor.save_preprocessed_data(processed_data, "processed_data.csv")

# Load preprocessed data
loaded_data = preprocessor.load_preprocessed_data("processed_data.csv")

# Print the first few rows of the preprocessed data
print(loaded_data.head())

# Generate a report
original_shape = data.shape
report = preprocessor.get_report(processed_data, original_shape)
print(report)
```

## Methods

### `read_data(file_path)`

Reads data from a CSV file.

- **Args:**

  - `file_path` (str): Path to the CSV file.

- **Returns:**
  - `pandas.DataFrame`: The loaded data.

### `handle_missing_values(data, strategy="fill")`

Handles missing values in the dataset.

- **Args:**

  - `data` (pandas.DataFrame): The input data.
  - `strategy` (str, optional): The strategy to handle missing values. Options are `'fill'` (default) and `'drop'`.

- **Returns:**
  - `pandas.DataFrame`: The data with missing values handled.

### `encode_categorical_features(data)`

Encodes categorical features using label encoding.

- **Args:**

  - `data` (pandas.DataFrame): The input data.

- **Returns:**
  - `pandas.DataFrame`: The data with categorical features encoded.

### `normalize_numerical_features(data, norm_type="standard", variance_ratio_threshold=0.1)`

Normalizes numerical features with high variance relative to other numerical features.

- **Args:**

  - `data` (pandas.DataFrame): The input data.
  - `norm_type` (str, optional): The normalization type. Options are `'standard'` (default) and `'minmax'`.
  - `variance_ratio_threshold` (float, optional): The threshold for the ratio of feature variance to the maximum variance among numerical features. Default is `0.1`.

- **Returns:**
  - `pandas.DataFrame`: The data with numerical features normalized.

### `select_features(data, target_column, n_features_to_select=None)`

Performs feature selection based on correlation with the target variable.

- **Args:**

  - `data` (pandas.DataFrame): The input data.
  - `target_column` (str): The name of the target column.
  - `n_features_to_select` (int, optional): The number of features to select. If `None`, all features will be kept.

- **Returns:**
  - `pandas.DataFrame`: The data with selected features.

### `drop_useless_columns(data, useless_columns=None)`

Drops useless columns from the dataset.

- **Args:**

  - `data` (pandas.DataFrame): The input data.
  - `useless_columns` (list, optional): A list of column names to drop. If `None`, a default list of common useless columns will be used.

- **Returns:**
  - `pandas.DataFrame`: The data with useless columns dropped.

### `preprocess_data(file_path, target_column, handle_missing="fill", encode_categorical=True, normalize_numerical="standard", feature_selection_method="correlation", n_features_to_select=None, variance_ratio_threshold=0.1)`

Preprocesses the dataset for machine learning.

- **Args:**

  - `file_path` (str): Path to the CSV file.
  - `target_column` (str): The name of the target column.
  - `handle_missing` (str, optional): Strategy to handle missing values. Default is `'fill'`.
  - `encode_categorical` (bool, optional): Whether to encode categorical features. Default is `True`.
  - `normalize_numerical` (str, optional): The normalization type for numerical features. Default is `'standard'`.
  - `feature_selection_method` (str, optional): The feature selection method to use. Options are `'correlation'` (default).
  - `n_features_to_select` (int, optional): The number of features to select. If `None`, all features will be kept.
  - `variance_ratio_threshold` (float, optional): The threshold for the ratio of feature variance to the maximum variance among numerical features. Default is `0.1`.

- **Returns:**
  - `pandas.DataFrame`: The preprocessed data.

### `save_preprocessed_data(data, file_path)`

Saves the preprocessed data to a file.

- **Args:**
  - `data` (pandas.DataFrame): The preprocessed data.
  - `file_path` (str): The path to save the preprocessed data.

### `load_preprocessed_data(file_path)`

Loads the preprocessed data from a file.

- **Args:**

  - `file_path` (str): The path to load the preprocessed data from.

- **Returns:**
  - `pandas.DataFrame`: The loaded preprocessed data.

### `get_report(data, original_shape)`

Generates a report summarizing the preprocessing steps applied to the dataset.

- **Args:**

  - `data` (pandas.DataFrame): The preprocessed data.
  - `original_shape` (tuple): The original shape of the dataset before preprocessing.

- **Returns:**
  - `str`: The report summarizing the preprocessing steps.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## Author

Abhishek Nair - [abhishek,naiir@gmail.com](mailto:abhishek,naiir@gmail.com)
Priyanshi Furiya - [furiyapriyanshi@gmail.com](furiyapriyanshi@gmail.com)
