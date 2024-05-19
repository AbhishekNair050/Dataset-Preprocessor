import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import logging


class DatasetPreprocessor:
    def __init__(self, log_file="preprocess.log"):
        # Configure logging
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def read_data(self, file_path):
        """
        Reads data from a CSV file.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pandas.DataFrame: The loaded data.
        """
        self.logger.info(f"Reading data from {file_path}")
        data = pd.read_csv(file_path)
        return data

    def handle_missing_values(self, data, strategy="fill"):
        """
        Handles missing values in the dataset.

        Args:
            data (pandas.DataFrame): The input data.
            strategy (str, optional): The strategy to handle missing values.
                'fill': Fill with column mean (for numerical columns) or mode (for categorical columns).
                'drop': Drop rows with missing values.

        Returns:
            pandas.DataFrame: The data with missing values handled.
        """
        if strategy == "drop":
            self.logger.info("Handling missing values by dropping rows")
            data = data.dropna()
        else:
            self.logger.info("Handling missing values by filling")
            null_columns = data.columns[data.isnull().any()]
            for column in null_columns:
                if data[column].dtype == "object":
                    fill_value = data[column].mode()[0]
                else:
                    fill_value = data[column].mean()
                data[column].fillna(fill_value, inplace=True)
        return data

    def encode_categorical_features(self, data):
        """
        Encodes categorical features using label encoding.

        Args:
            data (pandas.DataFrame): The input data.

        Returns:
            pandas.DataFrame: The data with categorical features encoded.
        """
        self.logger.info("Encoding categorical features using label encoding")
        le = LabelEncoder()
        for column in data.columns:
            if data[column].dtype == "object":
                data[column] = le.fit_transform(data[column])
        return data

    def normalize_numerical_features(
        self, data, norm_type="standard", variance_ratio_threshold=0.1
    ):
        """
        Normalizes numerical features with high variance relative to other numerical features.

        Args:
            data (pandas.DataFrame): The input data.
            norm_type (str, optional): The normalization type.
                'standard': Standardize features by removing the mean and scaling to unit variance.
                'minmax': Scale features to the range [0, 1].
            variance_ratio_threshold (float, optional): The threshold for the ratio of feature variance
                to the maximum variance among numerical features. Features with a variance ratio above
                this threshold will be normalized.

        Returns:
            pandas.DataFrame: The data with numerical features normalized.
        """
        self.logger.info(f"Normalizing numerical features using {norm_type} scaling")

        columns_to_normalize = []
        numerical_columns = [col for col in data.columns if data[col].dtype != "object"]
        max_variance = max(data[numerical_columns].var())

        for column in numerical_columns:
            variance = data[column].var()
            variance_ratio = variance / max_variance
            if variance_ratio > variance_ratio_threshold:
                columns_to_normalize.append(column)
            else:
                self.logger.info(
                    f"Skipping normalization for column '{column}' (variance ratio: {variance_ratio:.4f})"
                )

        if columns_to_normalize:
            if norm_type == "standard":
                scaler = StandardScaler()
            elif norm_type == "minmax":
                scaler = MinMaxScaler()

            data[columns_to_normalize] = scaler.fit_transform(
                data[columns_to_normalize]
            )

        return data

    def select_features(self, data, target_column, n_features_to_select=None):
        """
        Performs feature selection based on correlation with the target variable.

        Args:
            data (pandas.DataFrame): The input data.
            target_column (str): The name of the target column.
            n_features_to_select (int, optional): The number of features to select.
                If None, all features will be kept.

        Returns:
            pandas.DataFrame: The data with selected features.
        """
        self.logger.info("Performing feature selection based on correlation")

        # Calculate correlation matrix
        correlation_matrix = data.corr()

        # Get absolute correlation with target column
        target_correlation = correlation_matrix[target_column].abs()

        # Drop target column from the correlation series
        target_correlation = target_correlation.drop(target_column)

        # Sort correlations in descending order
        sorted_correlation = target_correlation.sort_values(ascending=False)

        # Select top n features based on correlation
        if n_features_to_select is not None:
            selected_features = sorted_correlation.head(n_features_to_select).index
        else:
            selected_features = sorted_correlation.index

        selected_columns = selected_features.tolist() + [target_column]
        data = data[selected_columns]

        self.logger.info(f"Selected features: {', '.join(selected_features)}")

        return data

    def drop_useless_columns(self, data, useless_columns=None):
        """
        Drops useless columns from the dataset.

        Args:
            data (pandas.DataFrame): The input data.
            useless_columns (list, optional): A list of column names to drop.
                If None, a default list of common useless columns will be used.

        Returns:
            pandas.DataFrame: The data with useless columns dropped.
        """
        if useless_columns is None:
            useless_columns = [
                "id",
                "Unnamed",
                "index",
                "customer_id",
                "customer_name",
                "address",
                "zip_code",
                "phone",
                "email",
                "date",
                "time",
                "timestamp",
                "datetime",
                "url",
                "image",
                "file_path",
                "description",
            ]

        columns_to_drop = [
            col
            for col in data.columns
            if col.lower() in (name.lower() for name in useless_columns)
        ]

        if columns_to_drop:
            self.logger.info(f"Dropping useless columns: {', '.join(columns_to_drop)}")
            data = data.drop(columns=columns_to_drop, axis=1)

        return data

    def preprocess_data(
        self,
        file_path,
        target_column,
        handle_missing="fill",
        encode_categorical=True,
        normalize_numerical="standard",
        feature_selection_method="correlation",
        n_features_to_select=None,
        variance_ratio_threshold=0.1,
    ):
        """
        Preprocesses the dataset for machine learning.

        Args:
            file_path (str): Path to the CSV file.
            target_column (str): The name of the target column.
            handle_missing (str, optional): Strategy to handle missing values.
            encode_categorical (bool, optional): Whether to encode categorical features.
            normalize_numerical (str, optional): The normalization type for numerical features.
            feature_selection_method (str, optional): The feature selection method to use.
                'correlation': Feature selection based on correlation with the target variable.
            n_features_to_select (int, optional): The number of features to select.
                If None, all features will be kept.
            variance_ratio_threshold (float, optional): The threshold for the ratio of feature variance
                to the maximum variance among numerical features. Features with a variance ratio above
                this threshold will be normalized.

        Returns:
            pandas.DataFrame: The preprocessed data.
        """
        data = self.read_data(file_path)
        data = self.drop_useless_columns(data)
        data = self.handle_missing_values(data, strategy=handle_missing)
        data = self.normalize_numerical_features(
            data,
            norm_type=normalize_numerical,
            variance_ratio_threshold=variance_ratio_threshold,
        )
        if encode_categorical:
            data = self.encode_categorical_features(data)
        if feature_selection_method == "correlation":
            data = self.select_features(
                data,
                target_column=target_column,
                n_features_to_select=n_features_to_select,
            )
        return data

    def save_preprocessed_data(self, data, file_path):
        """
        Saves the preprocessed data to a file.

        Args:
            data (pandas.DataFrame): The preprocessed data.
            file_path (str): The path to save the preprocessed data.
        """
        self.logger.info(f"Saving preprocessed data to {file_path}")
        data.to_csv(file_path, index=False)

    def load_preprocessed_data(self, file_path):
        """
        Loads the preprocessed data from a file.

        Args:
            file_path (str): The path to load the preprocessed data from.

        Returns:
            pandas.DataFrame: The loaded preprocessed data.
        """
        self.logger.info(f"Loading preprocessed data from {file_path}")
        data = pd.read_csv(file_path)
        return data

    def get_report(self, data, original_shape):
        """
        Generates a report summarizing the preprocessing steps applied to the dataset.

        Args:
            data (pandas.DataFrame): The preprocessed data.
            original_shape (tuple): The original shape of the dataset before preprocessing.

        Returns:
            str: The report summarizing the preprocessing steps.
        """
        report = f"Original dataset shape: {original_shape}\n"
        report += f"Preprocessed dataset shape: {data.shape}\n"
        report += f"Number of columns dropped: {original_shape[1] - data.shape[1]}"
        return report
