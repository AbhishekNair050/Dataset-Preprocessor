from DatasetProcessor.DatasetProcessor import *
import unittest
import pandas as pd
import numpy as np


class TestDataPreprocessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.preprocessor = DatasetProcessor()

    def setUp(self):
        # Sample data for testing
        self.data = pd.DataFrame(
            {
                "A": [1, 2, np.nan, 4],
                "B": ["a", "b", "a", np.nan],
                "C": [10, 20, 30, 40],
                "D": [np.nan, 1.5, np.nan, 4.5],
                "E": [100, 200, 300, 400],
            }
        )

    def test_read_data(self):
        # Mocking read_data method
        test_file_path = "Iris.csv"
        self.data.to_csv(test_file_path, index=False)
        loaded_data = self.preprocessor.read_data(test_file_path)
        pd.testing.assert_frame_equal(loaded_data, self.data)

    def test_handle_missing_values_fill(self):
        filled_data = self.preprocessor.handle_missing_values(
            self.data.copy(), strategy="fill"
        )
        self.assertFalse(filled_data.isnull().values.any())

    def test_handle_missing_values_drop(self):
        dropped_data = self.preprocessor.handle_missing_values(
            self.data.copy(), strategy="drop"
        )
        self.assertEqual(len(dropped_data), 1)

    def test_encode_categorical_features(self):
        encoded_data = self.preprocessor.encode_categorical_features(self.data.copy())
        self.assertTrue(np.issubdtype(encoded_data["B"].dtype, np.number))

    def test_normalize_numerical_features_standard(self):
        normalized_data = self.preprocessor.normalize_numerical_features(
            self.data.copy(), norm_type="standard"
        )
        self.assertTrue(np.allclose(normalized_data.mean(), 0, atol=1e-1))
        self.assertTrue(np.allclose(normalized_data.std(), 1, atol=1e-1))

    def test_normalize_numerical_features_minmax(self):
        normalized_data = self.preprocessor.normalize_numerical_features(
            self.data.copy(), norm_type="minmax"
        )
        self.assertTrue(
            (normalized_data >= 0).all().all() and (normalized_data <= 1).all().all()
        )

    def test_select_features(self):
        # Adding a target column for feature selection
        self.data["Target"] = [1, 0, 1, 0]
        selected_data = self.preprocessor.select_features(
            self.data.copy(), target_column="Target", n_features_to_select=2
        )
        self.assertEqual(
            len(selected_data.columns), 3
        )  # 2 selected features + target column

    def test_drop_useless_columns(self):
        # Adding some "useless" columns for testing
        self.data["id"] = [1, 2, 3, 4]
        self.data["Unnamed: 0"] = [1, 2, 3, 4]
        cleaned_data = self.preprocessor.drop_useless_columns(self.data.copy())
        self.assertNotIn("id", cleaned_data.columns)
        self.assertNotIn("Unnamed: 0", cleaned_data.columns)

    def test_preprocess_data(self):
        # Test the full preprocessing pipeline
        test_file_path = "test_data.csv"
        self.data["Target"] = [1, 0, 1, 0]
        self.data.to_csv(test_file_path, index=False)
        processed_data = self.preprocessor.preprocess_data(
            test_file_path,
            target_column="Target",
            handle_missing="fill",
            encode_categorical=True,
            normalize_numerical="standard",
            feature_selection_method="correlation",
            n_features_to_select=2,
            variance_ratio_threshold=0.1,
        )
        self.assertEqual(
            len(processed_data.columns), 3
        )  # 2 selected features + target column

    def test_save_and_load_preprocessed_data(self):
        # Save preprocessed data and load it back
        test_file_path = "preprocessed_data.csv"
        self.preprocessor.save_preprocessed_data(self.data, test_file_path)
        loaded_data = self.preprocessor.load_preprocessed_data(test_file_path)
        pd.testing.assert_frame_equal(loaded_data, self.data)

    def test_get_report(self):
        original_shape = self.data.shape
        preprocessed_data = self.data.dropna()
        report = self.preprocessor.get_report(preprocessed_data, original_shape)
        self.assertIn(f"Original dataset shape: {original_shape}", report)
        self.assertIn(f"Preprocessed dataset shape: {preprocessed_data.shape}", report)
        self.assertIn(
            f"Number of columns dropped: {original_shape[1] - preprocessed_data.shape[1]}",
            report,
        )


if __name__ == "__main__":
    unittest.main()
