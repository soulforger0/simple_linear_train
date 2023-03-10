import unittest
from training_data_pipe import data_extract, data_validation, data_profile_drift, data_split, data_transform, model_train_tune, model_eval
from simple_linear_regr import SimpleLinearRegression
import numpy as np


class TestDataExtract(unittest.TestCase):
    
    def test_data_extract(self):
        diabetes_X, diabetes_y = data_extract()
        self.assertIsInstance(diabetes_X, np.ndarray)
        self.assertIsInstance(diabetes_y, np.ndarray)
        self.assertEqual(diabetes_X.shape[1], 1)
        self.assertEqual(diabetes_X.shape[0], diabetes_y.shape[0])
        
class TestDataValidation(unittest.TestCase):
    
    def test_data_validation(self):
        diabetes_X, diabetes_y = data_extract()
        with self.assertRaises(AssertionError):
            data_validation([1, 2, 3], diabetes_y)
        with self.assertRaises(AssertionError):
            data_validation(diabetes_X, [1, 2, 3])
        with self.assertRaises(AssertionError):
            data_validation(np.array([[1, 2], [3, 4]]), diabetes_y)
        with self.assertRaises(AssertionError):
            data_validation(diabetes_X, np.array([1, 2, 3]))
        self.assertIsNone(data_validation(diabetes_X, diabetes_y))
        
class TestDataProfileDrift(unittest.TestCase):
    
    def test_data_profile_drift(self):
        diabetes_X, diabetes_y = data_extract()
        with self.assertRaises(ValueError):
            data_profile_drift(diabetes_X + 1, diabetes_y)
        
class TestDataSplit(unittest.TestCase):
    
    def test_data_split(self):
        diabetes_X, diabetes_y = data_extract()
        X_train, y_train, X_test, y_test = data_split(diabetes_X, diabetes_y)
        self.assertEqual(len(X_train), diabetes_X.shape[0] - int(diabetes_X.shape[0]*0.05))
        self.assertEqual(len(y_train), diabetes_y.shape[0] - int(diabetes_y.shape[0]*0.05))
        self.assertEqual(len(X_test), int(diabetes_X.shape[0]*0.05))
        self.assertEqual(len(y_test), int(diabetes_y.shape[0]*0.05))
        
class TestDataTransform(unittest.TestCase):
    
    def test_data_transform(self):
        diabetes_X, diabetes_y = data_extract()
        transformed_X = data_transform(diabetes_X)
        self.assertEqual(transformed_X.shape, diabetes_X.shape)
        
class TestModelTrainTune(unittest.TestCase):
    
    def test_model_train_tune(self):
        diabetes_X, diabetes_y = data_extract()
        X_train, y_train, X_test, y_test = data_split(diabetes_X, diabetes_y)
        model = model_train_tune(X_train, y_train)
        self.assertIsInstance(model, SimpleLinearRegression)
        
class TestModelEval(unittest.TestCase):
    
    def test_model_eval(self):
        diabetes_X, diabetes_y = data_extract()
        X_train, y_train, X_test, y_test = data_split(diabetes_X, diabetes_y)
        model = model_train_tune(X_train, y_train)
        model_file_name = model_eval(model, X_test, y_test)
        self.assertIsNotNone(model_file_name)