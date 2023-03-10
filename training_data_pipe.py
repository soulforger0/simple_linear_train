
import shutil
import os
import numpy as np
from simple_linear_regr_utils import generate_data, evaluate
from sklearn.datasets import load_diabetes
import time
from datetime import datetime
from scipy.stats import ks_2samp
from simple_linear_regr import SimpleLinearRegression


def data_extract():
    """
    put data extact fucntion here
    
    return (x, y)
    """
    diabetes_X, diabetes_y = load_diabetes(return_X_y=True)
    
    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]
    
    return diabetes_X, diabetes_y


def data_validation(diabetes_X, diabetes_y):
    """
    Assert data shape should be 1
    Assert data type should be np array
    """
    assert isinstance(diabetes_X[0], np.ndarray), "x data type is not np array"
    assert isinstance(diabetes_y, np.ndarray), "y data type is not np array"
    assert diabetes_X.shape[1] == 1, "training data x shape is incorrect"
    assert diabetes_y.shape[0] == diabetes_X.shape[0], "number of record of y does not match x"

    

def data_profile_drift(diabetes_X, diabetes_y):
    """
    Check for same number of feature
    Check for feature drift
    Check for distribution drift
    """
    
    # Load reference data
    # i'm using the same data as reference data
    reference_X, reference_y = data_extract()
    
    # Ensure that the two datasets have the same columns
    # given we are using np array, we skip this
    #if set(diabetes_X.columns) != set(reference_X.columns):
    #    raise ValueError("The two datasets do not have the same columns")
    
    # Loop over columns and use ks_2samp to compare the distributions
    for i in range(diabetes_X.shape[1]):
        stat, p_value = ks_2samp(diabetes_X[:, i], reference_X[:, i])
        print(f"KS statistic for column {i}: {stat:.4f}")
        print(f"P-value for column {i}: {p_value:.4f}")
        
        # Find KS critical value
        alpha = 0.05
        ks_crit = np.sqrt(-0.5 * np.log(alpha / 2) / np.sqrt(diabetes_X.shape[0]))
        
        # raise error if 2 distrubtion is too far apart
        if p_value < alpha and stat > ks_crit:
            raise ValueError("The two datasets distribution is too far")

    # save KS state and p_value for each col to meta data store
    # i'm skipping here 


def data_split(diabetes_X, diabetes_y):
    """
    Do 95 / 5 split
    No shuffle
    Return train X, train y, test X, test y
    """
    
    # get number of test record we need
    test_vol = int(diabetes_X.shape[0]*0.05)
    
    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-test_vol]
    diabetes_X_test = diabetes_X[-test_vol:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-test_vol].reshape(-1,1)
    diabetes_y_test = diabetes_y[-test_vol:].reshape(-1,1)

    print(f"# Training Samples: {len(diabetes_X_train)}; # Test samples: {len(diabetes_X_test)};")
    return diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test
    


def data_transform(diabetes_X):
    """
    Do tranformation
    Return transformed X
    """
    
    # no transform done, 
    return diabetes_X



def model_train_tune(diabetes_X_train, diabetes_y_train):
    """
    Parameters
    ----------
    diabetes_X_train : TYPE np array of float shape :,1
    diabetes_y_train : TYPE np array of float shape :,

    Returns
    -------
    fited SimpleLinearRegression Class model

    """
    model = SimpleLinearRegression()
    model.fit(diabetes_X_train, diabetes_y_train)

    return model

def model_eval(model, diabetes_X_test, diabetes_y_test):
    """
    Parameters
    ----------
    model : TYPE SimpleLinearRegression class model

    Returns
    -------
    None.

    """
    
    # Get MSE and r2 from prediction
    predicted = model.predict(diabetes_X_test)
    mse, r2 = evaluate(model, diabetes_X_test, diabetes_y_test, predicted)
    
    # read best model mse and r2 from meta data store
    # i will just hard code them here
    best_mse = 2450
    best_r2 = 0.42
    best_model_filename = "BASE_MODEL_FILE.pickle"
    
    # Compare current model and best model result
    # if current model is better, save the model, return model file name
    # else output pass best model file name
    
    if mse < best_mse and r2 > best_r2:
        # current model is the best model
        model_file_name = model.save_model()
        print(f"current model is the best one. with mse:{mse}, r2:{r2}, file name: {model_file_name}")
        print("")
        
        # delete the prod model folder
        # and recreate folder for copy
        folder_path = './prod_model'
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        
        # we also save a copy to prod model folder to promote into serving pipeline
        file_path = f'./model/{model_file_name}.pickle'
        dest_path = './prod_model'
        shutil.copy(file_path, dest_path)
        
        # also save mse, r2 into meta data store
        # i'm skipping here
        return model_file_name
    
    else:
        print(f"Current model was not better than the past with current model mse:{mse}  r2:{r2}")
        return best_model_filename
    



if __name__ == "__main__":
    diabetes_X, diabetes_y = data_extract()
    data_validation(diabetes_X, diabetes_y)
    data_profile_drift(diabetes_X, diabetes_y)
    diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test = data_split(diabetes_X, diabetes_y)
    model = model_train_tune(diabetes_X_train, diabetes_y_train)
    model_eval(model, diabetes_X_test, diabetes_y_test)
    
    
    