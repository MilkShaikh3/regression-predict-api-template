"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------

    dataset1 = pd.read_csv('/home/explore-student/regression-predict-api-template/utils/data/train_data.csv', index_col=0)
    dataset2 = pd.read_csv('/home/explore-student/regression-predict-api-template/utils/data/riders.csv', index_col=0)

    train = pd.merge(dataset1, dataset2, how='left', on='Rider Id') 

    train.columns = [col.replace(" - "," ") for col in train.columns]
    train.columns = [col.replace(" ","_") for col in train.columns]
    train.columns = [col.replace("_(Mo_=_1)","") for col in train.columns]
    train.columns = [col.replace("(KM)","KM") for col in train.columns]

    def convert_time(input_df, column_name):
        
        input_df[column_name] = pd.to_datetime(input_df[column_name]).dt.hour
        bins=[-1, 3, 7, 11, 15, 19, 23]
        labels=pd.Categorical(['Night', 'EarlyMorning', 'Morning', 'Afternoon', 'Evening', 'Night'])
        input_df[column_name] = pd.cut(x=input_df[column_name], bins=bins, labels=labels)
  
        return 

    convert_time(train, 'Pickup_Time')
    convert_time(train, 'Placement_Time')
    convert_time(train, 'Confirmation_Time')
    convert_time(train, 'Arrival_at_Pickup_Time')
    convert_time(train, 'Arrival_at_Destination_Time')

    columns = ['Personal_or_Business', 'Platform_Type', 'Arrival_at_Pickup_Time', 
           'Confirmation_Time', 'Placement_Time', 'Pickup_Time']
           
    train_dummies = pd.get_dummies(train, columns=columns, drop_first=True) 

    train_dummies.columns = [col.replace(" ", "_") for col in train_dummies.columns]

    column_titles = [col for col in train_dummies.columns if col!= 'Time_from_Pickup_to_Arrival'] + ['Time_from_Pickup_to_Arrival']
    train_dummies=train_dummies.reindex(columns=column_titles)

    y_data = train_dummies['Time_from_Pickup_to_Arrival']

    X_names = ['Distance_KM', 'No_Of_Orders', 'Average_Rating', 
           'Personal_or_Business_Personal','Pickup_Time_EarlyMorning', 
           'Pickup_Time_Evening', 'Pickup_Time_Morning','Pickup_Time_Night', 
           'Arrival_at_Pickup_Time_EarlyMorning',
           'Arrival_at_Pickup_Time_Evening', 'Arrival_at_Pickup_Time_Morning', 
           'Arrival_at_Pickup_Time_Night']               

    X_data = train_dummies[X_names]

    X_remove = ['Arrival_at_Pickup_Time_EarlyMorning', 'Arrival_at_Pickup_Time_Evening', 'Arrival_at_Pickup_Time_Morning',
                'Arrival_at_Pickup_Time_Night']
    X_corr_names = [col for col in X_names if col not in X_remove]

    y_train = train_dummies['Time_from_Pickup_to_Arrival']
    X_train = train_dummies[X_corr_names]

    dataset3 = pd.read_csv('/home/explore-student/regression-predict-api-template/utils/data/test_data.csv', index_col=0)

    test = pd.merge(dataset3, dataset2, how='left', on='Rider Id') 

    test.columns = [col.replace(" - "," ") for col in test.columns]
    test.columns = [col.replace(" ","_") for col in test.columns]
    test.columns = [col.replace("_(Mo_=_1)","") for col in test.columns]
    test.columns = [col.replace("(KM)","KM") for col in test.columns]\
    
    convert_time(test, 'Pickup_Time')

    columns = ['Personal_or_Business', 'Arrival_at_Pickup_Time', 'Pickup_Time']
    test_dummies = pd.get_dummies(test, columns=columns, drop_first=True) 

    test_dummies.columns = [col.replace(" ", "_") for col in test_dummies.columns]

    X_test = test_dummies[X_corr_names]
    
    predict_vector = X_test

    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
