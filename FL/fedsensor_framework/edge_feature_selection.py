import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel, RFECV, RFE
import utils
import pandas as pd



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedSensor Participant Feature Selection")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset. Available datasets: AQI, MOTOR, PERSON, UMAFALL.",
    )
    parser.add_argument(
        "--fs",
        type=str,
        required=True,
        help="Feature Selection algorithm. Available: ExtraTreesClassifier, RandomForestClassifier, ExtraTreesRegressor, RandomForestRegressor, logreg.",
    )
    parser.add_argument(
        "--estimators",
        type=int,
        required=True,
        help="Number of estimators for extra_tree and random_forest models.",
    )
    parser.add_argument(
        "--min_features",
        type=int,
        required=False,
        default=None,
        help="Number of features to select. Default = None (SelectFromModel).",
    )
    

    args = parser.parse_args()
    dataset = args.dataset
    fs = args.fs
    n_estimators = args.estimators
    min_features = args.min_features

    # dataset = 'AQI'
    # # fs = 'RandomForestRegressor'
    # fs = 'RandomForestClassifier'
    # n_estimators = 10
    # min_features = 3

    if (dataset != 'AQI') and (fs == 'ExtraTreesRegressor' or fs == 'RandomForestRegressor'):
        print("Error - regression only available to AQI dataset")
        exit(0)
    elif (dataset == 'AQI') and (fs == 'ExtraTreesRegressor' or fs == 'RandomForestRegressor'):
        result = 'regression'
    else:
        result = 'multiclass'


    if dataset == 'AQI':
        # Load dataset
        (X_train, y_train), (X_test, y_test), features_dict, resulting_dict = utils.load_AQI_dataset(sensors='aqi_all', result=result)
    elif dataset == 'PERSON':
        # Load dataset
        (X_train, y_train), (X_test, y_test), features_dict, resulting_dict = utils.load_PERSON_dataset(sensors='person_all', result=result)
    elif dataset == 'UMAFALL':
        # Load dataset
        (X_train, y_train), (X_test, y_test), features_dict, resulting_dict = utils.load_UMAFALL_dataset(sensors='umafall_all', result=result)
    elif dataset == 'MOTOR':
        # Load dataset
        (X_train, y_train), (X_test, y_test), features_dict, resulting_dict = utils.load_MOTOR_dataset(sensors='motor_all', result=result)
    else:
        print("Dataset not available.")
        exit(0)

    X_train = pd.DataFrame(X_train, columns=features_dict.values())
    X_test = pd.DataFrame(X_test, columns=features_dict.values())


    #Feature selection
    if fs == 'ExtraTreesClassifier':
        estimator = ExtraTreesClassifier(n_estimators=n_estimators)
    elif fs == 'RandomForestClassifier':
        estimator = RandomForestClassifier(n_estimators=n_estimators)
    elif fs == 'ExtraTreesRegressor':
        estimator = ExtraTreesRegressor(n_estimators=n_estimators)
    elif fs == 'RandomForestRegressor':
        estimator = RandomForestRegressor(n_estimators=n_estimators)
    elif fs == 'logreg':
        estimator = LogisticRegression(max_iter=500, n_jobs=-1)



    if min_features is None: #SelectFromModel
        estimator = estimator.fit(X_train, y_train)
        fs_model = SelectFromModel(estimator, prefit=True)
        X_train = X_train.loc[:, fs_model.get_support()]
        X_test = X_test.loc[:, fs_model.get_support()]
        message = 'Feature Selection result (SelectFromModel):'
    
        
    else: #RFE
        selector = RFE(estimator, n_features_to_select=min_features, step=1)
        selector = selector.fit(X_train, y_train)
        X_train = X_train.loc[:, selector.support_]
        X_test = X_test.loc[:, selector.support_]
        message = 'Feature Selection result (RFE):'
    
    new_features_dict = {k: v for k, v in features_dict.items() if v in list(X_train.columns)}
    print(message, new_features_dict)











