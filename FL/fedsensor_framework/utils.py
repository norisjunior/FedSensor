from typing import Tuple, Union, List
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
KMeansParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

algorithms_available = ['logreg', 'kmeans', 'linreg']


def get_model_kmeans_parameters(model: KMeans) -> KMeansParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    params = (model.cluster_centers_,)
    return params


def set_model_kmeans_params(
    model: KMeans, params: KMeansParams
) -> KMeans:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.cluster_centers_ = params[0]
    return model


def set_initial_kmeans_params(model: KMeans, dataset):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """

    if dataset == 'AQI':
        _, (X_test, y_test), _, _ = load_AQI_dataset()
    elif dataset == 'FALLUP':
        _, (X_test, y_test), _, _ = load_FALLUP_dataset()
    elif dataset == 'PERSON':
        _, (X_test, y_test), _, _ = load_PERSON_dataset()
    elif dataset == 'UMAFALL':
        _, (X_test, y_test), _, _ = load_UMAFALL_dataset()
    elif dataset == 'MOTOR':
        _, (X_test, y_test), _, _ = load_MOTOR_dataset()

    #Why to fit on a small dataset? Answer: https://stackoverflow.com/questions/62186418/kmeans-object-has-no-attribute-n-threads
    model.fit(X_test[:50])









def anomaly_detection(client_dataset, anomaly_detection='No'):



    if anomaly_detection == 'No':
        print("Inserting +30% of anomalies in the data...")
        df_anomalies = client_dataset.sample(n=int(client_dataset.shape[0]*0.9)).copy()

        temp_anomalias = df_anomalies.iloc[:,:-3].sample(frac=1).reset_index()
        shuffle_anomalias = df_anomalies.iloc[:,-3:].sample(frac=1).reset_index()

        df_anomalies = pd.concat([temp_anomalias, shuffle_anomalias], axis=1)
        df_anomalies.drop(columns=['index'], inplace=True)

        if round(random.uniform(0.1, 1.0), 2) > 0.5:
            anomaly_randomness = 1+round(random.uniform(0.3, 0.7), 2)
        else:
            anomaly_randomness = -1-round(random.uniform(0.3, 0.7), 2)

        print(anomaly_randomness)

        anomalies = df_anomalies.iloc[:, :-3]*1.8
        df_anomalies.update(anomalies)
        client_dataset = pd.concat([client_dataset,df_anomalies])
        print("No anomaly detection. ")

    else:

        if anomaly_detection == 'ECOD':
            # train an ECOD detector
            from pyod.models.ecod import ECOD
            clf = ECOD()
        elif anomaly_detection == 'IForest':
            from pyod.models.iforest import IForest
            clf = IForest()
        elif anomaly_detection == 'LOF':
            from pyod.models.lof import LOF
            clf = LOF()

        client_dataset['Outlier'] = None


        is_AQI_regression=False
        find_string = ['AQI']
        if (any(element in list(client_dataset.columns) for element in find_string)):
            if (all(client_dataset['AQI'] == client_dataset['encoded_target'])):
                #valor contínuo
                #regressão linear
                is_AQI_regression=True
                df_sample = client_dataset.copy()
                clf.fit(df_sample.sample(n=int(df_sample.shape[0]*0.8)).iloc[:, :-4].values)
                outliers = clf.predict(df_sample.iloc[:, :-4].values)
                df_sample.loc[:,['Outlier']] = list(outliers)
                client_dataset.update(df_sample)
                print("Linear Regression experiment.")
                del df_sample


        if (not is_AQI_regression):
            #classe ou grupo
            #regressão logística ou kmeans
            for index in client_dataset['encoded_target'].unique():
                # print(index, client_dataset[client_dataset['encoded_target'] == index]['text_target'].unique())
                df_sample = client_dataset[client_dataset['encoded_target'] == index].copy()
                clf.fit(df_sample.sample(n=int(df_sample.shape[0]*0.8)).iloc[:, :-4].values)
                outliers = clf.predict(df_sample.iloc[:, :-4].values)
                df_sample.loc[:,['Outlier']] = list(outliers)
                client_dataset.update(df_sample)
                print(f"Class[{index}] is {client_dataset[client_dataset['encoded_target'] == index]['text_target'].unique()} with len {len(df_sample)} and with {len(df_sample[(df_sample['Outlier'] == 1)])} outliers")
                del df_sample

        try:
            client_dataset['Outlier'] = client_dataset['Outlier'].astype('int')
        except ValueError:
            # Handle the exception
            print('Anomaly detection error')
            exit(0)

        print(f"{anomaly_detection} removed {len(client_dataset[(client_dataset['Outlier']==1)])} outliers")
        client_dataset.drop(client_dataset[client_dataset['Outlier'] == 1].index, inplace=True)
        drop_list = ['Outlier']

        client_dataset.drop(drop_list, axis=1, inplace=True)
    #endif - else

    return client_dataset










def encode_dataset(client_dataset, full_dataset, sensors=None, result=None):


    client_dataset["encoded_target"] = client_dataset["encoded_target"].astype(int)
    full_dataset["encoded_target"] = full_dataset["encoded_target"].astype(int)

    drop_list = list(client_dataset.columns)[-3:]

    X = client_dataset.drop(drop_list, axis=1)
    X_full = full_dataset.drop(drop_list, axis=1)


    # Adjusting sensorss
    if sensors == 'aqi_pm_only':
        key_list = [33250, 33251]
    elif sensors == 'aqi_pm_plus':
        key_list = [33250, 33251, 33256, 33258]
    elif sensors == 'aqi_all':
        key_list = [33250, 33251, 33252, 33253, 33254, 33255, 33256, 33257, 33258]
    elif sensors == 'motor_all':
        key_list = [33460, 33130, 33131, 33132]
    elif sensors == 'motor_acc':
        key_list = [33130, 33131, 33132]
    else:
        print("Wrong sensors name for drop_list\n")
        exit(0)
    #endif sensors ==



    features_dict = {key_list[i]: X.columns[i] for i in range(len(key_list))}



    # result = 'multiclass'
    if ( result == 'binary' or result == 'multiclass' ) :
        y = client_dataset['encoded_target']
        y_full = full_dataset['encoded_target']
        resulting_y = {client_dataset['encoded_target'].unique()[i]: client_dataset['text_target'].unique()[i] for i in range(len(client_dataset['encoded_target'].unique()))}
    elif (result == 'regression' and sensors != 'motor_all'):
        y = client_dataset['AQI']
        y_full = full_dataset['AQI']
        resulting_y = client_dataset['AQI']
    else:
        print('Error - only binary and multiclass model result available')
        exit(0)
    #endif result ==


    # Split train and test
    #Get train for the partial dataset, but the test from de full dataset
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
    _, X_test, _, y_test = train_test_split(X_full, y_full, test_size=0.2)


    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return (X_train, y_train), (X_test, y_test), features_dict, resulting_y



















def load_FALLUP_dataset(algorithm='logreg', num_classes=11, sensors='fallup_all', result='multiclass') -> Dataset:
    # Load data
    X=pd.read_csv('datasets/FALL-UP-CompleteDataSet.csv', low_memory=False)
    X = X.dropna()
    X = X.drop(X.index[0])

    # algorithm='logreg'
    # num_classes=11
    # sensors='fallup_all'
    # result='multiclass'

    # result = 'regression'
    if result == 'binary':
        print('Error - only multiclass and regression model results available')
        exit(0)
    elif result in ['multiclass', 'regression']:
        y = X.iloc[:, 44]
    else:
        print('Error - only multiclass and regression model results available')
        exit(0)


    # Adjusting sensorss
    # sensors = 'all'
    if sensors == 'fallup_all':
        drop_list = [0, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
    elif sensors == 'fallup_acc_veloc':
        drop_list = [0, 7, 14, 21, 28, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
    elif sensors == 'fallup_acc_only':
        drop_list = [0, 4, 5, 6, 7, 11, 12, 13, 14, 18, 19, 20, 21, 25, 26, 27, 28, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
    elif sensors == 'fallup_velocity_only':
        drop_list = [0, 1, 2, 3, 7,  8,  9, 10, 14, 15, 16, 17, 21, 22, 23, 24, 28, 29, 30, 31, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
    elif sensors == 'fallup_others_only':
        drop_list = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]

    else:
        print("Wrong sensors name for drop_list\n")
        exit(0)

    X = X.drop(X.iloc[:, drop_list], axis=1)


    le = LabelEncoder()
    y = le.fit_transform(y)

    # np.unique(y)
    le.inverse_transform(np.unique(y))
    resulting_y = {np.unique(y)[i]: le.inverse_transform(np.unique(y))[i] for i in range(len(le.inverse_transform(np.unique(y))))}


    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    X_train = X_train.to_numpy(dtype = 'float32')
    X_test = X_test.to_numpy(dtype = 'float32')

    # y_train = y_train.to_numpy()
    # y_test = y_test.to_numpy()

    return (X_train, y_train), (X_test, y_test), list(X.columns), resulting_y









def load_AQI_dataset(algorithm='logreg', num_classes=3, sensors='aqi_pm_plus', result='multiclass', anomaly_detection='No') -> Dataset:
    # Load data
    dataset=pd.read_csv('datasets/AQI-INDIA-city_hour.csv')
    dataset = dataset.dropna()

    # algorithm='logreg'
    # num_classes=2
    # sensors='aqi_pm_plus'
    # result='multiclass'
    # anomaly_detection='IForest'

    if algorithm == 'logreg':
        if num_classes == 3:
            dataset.loc[dataset['AQI_Bucket'] == 'Satisfactory',['AQI_Bucket']] = 'Good'
            dataset.loc[dataset['AQI_Bucket'] == 'Poor',['AQI_Bucket']] = 'Very Poor'
            dataset.loc[dataset['AQI_Bucket'] == 'Very Poor',['AQI_Bucket']] = 'Severe'

            dataset.loc[dataset['AQI_Bucket'] == 'Good',['AQI_Bucket']] = 'Good'
            dataset.loc[dataset['AQI_Bucket'] == 'Moderate',['AQI_Bucket']] = 'Acceptable'
            dataset.loc[dataset['AQI_Bucket'] == 'Severe',['AQI_Bucket']] = 'Not acceptable'

        elif num_classes in [2, 6]:
            pass
        else:
            print("Wrong number of classes, for this experiment must be 3 or 6")
            exit(0)
    #endif algorithm == 'logreg'

    # Adjusting sensorss
    # sensors = 'aqi_all'
    if sensors == 'aqi_pm_only':
        #drop_list = ['AQI', 'City', 'Datetime', 'AQI_Bucket', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
        key_list = [33250, 33251]
        anomaly_drop_list = ['City', 'Datetime', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
    elif sensors == 'aqi_pm_plus':
        #drop_list = ['AQI', 'City', 'Datetime', 'AQI_Bucket', 'NO', 'NO2', 'NOx', 'NH3', 'SO2', 'Benzene', 'Toluene', 'Xylene']
        key_list = [33250, 33251, 33256, 33258]
        anomaly_drop_list = ['City', 'Datetime', 'NO', 'NO2', 'NOx', 'NH3', 'SO2', 'Benzene', 'Toluene', 'Xylene']
    elif sensors == 'aqi_all':
        #drop_list = ['AQI', 'City', 'Datetime', 'AQI_Bucket', 'Toluene', 'Xylene']
        #drop_list = ['AQI', 'City', 'Datetime', 'AQI_Bucket', 'Benzene', 'Toluene', 'Xylene']
        key_list = [33250, 33251, 33252, 33253, 33254, 33255, 33256, 33257, 33258]
        anomaly_drop_list = ['City', 'Datetime', 'Benzene', 'Toluene', 'Xylene']
    else:
        print("Wrong sensors name for drop_list\n")
        exit(0)
    #endif sensors ==


    dataset = dataset.drop(anomaly_drop_list, axis=1)
    drop_list = []


    if anomaly_detection != 'No':
        #Removing outliers by each class:
        if anomaly_detection == 'ECOD':
            # train an ECOD detector
            from pyod.models.ecod import ECOD
            clf = ECOD()
        elif anomaly_detection == 'IForest':
            from pyod.models.iforest import IForest
            clf = IForest()
        elif anomaly_detection == 'LOF':
            from pyod.models.lof import LOF
            clf = LOF()
        #endif anomaly_detection

        # classes = list(dataset['AQI_Bucket'].unique())
        dataset['Outlier'] = None

        for index in dataset['AQI_Bucket'].unique():
            # sample = dataset.sample(n=int(dataset.shape[0]*0.8)).iloc[:, 2:11]
            df_sample = dataset[dataset['AQI_Bucket'] == index].copy()
            #clf.fit(df_sample.sample(n=int(df_sample.shape[0]*0.8)).iloc[:, 2:11].values)
            clf.fit(df_sample.sample(n=int(df_sample.shape[0]*0.8)).iloc[:, :-3].values)
            outliers = clf.predict(df_sample.iloc[:, :-3].values)
            df_sample.loc[:,['Outlier']] = list(outliers)
            dataset.update(df_sample)
            print(f"Class[{index}] is {dataset[dataset['AQI_Bucket'] == index]['AQI_Bucket'].unique()} with len {len(df_sample)} and with {len(df_sample[(df_sample['Outlier'] == 1)])} outliers")
            del df_sample

        try:
            dataset['Outlier'] = dataset['Outlier'].astype('int')
        except ValueError:
            # Handle the exception
            print('Anomaly detection error')
            exit(0)

        print(f"{anomaly_detection} removed {len(dataset[(dataset['Outlier']==1)])} outliers")
        dataset.drop(dataset[dataset['Outlier'] == 1].index, inplace=True)
        drop_list = ['Outlier']
    #endif anomaly_detection != 'No'

    drop_list.extend(['AQI_Bucket', 'AQI'])

    X = dataset.drop(drop_list, axis=1)
    features_dict = {key_list[i]: X.columns[i] for i in range(len(key_list))}

    # result = 'multiclass'
    if result == 'binary':
        le = LabelEncoder()
        dataset['AQI_is_severe'] = np.where( ( (dataset['AQI_Bucket'] != 'Good') & (dataset['AQI_Bucket'] != 'Satisfactory') ), 'Severe', 'Not severe')
        dataset['AQI_is_severe'] = le.fit_transform(dataset['AQI_is_severe'])
        resulting_y = {dataset['AQI_is_severe'].unique()[i]: le.inverse_transform(dataset['AQI_is_severe'].unique())[i] for i in range(len(le.inverse_transform(dataset['AQI_is_severe'].unique())))}
        drop_list.append('AQI_is_severe')
        y = dataset['AQI_is_severe']
    elif result == 'multiclass':
        le = LabelEncoder()
        dataset['AQI_Bucket'] = le.fit_transform(dataset['AQI_Bucket'])
        resulting_y = {dataset['AQI_Bucket'].unique()[i]: le.inverse_transform(dataset['AQI_Bucket'].unique())[i] for i in range(len(le.inverse_transform(dataset['AQI_Bucket'].unique())))}
        y = dataset['AQI_Bucket']
    elif result == 'regression':
        y = dataset['AQI']
        resulting_y = dataset['AQI']
    else:
        print('Error - only binary and multiclass model result available')
        exit(0)
    #endif result ==


    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return (X_train, y_train), (X_test, y_test), features_dict, resulting_y






















def load_PERSON_dataset(algorithm='logreg', sensors='person_all', result='multiclass') -> Dataset:
    # data: https://archive.ics.uci.edu/ml/datasets/Localization+Data+for+Person+Activity
    # Load data
    dataset = pd.read_csv('datasets/PersonActivity.csv')
    dataset = dataset.dropna()


    # algorithm='logreg'
    # num_classes=11
    # sensors='person_all'
    # result='multiclass'


    # Adjusting sensorss
    # sensors = 'person_all'
    if sensors == 'person_all':
        drop_list = ['SeqName', 'Tag', 'Timestamp', 'Datetime', 'Activity']
        key_list = [33130, 33131, 33132]
    else:
        print("Wrong sensors name for drop_list\n")
        exit(0)




    # result = 'multiclass'
    if result == 'binary':
        le = LabelEncoder()
        dataset['Binary_activity'] = np.where(dataset['Activity'] == 'falling', 'Fall', 'No Fall')
        dataset['binary_target'] = le.fit_transform(dataset['Binary_activity'])
        drop_list.append('Binary_activity')
        drop_list.append('binary_target')
        y = dataset['binary_target']
        resulting_y = {dataset['binary_target'].unique()[i]: le.inverse_transform(dataset['binary_target'].unique())[i] for i in range(len(le.inverse_transform(dataset['binary_target'].unique())))}
    elif result == 'multiclass':
        le = LabelEncoder()
        dataset['target'] = le.fit_transform(dataset['Activity'])
        drop_list.append('target')
        y = dataset['target']
        resulting_y = {dataset['target'].unique()[i]: le.inverse_transform(dataset['target'].unique())[i] for i in range(len(le.inverse_transform(dataset['target'].unique())))}
    else:
        print('Error - only binary and multiclass model result available')
        exit(0)


    X = dataset.drop(drop_list, axis=1)
    # X.columns

    features_dict = {key_list[i]: X.columns[i] for i in range(len(key_list))}

    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return (X_train, y_train), (X_test, y_test), features_dict, resulting_y




def load_UMAFALL_dataset(algorithm='logreg', sensors='umafall_all', result='multiclass') -> Dataset:

    # algorithm='logreg'
    # num_classes=4
    # sensors='type0'
    # result='multiclass'

    dataset = pd.read_csv('datasets/UMAFall_complete_dataset_reduced_all_falls_separated.csv', low_memory=False)

    dataset['X-Axis'] = pd.to_numeric(dataset['X-Axis'], errors='coerce')
    dataset['Y-Axis'] = pd.to_numeric(dataset['Y-Axis'], errors='coerce')
    dataset['Z-Axis'] = pd.to_numeric(dataset['Z-Axis'], errors='coerce')

    dataset.dropna(inplace=True)


    if sensors == 'umafall_all':
        pass
        # drop_list = ['ID1', 'ID2', 'TimeStamp', 'Sample No', 'Sensor Type', 'Sensor ID', 'Activity']
    elif sensors == 'type0':
        dataset = dataset.loc[dataset['Sensor Type'] == 0]
    elif sensors == 'type1':
        dataset = dataset.loc[dataset['Sensor Type'] == 1]
    elif sensors == 'type2':
        dataset = dataset.loc[dataset['Sensor Type'] == 2]
    else:
        print("Wrong sensors name, available: 'umafall_all', 'type0', 'type1', 'type2'\n")
        exit(0)

    #Removing outliers:
    # train an ECOD detector
    from pyod.models.ecod import ECOD
    clf = ECOD(contamination=0.3)
    clf.fit(dataset.sample(n=int(dataset.shape[0]*0.8))[['X-Axis', 'Y-Axis', 'Z-Axis']])
    dataset['Outlier'] = clf.predict(dataset[['X-Axis', 'Y-Axis', 'Z-Axis']])
    dataset.drop(dataset[dataset['Outlier'] == 1].index, inplace=True)

    drop_list = ['ID1', 'ID2', 'TimeStamp', 'Sample No', 'Sensor Type', 'Sensor ID', 'Activity', 'Outlier']
    key_list = [33130, 33131, 33132]


    # result = 'multiclass'
    if result == 'binary':
        le = LabelEncoder()
        dataset['Binary_activity'] = np.where( ( (dataset['Activity'] == 'backwardFall') | \
                                                 (dataset['Activity'] == 'lateralFall') | \
                                                  (dataset['Activity'] == 'forwardFall') ),
                                              'Fall', 'No Fall')
        dataset['binary_target'] = le.fit_transform(dataset['Binary_activity'])
        drop_list.append('Binary_activity')
        drop_list.append('binary_target')
        y = dataset['binary_target']
        resulting_y = {dataset['binary_target'].unique()[i]: le.inverse_transform(dataset['binary_target'].unique())[i] for i in range(len(le.inverse_transform(dataset['binary_target'].unique())))}
    elif result == 'multiclass':
        le = LabelEncoder()
        dataset['Activity'].mask(((dataset['Activity'] != 'backwardFall') & \
                     (dataset['Activity'] != 'lateralFall') & \
                     (dataset['Activity'] != 'forwardFall')), 'No Fall', inplace=True)
        dataset['target'] = le.fit_transform(dataset['Activity'])
        drop_list.append('target')
        y = dataset['target']
        resulting_y = {dataset['target'].unique()[i]: le.inverse_transform(dataset['target'].unique())[i] for i in range(len(le.inverse_transform(dataset['target'].unique())))}
    else:
        print('Error - only binary and multiclass model result available')
        exit(0)


    X = dataset.drop(drop_list, axis=1)

    features_dict = {key_list[i]: X.columns[i] for i in range(len(key_list))}

    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return (X_train, y_train), (X_test, y_test), features_dict, resulting_y





def load_MOTOR_dataset(algorithm='logreg', sensors='motor_all', result='multiclass', anomaly_detection='No') -> Dataset:
    # algorithm='logreg'
    # num_classes=2
    # sensors='motor_acc'
    # result='binary'
    # anomaly_detection = 'No'

    dataset = pd.read_csv('datasets/Motor_accelerometer.csv')
    dataset = dataset.dropna()


    if result == 'regression':
        print('Error - only binary and multiclass model result available')
        exit(0)

    drop_list = []


    # Adjusting sensorss
    # sensors = 'person_all'
    if sensors == 'motor_all':
        # drop_list = ['wconfid', 'pctid']
        # x: 33130, y: 33131, z: 33132, pctid: 33460
        key_list = [33460, 33130, 33131, 33132]
    elif sensors == 'motor_acc':
        # drop_list = ['wconfid', 'pctid']
        # x: 33130, y: 33131, z: 33132
        key_list = [33130, 33131, 33132]
        drop_list.extend(['pctid'])
    else:
        print("Wrong sensors name for drop_list\n")
        exit(0)

    cols = ['pctid', 'x', 'y', 'z', 'wconfid']
    dataset = dataset[cols]




    if anomaly_detection != 'No':
        #Removing outliers by each class:
        if anomaly_detection == 'ECOD':
            # train an ECOD detector
            from pyod.models.ecod import ECOD
            clf = ECOD()
        elif anomaly_detection == 'IForest':
            from pyod.models.iforest import IForest
            clf = IForest()
        elif anomaly_detection == 'LOF':
            from pyod.models.lof import LOF
            clf = LOF()
        #endif anomaly_detection

        classes = list(dataset['wconfid'].unique())
        dataset['Outlier'] = None

        for index in range(len(dataset['wconfid'].unique())):
            df_sample = dataset[dataset['wconfid'] == classes[index]].copy()
            clf.fit(df_sample.sample(n=int(df_sample.shape[0]*0.8)).loc[:, ['x', 'y', 'z', 'pctid']])
            outliers = clf.predict(df_sample.loc[:, ['x', 'y', 'z', 'pctid']].values)
            df_sample.loc[:,['Outlier']] = list(outliers)
            dataset.update(df_sample)
            print(f"Class[{index}] is {classes[index]} with len {len(df_sample)} and with {len(df_sample[(df_sample['Outlier'] == 1)])} outliers")
            del df_sample

        try:
            dataset['Outlier'] = dataset['Outlier'].astype('int')
        except ValueError:
            # Handle the exception
            print('Anomaly detection error')
            exit(0)
        print(f"{anomaly_detection} removed {len(dataset[(dataset['Outlier']==1)])} outliers")
        dataset.drop(dataset[dataset['Outlier'] == 1].index, inplace=True)
        drop_list = ['Outlier']
    #endif anomaly_detection != 'No'

    drop_list.extend(['wconfid'])






    # result = 'multiclass'
    if result == 'binary':
        le = LabelEncoder()
        dataset['Binary_activity'] = np.where(dataset['wconfid'] == 1, 'Normal', 'Fail')
        dataset['binary_target'] = le.fit_transform(dataset['Binary_activity'])
        drop_list.append('Binary_activity')
        drop_list.append('binary_target')
        y = dataset['binary_target']
        resulting_y = {dataset['binary_target'].unique()[i]: le.inverse_transform(dataset['binary_target'].unique())[i] for i in range(len(le.inverse_transform(dataset['binary_target'].unique())))}
    elif result == 'multiclass':
        le = LabelEncoder()
        dataset['Activity'] = np.NaN
        dataset['Activity'].mask(dataset['wconfid'] == 1, 'Normal', inplace=True)
        dataset['Activity'].mask(dataset['wconfid'] == 2, 'Perpendicular', inplace=True)
        dataset['Activity'].mask(dataset['wconfid'] == 3, 'Opposite', inplace=True)
        dataset['target'] = le.fit_transform(dataset['Activity'])
        drop_list.append('Activity')
        drop_list.append('target')
        y = dataset['target']
        resulting_y = {dataset['target'].unique()[i]: le.inverse_transform(dataset['target'].unique())[i] for i in range(len(le.inverse_transform(dataset['target'].unique())))}
    else:
        print('Error - only binary and multiclass model result available')
        exit(0)

    X = dataset.drop(drop_list, axis=1)


    features_dict = {key_list[i]: X.columns[i] for i in range(len(key_list))}


    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return (X_train, y_train), (X_test, y_test), features_dict, resulting_y































































# np.array_split(dataset, 3)









def load_AQI_dataset2(algorithm='logreg', num_classes=3, sensors='aqi_pm_plus', result='multiclass', partition_fraction=1) -> Dataset:
    # Load data
    dataset=pd.read_csv('datasets/AQI-INDIA-city_hour.csv')
    dataset = dataset.dropna()

    # algorithm='logreg'
    # num_classes=2
    # sensors='aqi_all'
    # result='multiclass'
    # partition_fraction = 0.11

    if algorithm == 'logreg':
        if num_classes == 3:
            dataset.loc[dataset['AQI_Bucket'] == 'Satisfactory',['AQI_Bucket']] = 'Good'
            dataset.loc[dataset['AQI_Bucket'] == 'Poor',['AQI_Bucket']] = 'Very Poor'
            dataset.loc[dataset['AQI_Bucket'] == 'Very Poor',['AQI_Bucket']] = 'Severe'

            dataset.loc[dataset['AQI_Bucket'] == 'Good',['AQI_Bucket']] = 'Good'
            dataset.loc[dataset['AQI_Bucket'] == 'Moderate',['AQI_Bucket']] = 'Acceptable'
            dataset.loc[dataset['AQI_Bucket'] == 'Severe',['AQI_Bucket']] = 'Not acceptable'

        elif num_classes in [2, 6]:
            pass
        else:
            print("Wrong number of classes, for this experiment must be 3 or 6")
            exit(0)
    #endif algorithm == 'logreg'


    drop_list = []


    # Adjusting sensorss
    # sensors = 'aqi_all'
    if sensors == 'aqi_pm_only':
        #drop_list = ['AQI', 'City', 'Datetime', 'AQI_Bucket', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
        # key_list = [33250, 33251]
        drop_list = ['City', 'Datetime', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
    elif sensors == 'aqi_pm_plus':
        #drop_list = ['AQI', 'City', 'Datetime', 'AQI_Bucket', 'NO', 'NO2', 'NOx', 'NH3', 'SO2', 'Benzene', 'Toluene', 'Xylene']
        # key_list = [33250, 33251, 33256, 33258]
        drop_list = ['City', 'Datetime', 'NO', 'NO2', 'NOx', 'NH3', 'SO2', 'Benzene', 'Toluene', 'Xylene']
    elif sensors == 'aqi_all':
        #drop_list = ['AQI', 'City', 'Datetime', 'AQI_Bucket', 'Toluene', 'Xylene']
        #drop_list = ['AQI', 'City', 'Datetime', 'AQI_Bucket', 'Benzene', 'Toluene', 'Xylene']
        # key_list = [33250, 33251, 33252, 33253, 33254, 33255, 33256, 33257, 33258]
        drop_list = ['City', 'Datetime', 'Benzene', 'Toluene', 'Xylene']
    else:
        print("Wrong sensors name for drop_list\n")
        exit(0)
    #endif sensors ==



    # result = 'multiclass'
    if result == 'binary':
        le = LabelEncoder()
        dataset['AQI_is_severe'] = np.where(dataset['AQI_Bucket'] != 'Good', 'Severe', 'Not severe')
        dataset['encoded_target'] = le.fit_transform(dataset['AQI_is_severe'])
        drop_list.append('AQI_Bucket')
        dataset.rename(columns={"AQI_is_severe": "text_target"}, inplace=True)
    elif result == 'multiclass':
        le = LabelEncoder()
        dataset['encoded_target'] = le.fit_transform(dataset['AQI_Bucket'])
        dataset.rename(columns={"AQI_Bucket": "text_target"}, inplace=True)
    elif result == 'regression':
        dataset['encoded_target'] = dataset['AQI']
        dataset.rename(columns={"AQI_Bucket": "text_target"}, inplace=True)
    else:
        print('Error - only binary and multiclass model result available')
        exit(0)
    #endif result ==


    # print(dataset[['text_target','encoded_target']])


    dataset = dataset.drop(drop_list, axis=1)

    reduced_dataset = dataset.sample(n=int(dataset.shape[0]*partition_fraction), random_state=1*int(partition_fraction*1000)).copy()

    return reduced_dataset, dataset










def load_MOTOR_dataset2(sensors='motor_all', result='multiclass', partition_fraction=1) -> Dataset:
    # data: https://archive.ics.uci.edu/ml/datasets/Accelerometer
    # Load data

    # algorithm='logreg'
    # num_classes=3
    # sensors='motor_acc'
    # result='multiclass'
    # partition_fraction = 0.5

    dataset = pd.read_csv('datasets/Motor_accelerometer.csv')
    dataset = dataset.dropna()


    # dataset['pctid'].unique()


    if result == 'regression':
        print('Error - only binary and multiclass model result available')
        exit(0)



    # Adjusting sensorss
    # sensors = 'motor_acc'
    if sensors == 'motor_all':
        # drop_list = ['wconfid', 'pctid']
        # x: 33130, y: 33131, z: 33132, pctid: 33460
        key_list = [33460, 33130, 33131, 33132]
        cols = ['pctid', 'x', 'y', 'z', 'wconfid']
    elif sensors == 'motor_acc':
        # drop_list = ['wconfid', 'pctid']
        # x: 33130, y: 33131, z: 33132, pctid: 33460
        key_list = [33130, 33131, 33132]
        cols = ['x', 'y', 'z', 'wconfid']
    else:
        print("Wrong sensors name for drop_list\n")
        exit(0)


    dataset = dataset[cols]


    # if algorithm == 'logreg' or algorithm == 'kmeans':
    #     # Encoding for multiclass and binary classification/grouping


    #     # dataset['target'].unique()
    #     # le.inverse_transform(dataset['target'].unique())


    # drop_list = []





    if result == 'binary':
        le = LabelEncoder()
        dataset['text_target'] = np.where(dataset['wconfid'] == 1, 'Normal', 'Fail')
        dataset['encoded_target'] = le.fit_transform(dataset['text_target'])
    elif result == 'multiclass':
        le = LabelEncoder()
        dataset['text_target'] = np.NaN
        dataset['text_target'].mask(dataset['wconfid'] == 1, 'Normal', inplace=True)
        dataset['text_target'].mask(dataset['wconfid'] == 2, 'Perpendicular', inplace=True)
        dataset['text_target'].mask(dataset['wconfid'] == 3, 'Opposite', inplace=True)
        dataset['encoded_target'] = le.fit_transform(dataset['text_target'])
    else:
        print('Error - only binary and multiclass model result available')
        exit(0)


    # dataset = dataset.sample(n=int(dataset.shape[0]/NUMBER_OF_DATA_PARTITIONS))
    reduced_dataset = dataset.sample(n=int(dataset.shape[0]*partition_fraction), random_state=1*int(partition_fraction*1000)).copy()


    return reduced_dataset, dataset









def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )
