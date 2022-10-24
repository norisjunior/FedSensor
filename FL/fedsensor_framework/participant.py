#import sys
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
import logging
import flwr as fl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.regularizers import l2
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import utils
import lwpubsub_serialization
import subprocess
import time
from time import strftime


def serialize_sensors(sensors_dict):
    sensors_serialized = ';'

    last_key = list(sensors_dict.keys())[-1]

    i = 1
    for key, value in sensors_dict.items():
        sensors_serialized += str(i)
        sensors_serialized += '|0'
        sensors_serialized += str(key)
        if key != last_key: sensors_serialized += ';'
        i += 1

    return sensors_serialized


def print_commands(edgeserver_host, provision_message, command_algo, lwpubsub_msg):
    automated_script_path = './fedsensor-lwaiot-MLModels-automated.sh {} {} {} {}'
    print("Automated script:\n", (automated_script_path .format(edgeserver_host, provision_message, command_algo, lwpubsub_msg)))




class LogRegClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        global x_train, y_train, full_train_history, full_test_history, iteration

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        num_rounds: int = config["rounds"]
        # batch_size = 32
        # epochs = 1
        # num_rounds = 10
        # iteration = 0

        """Train parameters on the locally held training set."""

        # Update local model parameters
        model.set_weights(parameters)

        logging.info(f"iteration: {iteration}")
        x_train_partial = np.array_split(x_train, num_rounds)[iteration]
        y_train_partial = np.array_split(y_train, num_rounds)[iteration]

        logging.info(f"x_train: {x_train_partial}")
        logging.info(f"y_train: {y_train_partial}")
        logging.info(f"len(x_train): {len(x_train_partial)}")
        logging.info(f"len(y_train): {len(y_train_partial)}")


        # Train the model using hyperparameters from config
        history = model.fit(
            x_train_partial, y_train_partial, batch_size=batch_size, epochs=epochs, validation_split=0.1
        )

        iteration += 1

        # history.history['sparse_categorical_accuracy']

        for element in history.history['sparse_categorical_accuracy']:
            full_train_history.append(element)
        for element in history.history['val_sparse_categorical_accuracy']:
            full_test_history.append(element)

        # Return updated model parameters and validation results
        parameters_prime = model.get_weights()
        num_examples_train = len(x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["sparse_categorical_accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_sparse_categorical_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        loss, accuracy = model.evaluate(x_test, y_test, steps=steps)
        logging.info(f"Modelo gerado evaluate(): {model.get_weights()}")
        print("Modelo gerado evaluate(): \n", model.get_weights())
        return loss, len(x_test), {"accuracy": accuracy}





class KMeansClient(fl.client.NumPyClient):
    def get_parameters(self):  # type: ignore
        return utils.get_model_kmeans_parameters(model)

    def fit(self, parameters, config):  # type: ignore
        global x_train, y_train

        utils.set_model_kmeans_params(model, parameters)
        # Ignore convergence failure due to low local epochs
        model.fit(x_train)
        print(f"Training finished for round {config['rnd']}")
        #print("Clusters centers: \n", model.cluster_centers_)
        return utils.get_model_kmeans_parameters(model), len(x_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        global x_test, y_test

        utils.set_model_kmeans_params(model, parameters)

        # Get config values
        steps: int = config["val_steps"]

        loss = model.score(x_test, y_test)
        accuracy = homogeneity_score(model.predict(x_test), y_test)
        print("Modelo gerado evaluate(): \n", model.cluster_centers_)
        return loss, len(x_test), {"accuracy": accuracy}





class LinRegClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        #print("Modelo depois do set weights: \n", model.get_weights())

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        # batch_size = 16
        # epochs = 10

        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        #print("Modelo gerado: \n", model.get_weights())
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        print("Modelo gerado evaluate(): \n", model.get_weights())
        return loss, len(x_test), {"accuracy": accuracy}



def main() -> None:
    global algorithm, n_groups, devices, dataset, sensors, result, n_classes, anomaly_detection, input_dim, partition_id
    command_path = '/IoTArchitecture/FedSensor/EdgeServer/lwaiot_msgs.sh {} {} {} {} {}'
    lwaiot_message1 = '\"32001\" '

    if algorithm == 'logreg':
        # Start Flower client
        fl.client.start_numpy_client("[::]:8080", client=LogRegClient())

        print(f"Train history: {full_train_history}\n")
        print(f"Test history: {full_test_history}\n")
        logging.info(f"Train history: {full_train_history}\n")
        logging.info(f"Test history: {full_test_history}\n")


        print("\nModelo final main(): \n")
        for index, row in enumerate(model.get_weights()[1]):
            print(f"Class {index} - {resulting_dict[index]}: {row}")


        try:
            print('Enter target class value (int): ')
            model_target = input()
            model_target = int(model_target)
            if model_target not in range(model.get_weights()[1].shape[0]): # encoded labels when training ranging from 0 to 10
                print(f'Choose a class from the model provided. Class provided: {model_target}')
                exit(0)
            print('\n')
            print(devices.to_string(index=False))
            print('\nChoose the device (int): ')
            device_target = input()
            device_target = int(device_target)
            if device_target not in devices["Number"].values:
                print(f'Choose the device from the list above. Device provided: {device_target}')
                exit(0)
            print('\nInform the LWPubSub IoT Agent address (string): ')
            # device_target = 5
            device_name = '\"' + devices[devices['Number'] == device_target]['DeviceName'].values[0] + '\"'
            device_ID = '\"' + devices[devices['Number'] == device_target]['DeviceID'].values[0] + '\"'
            edgeserver_host = input()
            edgeserver_host = '\"' + edgeserver_host + '\"'
            command_algo = '\"32103\" '
            lwpubsub_msg = lwpubsub_serialization.logreg_message(decision=model_target, var_intercept=model.get_weights()[1], var_weights=model.get_weights()[0].T)
            lwpubsub_msg = '\"' + lwpubsub_msg + '\" '

            #Send the commands:
            #Message 32001 (activating the sensors)
            print('\nInform time interval to gather measurements with 3 digits (int): ')
            time_target = input()
            time_target = int(time_target)
            time_target = '\"' + str(time_target).zfill(3)
            experiment_sensors = serialize_sensors(features_dict)
            experiment_sensors += '\" '
            provision_message = time_target + experiment_sensors
            os.system(command_path .format(edgeserver_host, lwaiot_message1, provision_message, device_name, device_ID))
            time.sleep(2)
            print("Message 32001 sent. Sensors configured.\n")
            print("Command executed: \n", (command_path .format(edgeserver_host, lwaiot_message1, provision_message, device_name, device_ID)))

            os.system(command_path .format(edgeserver_host, command_algo, lwpubsub_msg, device_name, device_ID))
            print("Global model sent.\n")
            print("Command executed: \n", (command_path .format(edgeserver_host, command_algo, lwpubsub_msg, device_name, device_ID)))

            print_commands(edgeserver_host, provision_message, command_algo, lwpubsub_msg)

        except ValueError:
            # Handle the exception
            print('Please enter an positive integer')
            exit(0)




    elif algorithm == 'kmeans':
        # Start Flower client
        fl.client.start_numpy_client("[::]:8081", client=KMeansClient())
        print("\nModelo final main(): \n", model.cluster_centers_)
        for index, row in enumerate(model.cluster_centers_):
            print(f"Group {index} - {resulting_dict[index]}: {row}")

        try:
            print('Enter target group: ')
            model_target = input()
            model_target = int(model_target)
            if model_target not in range(n_groups):
                print(f'Choose a group from the model provided. Class provided: {model_target}')
                exit(0)
            print('\n')
            print(devices.to_string(index=False))
            print('\nChoose the device (int): ')
            device_target = input()
            device_target = int(device_target)
            if device_target not in devices["Number"].values:
                print(f'Choose the device from the list above. Device provided: {device_target}')
                exit(0)
            print('\nInform the LWPubSub IoT Agent address (string): ')
            device_name = '\"' + devices[devices['Number'] == device_target]['DeviceName'].values[0] + '\"'
            device_ID = '\"' + devices[devices['Number'] == device_target]['DeviceID'].values[0] + '\"'
            edgeserver_host = input()
            edgeserver_host = '\"' + edgeserver_host + '\"'
            command_algo = '\"32106\" '
            lwpubsub_msg = lwpubsub_serialization.kmeans_message(decision=model_target, var_weights=model.cluster_centers_)
            print(lwpubsub_msg)
            lwpubsub_msg = '\"' + lwpubsub_msg + '\" '

            #Send the commands:
            #Message 32001 (activating the sensors)
            print('\nInform time interval to gather measurements with 3 digits (int): ')
            time_target = input()
            time_target = int(time_target)
            time_target = '\"' + str(time_target).zfill(3)
            experiment_sensors = serialize_sensors(features_dict)
            experiment_sensors += '\" '
            provision_message = time_target + experiment_sensors
            os.system(command_path .format(edgeserver_host, lwaiot_message1, provision_message, device_name, device_ID))
            time.sleep(2)
            print("Message 32001 sent. Sensors configured.\n")
            print("Command executed: \n", (command_path .format(edgeserver_host, lwaiot_message1, provision_message, device_name, device_ID)))

            os.system(command_path .format(edgeserver_host, command_algo, lwpubsub_msg, device_name, device_ID))
            print("Global model sent.\n")
            print("Command executed: \n", (command_path .format(edgeserver_host, command_algo, lwpubsub_msg, device_name, device_ID)))

            print_commands(edgeserver_host, provision_message, command_algo, lwpubsub_msg)

        except ValueError:
            # Handle the exception
            print('Please enter an positive integer')
            exit(0)





    elif algorithm == 'linreg':
        # Start Flower client
        fl.client.start_numpy_client("[::]:8082", client=LinRegClient())
        print("\nModelo final main(): \n", model.get_weights())

        try:
            print('Enter target value: ')
            model_target = input()
            model_target = int(model_target)
            print('\n')

            print(devices.to_string(index=False))
            print('\nChoose the device (int): ')
            device_target = input()
            device_target = int(device_target)
            if device_target not in devices["Number"].values:
                print(f'Choose the device from the list above. Device provided: {device_target}')
                exit(0)

            print('\nInform the LWPubSub IoT Agent address (string): ')
            device_name = '\"' + devices[devices['Number'] == device_target]['DeviceName'].values[0] + '\"'
            device_ID = '\"' + devices[devices['Number'] == device_target]['DeviceID'].values[0] + '\"'
            edgeserver_host = input()
            edgeserver_host = '\"' + edgeserver_host + '\"'
            command_algo = '\"32102\" '
            lwpubsub_msg = lwpubsub_serialization.linreg_message(decision=model_target, var_intercept=model.get_weights()[1].item(0), var_weights=model.get_weights()[0])
            lwpubsub_msg = '\"' + lwpubsub_msg + '\" '

            #Send the commands:
            #Message 32001 (activating the sensors)
            print('\nInform time interval to gather measurements with 3 digits (int): ')
            time_target = input()
            time_target = int(time_target)
            time_target = '\"' + str(time_target).zfill(3)
            experiment_sensors = serialize_sensors(features_dict)
            experiment_sensors += '\" '
            provision_message = time_target + experiment_sensors
            os.system(command_path .format(edgeserver_host, lwaiot_message1, provision_message, device_name, device_ID))
            time.sleep(2)
            print("Message 32001 sent. Sensors configured.\n")
            print("Command executed: \n", (command_path .format(edgeserver_host, lwaiot_message1, provision_message, device_name, device_ID)))

            os.system(command_path .format(edgeserver_host, command_algo, lwpubsub_msg, device_name, device_ID))
            print("Global model sent.\n")
            print("Command executed: \n", (command_path .format(edgeserver_host, command_algo, lwpubsub_msg, device_name, device_ID)))

            print_commands(edgeserver_host, provision_message, command_algo, lwpubsub_msg)

        except ValueError:
            # Handle the exception
            print('Please enter an positive integer')
            exit(0)

    else:
        print('Algorithm error! Algorithm: ', algorithm)
        exit(0)

    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

    logging.shutdown()
    #end def main()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedSensor Participant")
    parser.add_argument(
        "--logexpname",
        type=str,
        help="logfilename",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        help="Algorithm. Available algorithms: logreg (Logistic Regression), kmeans (k-means), linreg (Linear Regression)",
    )
    parser.add_argument(
        "--sensors",
        required=True,
        type=str,
        help="Sensors of the MOTOR dataset: 'motor_all' (motor velocity, acc. X, acc. Y, acc. Z), 'motor_acc' (acc. X, acc. Y, acc. Z) \
             Sensors of the AQI dataset: \
                     'aqi_pm_only' (PM2.5, PM10), 'aqi_pm_plus' \
                     (PM2.5, PM10, CO, O3), or 'aqi_all' (all sensors) \n \
             Sensors of the FALLUP dataset: \
                     'fallup_all' (all accelerometer and velocity from \
                     ankle, right procket, belt, neck, and wrist; plus BrainSensor \
                     and luminosity), \
                     'fallup_acc_veloc' (only all accelerometer and velocity sensors), \
                    'fallup_acc_only' (only all accelerometer sensors), \
                    'fallup_velocity_only' (only all velocity sensors), \
                    'fallup_others_only' (only other sensors: Luminosity and Brain) \
            Sensors of the PERSON dataset: 'person_all' (accelerometer X, Y, Z) \
            Sensors of the UMAFALL dataset: 'umafall_all' (accelerometer X, Y, Z)",
    )
    parser.add_argument(
        "--result",
        type=str,
        default='multiclass',
        help="Model result. Available results: regression (exclusive for Linear Regression), multiclass and binary (both for Logistic Regression and k-means)",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=3,
        help="Integer number of classes. For this AQI experiment values must be 3 (default) or 6",
    )
    parser.add_argument(
        "--n_groups",
        type=int,
        default=3,
        help="Integer number of cluster groups. Default = 3",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset. Available datasets: AQI, MOTOR, FALLUP, PERSON, UMAFALL",
    )
    parser.add_argument(
        "--anomaly_detection",
        type=str,
        default='No',
        required=False,
        help="Anomaly detector. Available algorithms: No (default - no anomaly detection), ECOD, IForest, LOF",
    )
    parser.add_argument(
        "--partition",
        type=float,
        required=True,
        help="Partition fraction. Max = 1 (no partition, full dataset training).",
    )
    args = parser.parse_args()

    algorithm = args.algorithm
    sensors = args.sensors
    result = args.result
    dataset = args.dataset
    anomaly_detection = args.anomaly_detection
    partition_id = args.partition
    if partition_id > 1: partition_id = 0.99


    devicesList = {'Number': [1, 2, 3, 4, 5],
        'DeviceName': ['Native', 'Sensortag', 'Remote', 'Launchpad', 'WindowsWiFi'],
        'DeviceID': ['010203060708', '00124b05257a', '00124b4a527d', '00124ba1ad06', '00122518499d']}
    devices = pd.DataFrame(devicesList)



    if args.algorithm not in utils.algorithms_available:
        print('Algorithm not available. Available algorithms: ', utils.algorithms_available)
        exit(0)


    if result == 'binary' and ((args.n_classes != 2) or (args.n_groups != 2)):
        print('Number of classes/groups differ from 2. We set it to 2.')
        n_classes = 2
        n_groups = 2
    else:
        n_classes = args.n_classes
        n_groups = args.n_groups


    if algorithm == 'logreg':
        # algorithm = 'logreg'
        # sensors = 'aqi_all'
        # result = 'multiclass'
        # n_classes = 6
        # partition_id = 0.51
        # dataset = 'AQI'
        # model_target = 1
        # anomaly_detection = 'IForest'

        if args.n_classes < 2:
            print('Incorrect number of classes: minimum 2, provided: ', args.n_classes)
            exit(0)

        if args.result == 'regression':
            print('Regression result only available for Linear Regression algorithm. Provided: ', algorithm)
            exit(0)

        if dataset == 'AQI':
            # Load dataset
            partial_dataset, full_dataset = utils.load_AQI_dataset2(algorithm=algorithm, num_classes=n_classes, sensors=sensors, result=result, partition_fraction=partition_id)
            no_anomaly_dataset = utils.anomaly_detection(partial_dataset, anomaly_detection=anomaly_detection)
            (x_train, y_train), (x_test, y_test), features_dict, resulting_dict = utils.encode_dataset(no_anomaly_dataset, full_dataset, sensors=sensors, result=result)
        elif dataset == 'FALLUP':
            # Load dataset
            (x_train, y_train), (x_test, y_test), features_dict, resulting_dict = utils.load_FALLUP_dataset(algorithm=algorithm, num_classes=n_classes, sensors=sensors, result=result)
        elif dataset == 'PERSON':
            # Load dataset
            (x_train, y_train), (x_test, y_test), features_dict, resulting_dict = utils.load_PERSON_dataset(algorithm=algorithm, sensors=sensors, result=result)
        elif dataset == 'UMAFALL':
            # Load dataset
            (x_train, y_train), (x_test, y_test), features_dict, resulting_dict = utils.load_UMAFALL_dataset(algorithm=algorithm, sensors=sensors, result=result)
        elif dataset == 'MOTOR':
            # Load dataset
            partial_dataset, full_dataset = utils.load_MOTOR_dataset2(sensors=sensors, result=result, partition_fraction=partition_id)
            no_anomaly_dataset = utils.anomaly_detection(partial_dataset, anomaly_detection=anomaly_detection)
            (x_train, y_train), (x_test, y_test), features_dict, resulting_dict = utils.encode_dataset(no_anomaly_dataset, full_dataset, sensors=sensors, result=result)
        else:
            print("Dataset not available.")
            exit(0)


        logging.info(f"TRAIN size: {x_train.shape[0]}")
        logging.info(f"TEST size: {x_test.shape[0]}")
        print("TRAIN size: ", x_train.shape[0])
        print("TEST size: ", x_test.shape[0])


        input_dim = len(x_train[0, :]) # number of variables
        output_dim = len(np.unique(y_train)) # number of possible outputs

        if (output_dim == 2):
            from_logits = True
        else:
            from_logits = False

        model = Sequential()
        model.add(Dense(output_dim, input_dim=input_dim, activation='softmax', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.001)))
        optim = keras.optimizers.SGD(learning_rate=0.001, momentum=0.01)
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
        model.compile(loss=loss, optimizer=optim, metrics=['sparse_categorical_accuracy'])




    if algorithm == 'kmeans':
        # algorithm = 'kmeans'
        # sensors = 'motor_acc'
        # result = 'multiclass'
        # n_groups = 3
        # partition_id = 0.1
        # dataset = 'MOTOR'
        # model_target = 1
        # anomaly_detection = 'IForest'

        if args.n_groups < 2:
            print('Incorrect number of groups: minimum 2, provided: ', args.n_groups)
            exit(0)

        if args.result == 'regression':
            print('Regression result only available for Linear Regression algorithm. Provided: ', algorithm)
            exit(0)

        if dataset == 'AQI':
            # Load dataset
            partial_dataset, full_dataset = utils.load_AQI_dataset2(algorithm=algorithm, num_classes=n_classes, sensors=sensors, result=result, partition_fraction=partition_id)
            no_anomaly_dataset = utils.anomaly_detection(partial_dataset, anomaly_detection=anomaly_detection)
            (x_train, y_train), (x_test, y_test), features_dict, resulting_dict = utils.encode_dataset(no_anomaly_dataset, full_dataset, sensors=sensors, result=result)
        elif dataset == 'FALLUP':
            # Load dataset
            (x_train, y_train), (x_test, y_test), features_dict, resulting_dict = utils.load_FALLUP_dataset(algorithm=algorithm, sensors=sensors, result=result)
        elif dataset == 'PERSON':
            # Load dataset
            (x_train, y_train), (x_test, y_test), features_dict, resulting_dict = utils.load_PERSON_dataset(algorithm=algorithm, sensors=sensors, result=result)
        elif dataset == 'UMAFALL':
            # Load dataset
            (x_train, y_train), (x_test, y_test), features_dict, resulting_dict = utils.load_UMAFALL_dataset(algorithm=algorithm, sensors=sensors, result=result)
        elif dataset == 'MOTOR':
            # Load dataset
            partial_dataset, full_dataset = utils.load_MOTOR_dataset2(sensors=sensors, result=result, partition_fraction=partition_id)
            no_anomaly_dataset = utils.anomaly_detection(partial_dataset, anomaly_detection=anomaly_detection)
            (x_train, y_train), (x_test, y_test), features_dict, resulting_dict = utils.encode_dataset(no_anomaly_dataset, full_dataset, sensors=sensors, result=result)
        else:
            print("Dataset not available.")
            exit(0)

        logging.info(f"TRAIN size: {x_train.shape[0]}")
        logging.info(f"TEST size: {x_test.shape[0]}")
        print("TRAIN size: ", x_train.shape[0])
        print("TEST size: ", x_test.shape[0])

        model = KMeans(n_clusters=n_groups)

        input_dim = len(x_train[0, :]) # number of variables

        # Setting initial parameters, akin to model.compile for keras models
        utils.set_initial_kmeans_params(model, dataset)





    if algorithm == 'linreg':
        # algorithm = 'linreg'
        # sensors = 'aqi_all'
        # result = 'regression'
        # partition_id = 0.3
        # dataset = 'AQI'
        # model_target = 250
        # anomaly_detection = 'IForest'
        # logexpname = 'anomaly_exp_AQI_logreg_9_sensores_2_classes_0__5_No_anomalydetection.log'


        result = 'regression'

        if dataset == 'AQI':
            # Load dataset
            partial_dataset, full_dataset = utils.load_AQI_dataset2(algorithm=algorithm, sensors=sensors, result=result, partition_fraction=partition_id)
            no_anomaly_dataset = utils.anomaly_detection(partial_dataset, anomaly_detection=anomaly_detection)
            (x_train, y_train), (x_test, y_test), features_dict, resulting_dict = utils.encode_dataset(no_anomaly_dataset, full_dataset, sensors=sensors, result=result)
        else:
            print("Dataset not available.")
            exit(0)

        logging.info(f"TRAIN size: {x_train.shape[0]}")
        logging.info(f"TEST size: {x_test.shape[0]}")
        print("TRAIN size: ", x_train.shape[0])
        print("TEST size: ", x_test.shape[0])

        input_dim = len(x_train[0, :]) # number of variables

        model = Sequential()
        model.add(Dense(1, input_dim=input_dim))
        optim = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(loss='mse', optimizer=optim, metrics=[keras.metrics.RootMeanSquaredError()])

    full_train_history = []
    full_test_history = []
    iteration = 0

    #now we will Create and configure logger
    logfile_time = '{}'.format(strftime('%Y-%m-%d_%H-%M-%S'))

    logfile = 'part_' + str(partition_id) + "_" + str(input_dim) + '_' + str(n_classes) + '_' + str(anomaly_detection) + '_' + str(args.logexpname)

    #print(logfile)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=logfile, level = logging.INFO,
                        format='%(asctime)s | %(message)s',
                        filemode='w')

    #Let us Create an object
    logger=logging.getLogger()

    #Now we are going to Set the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)
    logging.info(logfile_time)
    logging.info(logfile)


    main()
