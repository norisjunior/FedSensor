import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple
import flwr as fl
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from keras.utils import np_utils
from keras.regularizers import l2
from tensorflow import keras
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score
import utils

from time import strftime

import logging

rounds = 50

def main() -> None:

    global algorithm, x_val, y_val
    input_dim = len(x_val[0, :]) # number of variables

    if algorithm == 'logreg':
        output_dim = len(np.unique(y_val)) # number of possible outputs

        if (output_dim == 2):
            from_logits = True
        else:
            from_logits = False

        model = Sequential()
        model.add(Dense(output_dim, input_dim=input_dim, activation='softmax', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.001)))
        # model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))
        # optim = keras.optimizers.Adam(learning_rate=0.01)
        optim = keras.optimizers.SGD(learning_rate=0.001, momentum=0.01)
        # optim = keras.optimizers.SGD(learning_rate=0.1)
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
        model.compile(loss=loss, optimizer=optim, metrics=['sparse_categorical_accuracy'])

        # Create strategy
        # Enabling 'eval_fn' option, automatically nullify 'on_fit_config_fn',
        # or the evaluation occurs on the server, with 'eval_fn'
        # or the evaluation occurs on the client, with 'on_fit_config_fn'
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1,
            fraction_eval=1,
            min_fit_clients=3,
            min_eval_clients=3,
            min_available_clients=3,
            #eval_fn=keras_get_eval_fn(model),
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
        )

        # Start Flower server for four rounds of federated learning
        fl.server.start_server("[::]:8080", config={"num_rounds": rounds}, strategy=strategy)

        #print("Modelo gerado: \n", model.get_weights())

        logging.info("Sem_anomalia ")
        logging.info("Com_anomalia ")



    if algorithm == 'kmeans':
        global n_groups, dataset

        #model = KMeans(n_clusters=n_groups).fit(X_test[:10])
        model = KMeans(n_clusters=n_groups)

        utils.set_initial_kmeans_params(model, dataset)


        # Enabling 'eval_fn' option, automatically nullify 'on_fit_config_fn',
        # or the evaluation occurs on the server, with 'eval_fn'
        # or the evaluation occurs on the client, with 'on_fit_config_fn'
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1,
            fraction_eval=1,
            min_fit_clients=3,
            min_eval_clients=3,
            min_available_clients=3,
            #eval_fn=kmeans_get_eval_fn(model),
            on_fit_config_fn=fit_round,
            on_evaluate_config_fn=evaluate_config,
        )
        fl.server.start_server("0.0.0.0:8081", strategy=strategy, config={"num_rounds": 10})


    if algorithm == 'linreg':

        model = Sequential()
        model.add(Dense(1, input_dim=input_dim))
        # optim = keras.optimizers.Adam(learning_rate=0.01)
        optim = keras.optimizers.SGD(learning_rate=0.1, momentum=0.01)
        model.compile(loss='mse', optimizer=optim, metrics=[keras.metrics.RootMeanSquaredError()])


        # Create strategy
        # Enabling 'eval_fn' option, automatically nullify 'on_fit_config_fn',
        # or the evaluation occurs on the server, with 'eval_fn'
        # or the evaluation occurs on the client, with 'on_fit_config_fn'
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1,
            fraction_eval=1,
            min_fit_clients=3,
            min_eval_clients=3,
            min_available_clients=3,
            #eval_fn=keras_get_eval_fn(model),
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
        )

        # Start Flower server for four rounds of federated learning
        fl.server.start_server("[::]:8082", config={"num_rounds": 10}, strategy=strategy)

    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

    logging.shutdown()
    #end def main()



def keras_get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    global x_val, y_val

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        print("Modelo gerado: \n", model.get_weights())
        loss, accuracy = model.evaluate(x_val, y_val)
        return loss, {"accuracy": accuracy}

    return evaluate


def kmeans_get_eval_fn(model: KMeans):
    """Return an evaluation function for server-side evaluation."""
    global x_val, y_val

    # The `evaluate` function will be called after every round
    def evaluate(parameters: fl.common.Weights):
        # Update model with the latest parameters
        utils.set_model_kmeans_params(model, parameters)
        #loss = log_loss(y_test, model.predict_proba(X_test))
        print("Modelo gerado: \n", model.cluster_centers_)
        loss = model.score(x_val, y_val)
        accuracy = homogeneity_score(model.predict(x_val), y_val)
        return loss, {"accuracy": accuracy}

    return evaluate




def fit_round(rnd: int) -> Dict:
    """Send round number to client."""
    return {"rnd": rnd}


def fit_config(rnd: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1,
        "rounds": rounds
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5
    return {"val_steps": val_steps}





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedSensor Manager")
    parser.add_argument(
        "--logfilename",
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
        #default='aqi_pm_plus',
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
                    'fallup_others_only' (only other sensors: Luminosity and Brain)",
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
        help="Dataset. Available datasets: AQI, FALLUP, PERSON, UMAFALL",
    )
    args = parser.parse_args()

    algorithm = args.algorithm
    sensors = args.sensors
    result = args.result
    dataset = args.dataset

    # algorithm = 'logreg'
    # sensors = 'aqi_pm_plus'
    # result = 'binary'
    # n_classes = 2
    # n_groups = 2
    # dataset = 'AQI'


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
        if args.n_classes < 2:
            print('Incorrect number of classes: minimum 2, provided: ', args.n_classes)
            exit(0)

        if args.result == 'regression':
            print('Regression result only available for Linear Regression algorithm. Provided: ', algorithm)
            exit(0)

        if dataset == 'AQI':
            # Load dataset
            (_, _), (x_val, y_val), feature_list, resulting_dict = utils.load_AQI_dataset(algorithm=algorithm, num_classes=n_classes, sensors=sensors, result=result)
        elif dataset == 'FALLUP':
            # Load dataset
            print("Por enquanto, FALL UP funcionando com 11 classes. Substituindo qualquer <n_classes> informado por 11")
            n_classes=11
            (_, _), (x_val, y_val), feature_list, resulting_dict = utils.load_FALLUP_dataset(algorithm=algorithm, num_classes=n_classes, sensors=sensors, result=result)
        elif dataset == 'PERSON':
            # Load dataset
            (_, _), (x_val, y_val), feature_list, resulting_dict = utils.load_PERSON_dataset(algorithm=algorithm, sensors=sensors, result=result)
        elif dataset == 'UMAFALL':
            # Load dataset
            (_, _), (x_val, y_val), feature_list, resulting_dict = utils.load_UMAFALL_dataset(algorithm=algorithm, sensors=sensors, result=result)
        elif dataset == 'MOTOR':
            # Load dataset
            (_, _), (x_val, y_val), feature_list, resulting_dict = utils.load_MOTOR_dataset(algorithm=algorithm, sensors=sensors, result=result)
        else:
            print("Dataset not available.")
            exit(0)



    if algorithm == 'kmeans':
        if args.n_groups < 2:
            print('Incorrect number of groups: minimum 2, provided: ', args.n_groups)
            exit(0)

        if args.result == 'regression':
            print('Regression result only available for Linear Regression algorithm. Provided: ', algorithm)
            exit(0)


        if dataset == 'AQI':
            # Load dataset
            (_, _), (x_val, y_val), feature_list, resulting_dict = utils.load_AQI_dataset(algorithm=algorithm, sensors=sensors, result=result)
        elif dataset == 'FALLUP':
            # Load dataset
            (_, _), (x_val, y_val), feature_list, resulting_dict = utils.load_FALLUP_dataset(algorithm=algorithm, sensors=sensors, result=result)
        elif dataset == 'PERSON':
            # Load dataset
            (_, _), (x_val, y_val), feature_list, resulting_dict = utils.load_PERSON_dataset(algorithm=algorithm, sensors=sensors, result=result)
        elif dataset == 'UMAFALL':
            # Load dataset
            (_, _), (x_val, y_val), feature_list, resulting_dict = utils.load_UMAFALL_dataset(algorithm=algorithm, sensors=sensors, result=result)
        elif dataset == 'MOTOR':
            # Load dataset
            (_, _), (x_val, y_val), feature_list, resulting_dict = utils.load_MOTOR_dataset(algorithm=algorithm, sensors=sensors, result=result)
        else:
            print("Dataset not available.")
            exit(0)





    if algorithm == 'linreg':
        # if args.result != 'regression':
        #     print('Linear Regression require regression result. Provided: ', args.result)
        #     exit(0)
        result = 'regression'

        if dataset == 'AQI':
            # Load dataset
            (_, _), (x_val, y_val), feature_list, resulting_dict = utils.load_AQI_dataset(algorithm=algorithm, sensors=sensors, result=result)
        else:
            print("Dataset not available.")
            exit(0)

    n_sensors = input_dim = len(x_val[0, :])

    logfile = args.logfilename

    #now we will Create and configure logger
    logfile_time = '{}'.format(strftime('%Y-%m-%d_%H-%M-%S'))
    logging.basicConfig(filename=logfile,
                        format='%(asctime)s | %(message)s',
                        filemode='w')

    #Let us Create an object
    logger=logging.getLogger()

    #Now we are going to Set the threshold of logger to DEBUG
    logger.setLevel(logging.INFO)

    logging.info("anomalyexp_" + str(dataset) + '_' + str(algorithm) + '_' + str(n_sensors) + 'sensores_' + str(n_classes) + 'classes_' + logfile_time + '.log')

    main()
