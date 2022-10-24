import time
import paho.mqtt.client as mqtt
from getmac import get_mac_address
import platform
import random
import sys
#import pycryptodome
from Crypto.Cipher import AES
import Crypto.Cipher.AES
import Crypto.Util.Counter
from binascii import hexlify, unhexlify, b2a_uu
import os
import platform
import re
import pandas as pd
import numpy as np
from math import sqrt

if ( (platform.system() == 'Linux') and ( (platform.machine() == 'armv7l') or (platform.machine() == 'armv6l') ) ):
    from gpiozero import LED


###############################################################################
# Variables

LWML4AIoTMessages = pd.DataFrame({
    'objectID': [32001, 32002],
    'Message': ['Message 1 - Poll interval and sensor enablement', 'Mensagem 4 - decision making']
})

SupportedModels = pd.DataFrame({ #Message 3
    'objectID': [32101, 32102, 32103, 32104, 32105, 32106],
    'ModelName': ['Simple Linear Regression', 'Multiple Linear Regression', 'Logistic Regression', 'Naive Bayes', 'Artifical Neural Network', 'k-means'],
    'Class': ['Regression', 'Regression', 'Regression', 'Classification', 'Classification', 'Grouping']
})

IPSOSensors = pd.DataFrame({
    'objectID': [3301, 3302, 3303, 3304, 3305, 3306, 3311, 3313, 3323, 3325, 3338],
    'objectName': ['Illuminance', 'Presence', 'Temperatute', 'Humidity', 'Power Measurement', 'Actuation', 'Light Control', 'Accelerometer', 'Pressure', 'Concentration', 'Buzzer']
})



















###############################################################################
# Crypto functions

def aes_cbc_encrypt(payload, lwpubsub_key):
    IV = os.urandom(16)
    cipher = AES.new(lwpubsub_key,AES.MODE_CBC,IV)
    ciphertext = cipher.encrypt(payload)
    return IV, ciphertext


def aes_cbc_decrypt(lwpubsub_payload, lwpubsub_key, IV):
    cipher = AES.new(lwpubsub_key,AES.MODE_CBC,IV)
    plaintext = cipher.decrypt(lwpubsub_payload)
    return plaintext


def aes_ctr_encrypt(payload, lwpubsub_key):
    IV = os.urandom(16)
    ctr = Crypto.Util.Counter.new(128, initial_value=int(hexlify(IV), 16))
    cipher = AES.new(lwpubsub_key,AES.MODE_CTR,counter=ctr)
    ciphertext = cipher.encrypt(payload)
    return IV, ciphertext


def aes_ctr_decrypt(lwpubsub_payload, lwpubsub_key, IV):
    cipher = AES.new(lwpubsub_key,AES.MODE_CTR,IV)
    plaintext = cipher.decrypt(lwpubsub_payload)
    return plaintext


def aes_ccm_8_encrypt(payload, lwpubsub_key):
    nonce = os.urandom(13)
    #print("nonce:", nonce)
    aad = str.encode(pub_topic) #pub_topic = aditional authenticated data (aad)
    #print("aad: ", aad)
    cipher = AES.new(lwpubsub_key, AES.MODE_CCM, nonce=nonce, mac_len=8)
    cipher.update(aad)
    payload_bytes = str.encode(payload)
    ciphertext = cipher.encrypt(payload_bytes)
    #print("Ciphertext: ", ciphertext)
    auth_tag = cipher.digest()
    #print("Auth tag: ", auth_tag)
    return nonce, ciphertext, auth_tag


def aes_ccm_8_decrypt(lwpubsub_key, lwpubsub_payload, nonce, aad):
    cipher = AES.new(lwpubsub_key, AES.MODE_CCM, nonce=nonce, mac_len=8)
    cipher.update(aad)
    plaintext = cipher.decrypt(lwpubsub_payload)
    # try:
    #     cipher.verify(aad)
    #     print("The message is authentic: hdr=%s, pt=%s", aad, plaintext)
    # except ValueError:
    #     print("Incorrect key or auth_tag")

    return plaintext



###############################################################################
###############################################################################


















###############################################################################
###############################################################################
############################################
#Sensor measurements

def read_temp():
    #Dummy temp for tests
    t = '033030|' + str(random.randint(15, 30)) + "." + str(random.randint(0, 90))
    return t

def read_humidity():
    #Dummy humid for tests
    h = '033040|' + str(random.randint(35, 80)) + "." + str(random.randint(0, 90))
    return h


def read_dummy_AQI():
    #Random reading
    global count_dummy_AQI
    if count_dummy_AQI > 7:
        rand_reading = str(random.randint(40, 50)) + "." + str(random.randint(10, 90))
        count_dummy_AQI = 0
    else:
        rand_reading = str(random.randint(0, 1)) + "." + str(random.randint(0, 10))

    count_dummy_AQI += 1
    return rand_reading


def read_dummy_MOTOR():
    #Random reading
    global count_dummy_motor
    if count_dummy_motor > 7:
        rand_reading = str(random.randint(2, 4)) + "." + str(random.randint(10, 90))
        count_dummy_motor = 0
    else:
        rand_reading = str(random.randint(0, 1)) + "." + str(random.randint(0, 10))

    count_dummy_motor += 1
    return rand_reading

###############################################################################
#Motor
def read_033130():
    #Dummy value
    dummy_value = read_dummy_MOTOR()
    return dummy_value

def read_033131():
    #Dummy value
    dummy_value = read_dummy_MOTOR()
    return dummy_value

def read_033132():
    #Dummy value
    dummy_value = read_dummy_MOTOR()
    return dummy_value

def read_033460():
    #Dummy value
    global count_dummy_motor
    if count_dummy_motor > 7:
        pctid = str(random.randint(80, 100))
        count_dummy_motor = 0
    else:
        pctid = str(random.randint(20, 21))

    count_dummy_motor += 1
    return pctid
###############################################################################



###############################################################################
#AQI
def read_033250():
    #Dummy pm2.5
    global count_pm25
    if count_pm25 > 7:
        pm25 = str(random.randint(280, 320)) + "." + str(random.randint(0, 90))
        count_pm25 = 0
    else:
        pm25 = str(random.randint(5, 30)) + "." + str(random.randint(0, 90))

    count_pm25 += 1

    return pm25


def read_033251():
    #Dummy pm10
    global count_pm10
    if count_pm10 > 7:
        pm10 = str(random.randint(370, 400)) + "." + str(random.randint(0, 90))
        count_pm10 = 0
    else:
        pm10 = str(random.randint(10, 40)) + "." + str(random.randint(0, 90))

    count_pm10 += 1


    return pm10


def read_033252():
    #Dummy value
    dummy_value = read_dummy_AQI()
    return dummy_value

def read_033253():
    #Dummy value
    dummy_value = read_dummy_AQI()
    return dummy_value

def read_033254():
    #Dummy value
    dummy_value = read_dummy_AQI()
    return dummy_value

def read_033255():
    #Dummy value
    dummy_value = read_dummy_AQI()
    return dummy_value


def read_033256():
    #Dummy CO
    global count_co
    if count_co > 7:
        co = str(random.randint(3, 4)) + "." + str(random.randint(70, 90))
        count_co = 0
    else:
        co = str(random.randint(0, 1)) + "." + str(random.randint(0, 10))

    count_co += 1

    return co

def read_033257():
    #Dummy value
    dummy_value = read_dummy_AQI()
    return dummy_value

def read_033258():
    #Dummy O3
    o3 = str(random.randint(1, 15)) + "." + str(random.randint(0, 90))
    return o3

def read_033259():
    #Dummy value
    dummy_value = read_dummy_AQI()
    return dummy_value

###############################################################################



def read_led(led_instance):
    #check if it is 33110 (red) or 33111 (green)
    global green_led
    global red_led
    global blue_led

    if led_instance not in [0, 1, 2]:
        print("Wrong instance")
    elif led_instance == 0:
        #red led
        lwpubsub_led = '03311' + led_instance + '|' + red_led
    elif led_instance == 1:
        #green led
        lwpubsub_led = '03311' + led_instance + '|' + green_led
    elif led_instance == 2:
        #green led
        lwpubsub_led = '03311' + led_instance + '|' + blue_led
    return lwpubsub_led


def set_led(led_instance, state):
    global green_led
    global red_led
    global blue_led

    if ( (platform.system() == 'Linux') and ( (platform.machine() == 'armv7l') or (platform.machine() == 'armv6l') ) ):
        global gpio_green
        global gpio_red
        global gpio_blue

        print("Entrou gpio\n\n ")
        print("GPIO green: ", gpio_green)
        print("GPIO red: ", gpio_red)
        print("GPIO blue: ", gpio_blue)

    print("LED instance:", led_instance)
    print("State:", state)

    if (led_instance not in ['0', '1', '2']):
        print("Wrong instance")
    elif (led_instance == '0'):
        #red led
        red_led = state
        print("\n\n> RED LED <");
        if ( (platform.system() == 'Linux') and ( (platform.machine() == 'armv7l') or (platform.machine() == 'armv6l') ) ):
            gpio_red.on() if state == '1' else gpio_red.off()
    elif (led_instance == '1'):
        #green led
        green_led = state
        print("\n\n> GREEN LED <");
        if ( (platform.system() == 'Linux') and ( (platform.machine() == 'armv7l') or (platform.machine() == 'armv6l') ) ):
            gpio_green.on() if state == '1' else gpio_green.off()
    elif (led_instance == '2'):
        #green led
        blue_led = state
        print("\n\n> BLUE LED <");
        if ( (platform.system() == 'Linux') and ( (platform.machine() == 'armv7l') or (platform.machine() == 'armv6l') ) ):
            gpio_blue.on() if state == '1' else gpio_blue.off()

    print(" - Set led to: ", state)
    print("     >ON<\n") if state == '1' else print("     >OFF<\n")


###############################################################################
###############################################################################


















###############################################################################
# LWAIoT and LWML4AIoT functions

def measurementRequest(oID, iID, value):
    if (oID == '03311'): #led: execute command (turn led on/off)
        print("[LWPubSub] - LED - Command execute received")
        set_led(iID, value)

    if (oID == '03303' or oID == '03304'): #temperature: request command
        print("[LWPubSub] - Message request received for objectID ", oID)
        lwpubsub_publish(oID)
    return


def processLWML4AIoTMessage(payload):
    global SupportedModels
    #global lwml4aiot_model
    global poll_frequency
    global lwml4aiot_sensors

    #payload = "320010|32105"
    #print("\n\npayload: ", payload)

    oID = payload[0:payload.find('|')-1]
    payload = payload[payload.find('|')-1:]

    if (oID == '32001'): #Message1
        # #Define in global variable  'lwml4aiot_model' the ML model
        # #to be used by LWML4AIoT on the device
        # print("\n##################################################################################\nMessage 1\n")

        # iID = payload[0]
        # value = payload[payload.find('|')+1:]

        # if (iID == '0'):
        #     lwml4aiot_model = int(value)
        # else:
        #     print("Wrong instanceID")

        # print("Model defined: ", lwml4aiot_model, SupportedModels[SupportedModels['objectID'] == lwml4aiot_model]['ModelName'].values[0])

        global lwml4aiot_parameters_number
        global poll_frequency
        global make_decision

        #Define poll_frequency (using the global variable 'poll_frequency')
        #and all the sensors to be used
        print("\n##################################################################################\nMessage 1\n")
        #payload = "0|10;1|033250;2|033251;3|033252;4|033253"
        print(payload)
        lwpubsub_message = payload.split(";")

        #Poll frequency:
        poll_frequency = int(lwpubsub_message[0][lwpubsub_message[0].find('|')+1:])

        #Number of sensors
        print("Number of sensors: ", len(lwpubsub_message)-1)

        lwml4aiot_sensors = pd.DataFrame(lwpubsub_message[1:], columns=['Sensors']) #descarto poll frequency

        lwml4aiot_sensors[['payload_iID','SensorList']] = lwml4aiot_sensors.Sensors.str.split("|",expand=True)

        lwml4aiot_sensors = lwml4aiot_sensors[[lwml4aiot_sensors.columns[-1]]]

        lwml4aiot_sensors['objectID'] = lwml4aiot_sensors['SensorList'].str.strip().str[:-1] #objectID
        lwml4aiot_sensors['instanceID'] = lwml4aiot_sensors['SensorList'].str.strip().str[-1] #instanceID

        #lwml4aiot_sensors = lwml4aiot_sensors[['objectID', 'instanceID']]

        lwml4aiot_parameters_number = lwml4aiot_sensors['instanceID'].count()

        print("\nPoll frequency: ", poll_frequency)
        print("\nSensors:\n", lwml4aiot_sensors)
        print("\nNumber of parameters: ", lwml4aiot_parameters_number)

        #necessário quando fica trocando os sensores utilizados, depois da mensagem 1, só fazer novas previsões depois da mensagem 3
        make_decision = None



    elif (oID == '32002'): #Message2
        print("\n##################################################################################\nMessage 2\n")
        global lwml4aiot_action

        #payload = "0|33060;1|1"
        print(payload)
        lwpubsub_message = payload.split(";")

        lwml4aiot_action['objectID'] = int(lwpubsub_message[0][lwpubsub_message[0].find('|')+1:-1])
        lwml4aiot_action['instanceID'] = int(lwpubsub_message[0][-1])
        lwml4aiot_action['action'] = int(lwpubsub_message[1][-1])

        print("\nAction: \n", lwml4aiot_action)


    else:
        print("\nError - LWML4AIoT message not supported")

    return


def modelConfiguration(payload):
    global SupportedModels
    #global lwml4aiot_model
    global make_decision
    global lwml4aiot_sensors
    global lwml4aiot_algorithm
    global lwml4aiot_coef
    global lwml4aiot_intercept

    print("\n##################################################################################\nMessage 3\n")

    #payload = "321050|5;1|21.2#29.54#5.8#614;2|23.75#33.01#15.21#596;3|22.13#51.49#7.34#642;4|95.63#22.23#49.28#650;5|17.74#31.78#11.52#1346"
    #payload = "321020|5;1|15.778;2|0.837;3|0.450;4|8.068;5|0.274"
    #payload = "321050|6;1|120.66#241.24#2.09#42.38;2|49.10#103.33#0.81#37.83;3|22.94#52.65#0.53#26.67;4|76.55#157.54#1.11#41.84;5|316.86#494.36#2.74#57.87;6|197.82#344.04#2.69#48.90"
    #payload = "321030|6;1|-0.15#-0.13#-2.23#-0.10;2|-0.04#0.02#0.16#0.02;3|0.05#0.04#0.83#0.03;4|-0.09#-0.04#-0.90#-0.01;5|0.13#0.06#1.00#0.04;6|0.10#0.05#1.14#0.03"

    #Logistic Regression multiclass
    #payload = "321030|6;1|23.5627#7.6193#-5.4655#16.7481#-26.6631#-15.8016;2|-0.1485#-0.1310#-2.2277#-0.1036;3|-0.0438#0.0187#0.1571#0.0225;4|0.0465#0.0413#0.8281#0.0299;5|-0.0912#-0.0396#-0.9033#-0.0121;6|0.1332#0.0602#1.0023#0.0351;7|0.1038#0.0502#1.1436#0.0282"
    #payload = "321030|02;1|10.8151#1.4182#-12.2333;2|-0.0636#-0.0477#-1.0038#-0.0266;3|-0.0156#0.0121#0.1495#0.0092;4|0.0792#0.0356#0.8543#0.0174"


    #Logistic Regression binary
    #payload = "321030|1;1|-13.1231;2|0.0349#0.0122#-0.0998#0.0088"


    #Multiple Linear Regression
    #payload = "321020|250;1|16.0126;2|0.8444#0.4488#8.0331#0.2608"

    #Kmeans
    #payload = "321060|02;1|120.66#241.24#2.09#42.38;2|49.10#103.33#0.81#37.83;3|22.94#52.65#0.53#26.67"

    print(payload)


    lwpubsub_message = payload.split(";")

    make_decision = int(lwpubsub_message[0][lwpubsub_message[0].find('|')+1:])

    lwml4aiot_algorithm = int(lwpubsub_message[0][:lwpubsub_message[0].find('|')-1])

    lwml4aiot_message_received = pd.DataFrame(lwpubsub_message[1:], columns=['instanceID']) #descarto cluster que toma ação

    lwml4aiot_message_received[['instanceID','Parameters']] = lwml4aiot_message_received.instanceID.str.split("|",expand=True)


    if (lwml4aiot_algorithm not in SupportedModels['objectID'].values):
        print("\nError - ML Model not available")
        return
    elif (SupportedModels[SupportedModels['objectID'] == lwml4aiot_algorithm]['Class'].values == 'Regression'):
        print("\nReceived Regression Model")
        #Se é regressão, o instanceID 1 traz o(s) beta(s)0 e o instanceID2 ... instanceID9 traz os betas1 a 8
        #Regressão Logística é regressão e não classificação

        #Pego o número de colunas (que são os pesos nos quais os sensores serão "multiplicados", se estiver em número diferente,
        #o número de sensores é diferente do número de pesos, houve erro em alguma mensagem: ou na 1 ou nessa)
        lwml4aiot_received_parameters = int(lwml4aiot_message_received['Parameters'][1:].str.split("#",expand=True).shape[1])
        if lwml4aiot_received_parameters != lwml4aiot_parameters_number:
            print("Incorrect number of parameters in comparison with the Message 1")
            return

        lwml4aiot_intercept = lwml4aiot_message_received['Parameters'][0:1].str.split("#",expand=True)
        lwml4aiot_coef = lwml4aiot_message_received['Parameters'][1:].str.split("#",expand=True)

        columns = lwml4aiot_sensors['SensorList'].values.tolist()

        lwml4aiot_coef.columns = columns

    else: #Classification or Grouping
        print("\nClassification or Grouping")
        #Get the number of parameters

        lwml4aiot_received_parameters = int(lwml4aiot_message_received.Parameters.str.count("#").mean()+1)
        print("Classification / Group - Number of parameters: ", lwml4aiot_received_parameters)

        if lwml4aiot_received_parameters != lwml4aiot_parameters_number:
            print("Incorrect number of parameters in comparison with the Message 1")
            return

        lwml4aiot_coef = lwml4aiot_message_received.Parameters.str.split("#",expand=True)
        lwml4aiot_coef.columns = lwml4aiot_sensors['SensorList'].values.tolist()



    #print("\n ML Algorithm: ", lwml4aiot_algorithm, SupportedModels[SupportedModels['objectID'] == lwml4aiot_algorithm]['ModelName'].values[0])
    #print("\nML Parameters:\n", lwml4aiot_coef)
    print("\nDecision making: ", make_decision)
    #lwml4aiot_model = lwml4aiot_algorithm #Não serve pra nada essa lwml4aiot_model
    #lwml4aiot_ml(lwml4aiot_algorithm, lwml4aiot_coef)







def lwml4aiot_ml(model, coef, intercept=None):

    # model = lwml4aiot_algorithm
    # coef = lwml4aiot_coef.copy()
    # intercept = lwml4aiot_intercept.copy()

    global lwml4aiot_sensors
    print("\n\n\n-----lwml4aiot_ml----- function")
    print("\nModel: ", model, SupportedModels[SupportedModels['objectID'] == model]['ModelName'].values[0])
    print("\nIntercept:\n", intercept)
    print("\nCoefficient(s): \n", coef)


    #Reading measurements to get new observations:
    #lwml4aiot_sensors['SensorList'].shape[0]

    new_observation = np.zeros(lwml4aiot_sensors['SensorList'].shape[0])

    for index, row in lwml4aiot_sensors['SensorList'].items():
        #print(index)
        #print(row)
        #row = str(33250)
        new_observation[index] = globals()["read_"+str(row)]()


    new_observation = np.array([new_observation])
    print("---------------\n\n\n-----New observation: ", new_observation)


    if (lwml4aiot_algorithm == 32101): #SimpleLinearRegression
        print("Simple Linear Regression\n\n")

    elif (lwml4aiot_algorithm == 32102): #MultipleLinearRegression
        print("Multiple Linear Regression\n\n")
        #Criar o código para fazer a multiplicação dos valores dos sensores com os valores dos parâmetros e somar o intercept
        linear_regression_predict(new_observation, coef, intercept)






    elif (lwml4aiot_algorithm == 32103): #LogisticRegression
        print("Logistic Regression\n\n")
        # Logistic Regression
        logistic_regression_predict(new_observation, coef, intercept)





    elif (lwml4aiot_algorithm == 32104): #ArtificialNeuralNetwork
        print("Artificial Neural Network")





    elif (lwml4aiot_algorithm == 32106): #k-means
        print("k-means")
        #K-means
        print("------------------k-means -> make prediction\n\n")
        #new_observation = np.array([[30.39, 89.86, 0.86, 42.86]])
        kmeans_predict(new_observation, coef)


    else:
        print("Error - ML Model not available")
        return












###############################################################################
###############################################################################



def linear_regression_predict(measurements, weights, intercept):
    #Linear Regression
    #measurements = np.array([[30.39, 89.86, 0.86, 42.86]]) #1
    #measurements = np.array([[270, 350, 3.8, 6.23]])
    #weights = lwml4aiot_coef.copy()
    #intercept = lwml4aiot_intercept.copy()

    intercept = intercept.astype('float64')
    weights = weights.astype('float64')


    result = (intercept + np.dot(measurements, weights.T))

    print("Linear Regression Result: ", result)

    return(result)







def logistic_regression_predict(measurements, weights, intercept):
    #Logistic Regression
    #measurements = np.array([[30.39, 89.86, 0.86, 42.86]]) #1
    #measurements = np.array([[270, 350, 3.8, 6.23]])
    #measurements = np.array([[26.53, 21.78, 1.4, 2.67]])
    #weights = lwml4aiot_coef.copy()
    #intercept = lwml4aiot_intercept.copy()
    #measurements = new_observation.copy()

    intercept = intercept.astype('float64')
    weights = weights.astype('float64')

    intercept_np = np.array(intercept)
    weights_np = np.array(weights)

    print("Intercept:", intercept_np)
    print("Weights: \n", weights_np)

    #Ver se é uma decisão binária ou multiclasse:
    # if (intercept.shape[1] == 1): #binária
    #     prob = 1 / (1 + np.exp(- (intercept + np.dot(measurements, weights.T))))
    #     result = 0 if prob.values < 0.5 else 1

    #Errado!
    #prob = 1 / (1 + np.exp(- (intercept + np.dot(measurements, weights.T))))

    #Softmax:
    z = measurements@weights_np.T + intercept_np
    prob = np.exp(z) / np.sum(np.exp(z))

    #prob = prob.values

    if (intercept.shape[1] == 1): #binária
        class_index = 0 if prob < 0.5 else 1
    else: #multiclasse
        i = 0
        class_index = 0
        for i in range(prob.T.shape[0]):
            if (prob.T[i] >= prob.T[class_index]):
                class_index = i


    print("Prob: ", prob)
    print("Logistic Regression result - Class: ", class_index)

    return(class_index)






def kmeans_predict(measurements, weights):

    #K-means - India AQI
    #measurements = np.array([[30.39, 89.86, 0.86, 42.86]]) #1
    #measurements = np.array([[311.67, 25.63, 0.6, 4.58]]) #3
    #weights = lwml4aiot_coef.copy()
    weights = weights.astype('float64')
    weights = np.array(weights)

    euclid = np.zeros([weights.shape[0],1])

    result = weights.copy()

    measurements - weights

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            result[i][j] = measurements[0][j] - result[i][j]
            euclid[i] += (result[i][j]**2)
        euclid[i] = sqrt(euclid[i])

    k = 0
    for i in range(euclid.shape[0]):
        if (euclid[i] < euclid[k]):
            k = i

    print("K: ", k)

    return(k)























###############################################################################
#Payload functions

def parse_payload(message):
    global algorithmarg
    global client

    print(" - Decoded received message: ", hexlify(message).decode('utf-8'))
    message = hexlify(message).decode('utf-8')


    if (lwpubsub_sec_level == 1): #confidentiality only

        cryptomode = message[0:2]
        iv = unhexlify(message[2:34])
        encrypted_message = unhexlify(message[34:])

        if (cryptomode == '01'):
            algorithm = 'CBC'
            key = unhexlify('4e6f726973504144506f6c6955535021')
        elif (cryptomode == '02'):
            algorithm = 'CBC'
            key = unhexlify('4e6f7269734a756e696f725041444c5349506f6c69555350')
        elif (cryptomode == '03'):
            algorithm = 'CBC'
            key = unhexlify('4e6f7269734a756e696f725041444c5349506f6c697465636e69636155535021')
        elif (cryptomode == '11'):
            algorithm = 'CTR'
            key = unhexlify('4e6f726973504144506f6c6955535021')
        elif (cryptomode == '12'):
            algorithm = 'CTR'
            key = unhexlify('4e6f7269734a756e696f725041444c5349506f6c69555350')
        elif (cryptomode == '13'):
            algorithm = 'CTR'
            key = unhexlify('4e6f7269734a756e696f725041444c5349506f6c697465636e69636155535021')

        if (algorithm == 'CBC'):
            plaintext = aes_cbc_decrypt(encrypted_message, key, iv)
        elif (algorithm == 'CTR'):
            plaintext = aes_ctr_decrypt(encrypted_message, key, iv)

    elif (lwpubsub_sec_level == 2): #CIA, AES-128-CCM-8
        received_nonce = unhexlify(message[0:26])
        received_auth_tag = unhexlify(message[26:42])
        encrypted_message = unhexlify(message[42:])

        #print("\n\nreceived nonce:", hexlify(received_nonce).decode('utf-8'))
        #print("\n\nreceived auth tag:", hexlify(received_auth_tag).decode('utf-8'))
        #print("\n\nreceived message:", hexlify(encrypted_message).decode('utf-8'))

        key = unhexlify('4e6f726973504144506f6c6955535021')

        plaintext = aes_ccm_8_decrypt(key, encrypted_message, received_nonce, received_auth_tag)


    print("\n[LWPubSub] - plaintext after decryption: ", plaintext)

    #Command received from the cloud platform
    lwpubsub_parameter = bytearray.fromhex(hexlify(plaintext).decode('utf-8')).decode()[0:7]
    lwpubsub_msg_received = bytearray.fromhex(hexlify(plaintext).decode('utf-8')).decode()


    #Find | to get what is before: objectID||instanceID
    #lwpubsub_parameter = '320010|32105'
    #lwpubsub_parameter = '33110|1'
    #lwpubsub_parameter = '32105|0'
    #lwpubsub_parameter.find('|')

    #object_id = lwpubsub_parameter[0:4]
    #instance_id = lwpubsub_parameter[4]
    #value = lwpubsub_parameter[6]

    #lwpubsub_parameter = '321060|02;1|120.66#241.24#2.09#42.38;2|49.10#103.33#0.81#37.83;3|22.94#52.65#0.53#26.67'

    id_tuple = lwpubsub_parameter[0:lwpubsub_parameter.find('|')]

    object_id = id_tuple[:-1]
    instance_id = id_tuple[-1]
    value = lwpubsub_parameter[lwpubsub_parameter.find('|')+1:]
    print("objectID: ", object_id)
    print("instanceID: ", instance_id)
    print("value: ", value)


    #object_id = '3301'
    if (int(object_id) in IPSOSensors.objectID.values):
        print("\n\nSolicitou medição de sensor")
        measurementRequest(object_id, instance_id, value)
        return

    elif (int(object_id) in LWML4AIoTMessages.objectID.values):
        print("Solicitou usar algoritmo de ML")
        processLWML4AIoTMessage(lwpubsub_msg_received)
        return

    elif (int(object_id) in SupportedModels.objectID.values):
        print("Solicitou usar algoritmo de ML")
        modelConfiguration(lwpubsub_msg_received)
        return


    else:
        print("objectID not supported/available")
        return




###############################################################################
###############################################################################
























###############################################################################
############################################
#MQTT functions and callbacks

# when connecting to mqtt do this;
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected with result code ", rc)
        global Connected                #Use global variable
        Connected = True                #Signal connection

        # Subscribing in on_connect() - if we lose the connection and
        # reconnect then subscriptions will be renewed.
        client.subscribe(sub_topic)
    else:
        print("Connection failed", rc)

    # Connection Return Codes
    # 0: Connection successful
    # 1: Connection refused – incorrect protocol version
    # 2: Connection refused – invalid client identifier
    # 3: Connection refused – server unavailable
    # 4: Connection refused – bad username or password
    # 5: Connection refused – not authorised
    # 6-255: Currently unused.



# when receiving a mqtt message do this;
def on_message(client, userdata, msg):
    print('\n[LWPubSub] ########### Message received ###########')
    print(" - MQTT Message received, parsing...")
    print(" - message topic: ", msg.topic)
    print(" - message received: " , msg.payload)
    parse_payload(msg.payload)

# when publishing
def on_publish(mosq, obj, mid):
    print("mid: " + str(mid))

# log for debug purposes
def on_log(client, userdata, level, buf):
    print("log: ",buf)

# LWPubSub encrypted MQTT payload construction, and after, do publish:
def lwpubsub_publish(object_id):
    global client
    global pub_topic
    global algorithmarg
    global lwpubsub_key
    global cryptomode


    if object_id == '03303':
        lwpubsub_payload = read_temp()
    elif object_id == '03304':
        lwpubsub_payload = read_humidity()

    print('\n[LWPubSub] ########### Publish start ###########')
    print('[LWPubSub] - LWPubSub payload before encryption: ', lwpubsub_payload)
    print('     - Key: ', hexlify(lwpubsub_key).decode('utf-8'))


    if (lwpubsub_sec_level == 1): #confidentiality only
        lwpubsub_payload = lwpubsub_payload.ljust(16, '0')
        if (algorithmarg == 'CBC'):
            init_v, lwpubsub_encr_payload = aes_cbc_encrypt(lwpubsub_payload, lwpubsub_key)
        elif (algorithmarg == 'CTR'):
            init_v, lwpubsub_encr_payload = aes_ctr_encrypt(lwpubsub_payload, lwpubsub_key)
        lwpubsub_encr_payload = hexlify(cryptomode).decode('utf-8') + hexlify(init_v).decode('utf-8') + hexlify(lwpubsub_encr_payload).decode('utf-8')
        lwpubsub_encr_payload = unhexlify(lwpubsub_encr_payload)
        print('     - IV:', hexlify(init_v).decode('utf-8'))
        print('     - Algorithm: ', hexlify(cryptomode).decode('utf-8'))


    elif (lwpubsub_sec_level == 2): #CIA, AES-128-CCM-8
        #return nonce, ciphertext, auth_tag
        generated_nonce, lwpubsub_encr_payload, lwpubsub_auth_tag = aes_ccm_8_encrypt(lwpubsub_payload, lwpubsub_key)
        lwpubsub_encr_payload = hexlify(generated_nonce).decode('utf-8') + hexlify(lwpubsub_auth_tag).decode('utf-8') + hexlify(lwpubsub_encr_payload).decode('utf-8')
        lwpubsub_encr_payload = unhexlify(lwpubsub_encr_payload)
        print('     - Nonce:', hexlify(generated_nonce).decode('utf-8'))
        print('     - Algorithm: AES-128-CCM-8')


    print('[LWPubSub] - LWPubSub payload after encryption: ', hexlify(lwpubsub_encr_payload).decode('utf-8'))

    client.publish(pub_topic, lwpubsub_encr_payload)

    print('[LWPubSub] ########### Publish finish ###########\n')



###############################################################################
###############################################################################
































def main():
    global algorithmarg
    global keysizearg
    global brokeraddarg
    global lwpubsub_sec_level
    global lwpubsub_key
    global lwml4aiot_algorithm
    global lwml4aiot_coef

    ###############################################################
    #Starting the program:
    if (len(sys.argv) < 4):
        print('Provide at least 3 arguments: <algorithm (CCM, CBC or CTR)>, <keysize (128, 192, 256) - for CCM only 128>, Broker address IP <X.X.X.X>')
        exit(0)

    algorithmarg=sys.argv[1]
    keysizearg=sys.argv[2]
    brokeraddarg=sys.argv[3]

    print("keysize: ", keysizearg)

    if (algorithmarg not in ['CCM', 'CBC', 'CTR']):
        print('Wrong algorithm, exiting...')
        exit(0)
    elif (algorithmarg == 'CCM') and (keysizearg != '128'):
        print('CCM mode only with 128-bit key, exiting...')
        exit(0)

    if (keysizearg not in ['128', '192', '256']):
        print('Wrong keysize, exiting...')
        exit(0)



    if (algorithmarg == 'CBC') and (keysizearg == '128'):
        cryptomode = unhexlify('01')
        lwpubsub_sec_level = 1
    elif (algorithmarg == 'CBC' and keysizearg == '192'):
        cryptomode = unhexlify('02')
        lwpubsub_sec_level = 1
    elif (algorithmarg == 'CBC' and keysizearg == '256'):
        cryptomode = unhexlify('03')
        lwpubsub_sec_level = 1
    elif (algorithmarg == 'CTR' and keysizearg == '128'):
        cryptomode = unhexlify('11')
        lwpubsub_sec_level = 1
    elif (algorithmarg == 'CTR' and keysizearg == '192'):
        cryptomode = unhexlify('12')
        lwpubsub_sec_level = 1
    elif (algorithmarg == 'CTR' and keysizearg == '256'):
        cryptomode = unhexlify('13')
        lwpubsub_sec_level = 1
    elif (algorithmarg == 'CCM'):
        cryptomode = 'AES-128-CCM-8'
        lwpubsub_sec_level = 2

    print("Cryptomode: ", cryptomode) #if algorithmarg != 'CCM' else print("Cryptomode: AES-128-CCM-8")

    if (keysizearg == '128'):
        lwpubsub_key = unhexlify('4e6f726973504144506f6c6955535021')
    elif (keysizearg == '192'):
        lwpubsub_key = unhexlify('4e6f7269734a756e696f725041444c5349506f6c69555350')
    elif (keysizearg == '256'):
        lwpubsub_key = unhexlify('4e6f7269734a756e696f725041444c5349506f6c697465636e69636155535021')



    # If it's ok, start MQTT client
    ###############################################################
    #Starting the client
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_log=on_log
    client.connect(brokeraddarg, 1883, 60)
    client.loop_start()        #start the loop
    while Connected != True:    #Wait for connection
        time.sleep(0.1)

    ################################################################
    #After started, run forever sending measurements until CTRL+C
    try:
        while True:
            #Execute ML algorithm if the message was received
            time.sleep(poll_frequency)
            if not (make_decision is None):
                start_time = time.time()
                lwml4aiot_ml(lwml4aiot_algorithm, lwml4aiot_coef, lwml4aiot_intercept)
                print("--- Spent %s seconds for make decision using  ---" % (time.time() - start_time), (SupportedModels[SupportedModels['objectID'] == lwml4aiot_algorithm]['ModelName'].values[0]))
            #lwpubsub_publish('3303')
            start_time = time.time()
            #lwpubsub_publish(algorithmarg, '03303', lwpubsub_key, cryptomode)
            lwpubsub_publish('03303')
            print("--- Spent %s seconds for publishing ---" % (time.time() - start_time))


    except KeyboardInterrupt:
        # disconnect
        print("Stopping the client ...")
        client.loop_stop()
        client.disconnect()





if __name__ == "__main__":
    algorithmarg = None
    keysizearg = None
    brokeraddarg = None
    lwpubsub_sec_level = None
    lwpubsub_key = unhexlify('4e6f726973504144506f6c6955535021')
    #lwml4aiot_model = None
    make_decision = None
    lwml4aiot_sensors = None
    lwml4aiot_parameters_number = None
    lwml4aiot_algorithm = None
    lwml4aiot_coef = None
    lwml4aiot_intercept = None
    lwml4aiot_action = pd.DataFrame({
        'objectID': [np.nan],
        'instanceID': [np.nan],
        'action': [np.nan]
    })
    count_pm25 = 0
    count_pm10 = 0
    count_co = 0
    count_dummy = 0
    count_dummy_motor = 0


    ############################################
    # Construct Client ID (previously provisioned on the cloud platform)
    if platform.system() == 'Windows':
        client_id = get_mac_address(interface="WiFi")
    elif platform.system() == 'Linux':
        client_id = get_mac_address(interface="wlan0")

    client_id = client_id.replace(":", "")

    client_id = "0012" + client_id[4:12]

    client = mqtt.Client(client_id)

    ############################################
    #Publish and subscribe topics
    sub_topic = "/99/" + client_id + "/cmd"    # receive messages on this topic
    pub_topic = "/99/" + client_id       # send messages to this topic

    ############################################
    #Poll frequency (seconds)
    poll_frequency = 30

    ############################################
    #Control connection
    Connected = False

    green_led = 0

    red_led = 0

    blue_led = 0

    if ( (platform.system() == 'Linux') and ( (platform.machine() == 'armv7l') or (platform.machine() == 'armv6l') ) ):
        # green led on GPIO 17
        # red led on GPIO 27
        # green led: 3311 1 (objectID instanceID)
        # red led:   3311 0 (objectID instanceID)
        gpio_green = LED(27)
        gpio_red = LED(17)
        gpio_blue = LED(22)

    main()
