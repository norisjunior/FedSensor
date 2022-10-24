import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)




def kmeans_message(decision, var_weights):
    # decision=model_target
    # var_weights=model.cluster_centers_

    #lin = clf.coef_.shape[0]
    col = var_weights.shape[1]
    #decision = 5

    #message = "0" + str(decision) + ";1|"
    message = str(decision).zfill(3) + ";1|"
    previous = 0
    for idx, i in np.ndenumerate(var_weights):
        if idx[0] != previous:
            message += ";" + str(idx[0]+1) + "|"
            previous += 1
        message += str("{:.4f}".format(i))
        if ( (idx[0] == previous) and (idx[1] != col-1) ):
            message += "#"
    return message
    




def logreg_message(decision, var_intercept, var_weights):
    
    #var_intercept = clf.intercept_
    #var_weights = clf.coef_
    #decision = 5
    
    #message = "0" + str(decision) + ";"
    message = str(decision).zfill(3) + ";"

    
    #Espera-se mais de um coeficiente
    #Mas não se sabe o número de intercepts
    #Verificar o número de intercepts para saber se é regressão linear: binária ou multiclasse
    if var_intercept.shape[0] == 1: # Regressão Logística binária
        logging.debug("Regressão Logística binária")
        message += "1|" + str("{:.4f}".format(var_intercept[0]))
        
    else:
        logging.debug("Regressão Logística multiclasse")
        col = var_intercept.shape[0]

        message += "1|"
        for idx, i in np.ndenumerate(var_intercept):
            message += str("{:.4f}".format(i))
            if idx[0] != col-1: message += "#" #is not the last
    
    
    col = var_weights.shape[1]

    message += ";2|"
    previous = 0
    for idx, i in np.ndenumerate(var_weights):
        if idx[0] != previous:
            message += ";" + str(idx[0]+2) + "|"
            previous += 1
        message += str("{:.4f}".format(i))
        if ( (idx[0] == previous) and (idx[1] != col-1) ):
            message += "#" #is not the last
    return message
    




def linreg_message(decision, var_intercept, var_weights):
    
    # var_intercept = clf.intercept_
    # var_weights = clf.coef_
    #decision = 250
    
    #message = "0" + str(decision) + ";"
    message = str(decision).zfill(3) + ";"

    
    #Espera-se mais de um coeficiente
    #Mas não se sabe o número de intercepts
    #Verificar o número de intercepts para saber se é regressão linear: binária ou multiclasse
    
    #beta0
    message += "1|" + str("{:.4f}".format(var_intercept))

    
    if var_weights.shape[0] == 1: # Regressão Linear Simples
        logging.debug("Regressão Linear Simples")
        
    else:
        logging.debug("Regressão Linear Múltipla")
    
    
    message += ";2|"
    last = var_weights.shape[0]-1
    for idx, i in np.ndenumerate(var_weights):
        message += str("{:.4f}".format(i))
        if ( (idx[0] != last) ):
            message += "#" #is not the last    
    
    return message
    

