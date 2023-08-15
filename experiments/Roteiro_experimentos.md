# **Roteiro dos experimentos**

Para executar os experimentos, primeiro clonar o repositório https://github.com/norisjunior/FedSensor.git.
Nos experimentos da Tese primeiro foi criado o diretório "/IoTArchitecture/" e dentro dele clonado o FedSensor (portanto, estrutura dos experimentos no diretório "/IoTArchitecture/FedSensor").

Passos executados (_importante executar o submodule update, pois o contiki-ng é um submodulo do FedSensor e o próprio contiki-ng tem outros submódulos necessários para compilar o código nos dispositivos IoT avaliados na Tese_):
  ```bash
  mkdir IoTArchitecture
  cd IoTArchitecture
  git clone https://github.com/norisjunior/FedSensor.git
  cd FedSensor
  git fetch origin
  git reset --hard origin/master
  git submodule update --force
  cd contiki-ng-4.7
  git submodule update --init --recursive --force
  ```

> **_NOTA:_** Pode ser usado o **WSL 2**, mas o WSL 2 não suporta transferir o firmware para os dispositivos. Caso a intenção seja implantar o firmware nos dispositivos, isso tem que ser feito a partir de uma máquina virtual no VirtualBox ou VMWare. Os testes da aplicação em python e o modo _native_ do contiki-ng funcionam no WSL 2.


###### Requisitos gerais:
docker

###### Requisitos (python3):
flwr                      0.17.0\
getmac                    0.8.3\
keras                     2.6.0\
keras-preprocessing       1.1.2\
numpy                     1.19.5
paho-MQTT                 1.6.1\
pandas                    1.2.3\
pycryptodome              3.15.0\
pyod                      1.0.0\
scikit-learn              1.0.2\
scipy                     1.7.3\
tensorflow                2.6.3\
tensorflow-base           2.6.3\
tensorflow-cpu            2.6.3

###### Requisitos (Contiki-NG):
Pacotes (ubuntu 20.04):\
build-essential doxygen git curl python-serial srecord rlwrap

---

## Demonstração de um experimento para a realização do treinamento federado e transmissão do modelo de ML global gerado para um dispositivo simulado
Demonstração do FedSensor com um dispositivo IoT ultra-restrito executado no modo "Native" do Contiki-NG - execução do código no Linux, sem a necessidade de passar o firmware para um dispositivo (apenas para demonstração do FedSensor, há funções para coletar valores aleatórios dos sensores ativados).

Nessa demonstração todos os serviços são executados localmente: o dispositivo IoT ultra-restrito no Contiki-NG, o MQTT Broker (_mosquitto_), o LWPubSub IoT Agent, o Orion e o MongoDB.


##### Inicialização do ambiente de demonstração:
- Abrir 7 terminais (1 para a plataforma Edge, 1 para o FedSensor Manager e mais 5 terminais (cada um será um FedSensor Participant diferente)
- Abrir um oitavo terminal para executar o dispositivo IoT ultra-restrito no modo "Native" do Contiki-NG
- No terminal 1 (Edge Server) será: (1) inicializada a plataforma Edge do FedSensor, baseada no FIWARE, e; (2) provisionados os dispositivos IoT utilizados.
  - Inicialização da plataforma Edge:
    - ``` cd /IoTArchitecture/FedSensor/EdgeServer ```
    - ``` ./edge-platform-services create ```
    - ``` ./edge-platform-services start ```
      - Ao executar o comando ``` docker ps ```, deve ser exibido uma saída similar a:
        ```sh
        CONTAINER ID   IMAGE                                     COMMAND                  CREATED          STATUS                             PORTS                                            NAMES
        863fd064b48c   fiware/orion:2.3.0                        "/usr/bin/contextBro…"   16 seconds ago   Up 15 seconds (health: starting)   0.0.0.0:1026->1026/tcp                           fiware-orion
        f073ffdc2155   norisjunior/lwpubsub-iotagent-ccm:v2.10   "docker/entrypoint.s…"   16 seconds ago   Up 15 seconds (health: starting)   4061/tcp, 0.0.0.0:4041->4041/tcp, 7896/tcp       fiware-iot-agent
        1b925975d08f   eclipse-mosquitto:1.5.8                   "/docker-entrypoint.…"   18 seconds ago   Up 16 seconds                      0.0.0.0:1883->1883/tcp, 0.0.0.0:9001->9001/tcp   mosquitto
        ae96f55129dc   mongo:3.6                                 "docker-entrypoint.s…"   18 seconds ago   Up 16 seconds                      0.0.0.0:27017->27017/tcp                         db-mongo ```
        ```
  - Provisionamento dos dispositivos:
    - ``` ./provision_lwaiot_devices.sh ```
      - Conferir que o dispositivo "Native" foi provisionado:
        - ``` curl -X GET 'http://localhost:1026/v2/entities/010203060708?type=Native' -H 'fiware-service: school' -H 'fiware-servicepath: /fd00' | python -m json.tool ```
        - Deve ser exibida uma lista contendo todos os sensores provisionados e os modelos de ML (321020 a 321069)

- No terminal 8, executar o dispositivo IoT ultra-restrito no modo Native:
    - ``` cd /IoTArchitecture/FedSensor/contiki-ng-4.7/lwiotms/lwpubsub ```
    - ``` ./make_app.sh NATIVE ```
    - ``` sudo ./lwpubsub-fedsensor-ml_and_msg.native ```
    - Devem ser exibidos os *logs* do contiki, de maneira similar a:
      ```bash
      [INFO: Main      ] Starting Contiki-NG-release/v4.7-136-g7d0211780
      [INFO: Main      ] - Routing: RPL Lite
      [INFO: Main      ] - Net: tun6
      [INFO: Main      ] - MAC: nullmac
      [INFO: Main      ] - 802.15.4 PANID: 0xabcd
      [INFO: Main      ] - 802.15.4 Default channel: 26
      [INFO: Main      ] Node ID: 1800
      [INFO: Main      ] Link-layer address: 0102.0304.0506.0708
      [INFO: Main      ] Tentative link-local IPv6 address: fe80::302:304:506:708
      [INFO: Native    ] Added global IPv6 address fd00::302:304:506:708

      LWPubSub-FedSensor-LWAIoT - MQTT Client Process
      [INFO: LWPubSub  ] Crypto Algorithm: AES-CCM-8
      [INFO: LWPubSub  ] Poll frequency: 10 secs.
      [INFO: LWPubSub  ] TSCH Schedule: Minimal
      [INFO: LWPubSub  ] Sub Topic: /99/010203060708/cmd
      [INFO: LWPubSub  ] Pub Topic: /99/010203060708
      ```
    - O dispositivo fica aguardando o recebimento das mensagens 1 (configuração dos sensores que serão usados no modelo de ML global e o tempo para realização da inferência) e 2 (parâmetros/coeficientes do modelo de ML global e o alvo: classe, grupo, ou valor máximo aceitável).

- Nos terminal 2 e nos terminais 3 a 7, preparar para a execução do FedSensor Manager (terminal 2) e FedSensor Participant (terminais 3 a 7):
  - ``` cd /IoTArchitecture/FedSensor/FL/fedsensor_framework ```
  - Ativar o virtualenv. Exemplo usando conda:
    - ``` conda activate spyder-tensorflow-env ```, onde *spyder-tensorflow-env* é o nome do virtualenv
    - **Os demais passos serão apresentados a seguir. Escolha \*um\* dos modelos (regressão linear, regressão logística ou k-means) e depois de executar o Manager e todos os 5 Participantes, vá para o próximo passo.**


#### Experimento para demonstrar a transmissão do **modelo de ML global *regressão logística* com 2 classes no desfecho** no dispositivo IoT ultra-restrito:
Cenário: IQAr usando 4 sensores (PM2.5, PM10, CO, O3)

- No terminal 2 (Manager), executar:
  - ``` python3 manager.py --dataset='AQI' --algorithm='logreg' --sensors='aqi_pm_plus' --result='binary' --n_classes=2 ```


- Nos terminais 3 a 7 (Participant), executar:
  - Terminal 3: ``` python3 participant.py --dataset='AQI' --algorithm='logreg' --sensors='aqi_pm_plus' --result='binary' --n_classes=2 --anomaly_detection='ECOD' --partition=0.31 ```
  - Terminal 4: ``` python3 participant.py --dataset='AQI' --algorithm='logreg' --sensors='aqi_pm_plus' --result='binary' --n_classes=2 --anomaly_detection='ECOD' --partition=0.32 ```
  - Terminal 5: ``` python3 participant.py --dataset='AQI' --algorithm='logreg' --sensors='aqi_pm_plus' --result='binary' --n_classes=2 --anomaly_detection='ECOD' --partition=0.33 ```
  - Terminal 6: ``` python3 participant.py --dataset='AQI' --algorithm='logreg' --sensors='aqi_pm_plus' --result='binary' --n_classes=2 --anomaly_detection='ECOD' --partition=0.34 ```
  - Terminal 7: ``` python3 participant.py --dataset='AQI' --algorithm='logreg' --sensors='aqi_pm_plus' --result='binary' --n_classes=2 --anomaly_detection='ECOD' --partition=0.35 ```
  - O argumento ``` partition ``` tem duas finalidades: (1) indicar o percentual do dataset que será utilizado, e (2) ser um critério de aleatoriedade na seleção das amostras que irão compor o dataset do participante. Portanto, cada participante tem uma porção similar do dataset, mas as amostras utilizadas para treinamento local são diferentes.
  - Caso um dos participantes seja o raspberry pi, é necessário usar o arquivo "participant-pi3b.py" e definir o argumento "--server_address" com o IP do gerenciador. Por exemplo, supondo que o IP do gerenciador seja *192.168.75.155*, o comando seria:
    - ``` python3 participant_pi3b.py --server_address='192.168.75.155' --dataset='AQI' --algorithm='logreg' --sensors='aqi_pm_plus' --result='binary' --n_classes='2' --anomaly_detection='ECOD' --partition=0.15 ```
  - Exemplo de modelo de ML global regressão linear resultante:
    ```bash
    Modelo gerado evaluate():
     [array([[-0.25071678,  0.13121599],
           [ 0.02137789, -0.0277726 ],
           [ 0.6537366 , -0.16414966],
           [-0.36040947, -0.40747505]], dtype=float32),
      array([ 0.01446569, -0.01446569], dtype=float32)]
    ```


> **_NOTA:_** Caso se utilize o WSL2, é necessário: (1) desabilitar o antivírus, ou permitir as portas 8080 (regressão logística), 8081 (kmeans) e 8082 (regressão linear) no firewall, e; (2) realizar o _port forwarding_ dessas portas, para que o raspberry (que é um participante), encontre o gerenciador (que está no WSL2).


---


#### Experimento para demonstrar a transmissão do **modelo de ML global *regressão linear* ** no dispositivo IoT ultra-restrito:
Cenário: IQAr usando 4 sensores (PM2.5, PM10, CO, O3)

- No terminal 2 (Manager), executar:
  - ``` python3 manager.py --dataset='AQI' --algorithm='linreg' --sensors='aqi_pm_plus' --result='binary' --n_classes=2 ```


- Nos terminais 3 a 7 (Participant), executar:
  - Terminal 3: ``` python3 participant.py --dataset='AQI' --algorithm='linreg' --sensors='aqi_pm_plus' --result='regression' --anomaly_detection='IForest' --partition=0.31 ```
  - Terminal 4: ``` python3 participant.py --dataset='AQI' --algorithm='linreg' --sensors='aqi_pm_plus' --result='regression' --anomaly_detection='IForest' --partition=0.32 ```
  - Terminal 5: ``` python3 participant.py --dataset='AQI' --algorithm='linreg' --sensors='aqi_pm_plus' --result='regression' --anomaly_detection='IForest' --partition=0.33 ```
  - Terminal 6: ``` python3 participant.py --dataset='AQI' --algorithm='linreg' --sensors='aqi_pm_plus' --result='regression' --anomaly_detection='IForest' --partition=0.34 ```
  - Terminal 7: ``` python3 participant.py --dataset='AQI' --algorithm='linreg' --sensors='aqi_pm_plus' --result='regression' --anomaly_detection='IForest' --partition=0.35 ```

  - Exemplo de modelo de ML global regressão linear resultante:

    ```bash
    Modelo gerado evaluate():
     [array([
           [1.0614371 ],
           [0.3792903 ],
           [5.4981503 ],
           [0.55774575]], dtype=float32),
           array([5.705202], dtype=float32)]
    ```

#### Experimento para demonstrar a transmissão do **modelo de ML global *k-means* com 2 grupos no desfecho** no dispositivo IoT ultra-restrito:
Cenário: IQAr usando 4 sensores (PM2.5, PM10, CO, O3)

- No terminal 2 (Manager), executar:
  - ``` python3 manager.py --dataset='AQI' --algorithm='kmeans' --sensors='aqi_pm_plus' --result='binary' --n_groups=2 ```


- Nos terminais 3 a 7 (Participant), executar:
  - Terminal 3: ``` python3 participant.py --dataset='AQI' --algorithm='kmeans' --sensors='aqi_pm_plus' --result='binary' --n_groups=2 --anomaly_detection='IForest' --partition=0.31 ```
  - Terminal 4: ``` python3 participant.py --dataset='AQI' --algorithm='kmeans' --sensors='aqi_pm_plus' --result='binary' --n_groups=2 --anomaly_detection='IForest' --partition=0.32 ```
  - Terminal 5: ``` python3 participant.py --dataset='AQI' --algorithm='kmeans' --sensors='aqi_pm_plus' --result='binary' --n_groups=2 --anomaly_detection='IForest' --partition=0.33 ```
  - Terminal 6: ``` python3 participant.py --dataset='AQI' --algorithm='kmeans' --sensors='aqi_pm_plus' --result='binary' --n_groups=2 --anomaly_detection='IForest' --partition=0.34 ```
  - Terminal 7: ``` python3 participant.py --dataset='AQI' --algorithm='kmeans' --sensors='aqi_pm_plus' --result='binary' --n_groups=2 --anomaly_detection='IForest' --partition=0.35 ```

  - Exemplo de modelo de ML global k-means resultante:

    ```bash
    Modelo final main():
     [[ 36.23342223  77.57150834   0.62423439  34.19915968]
     [ 92.14738605 195.26256036   1.07125313  35.90533105]]
    ```

#### Depois de executar o Manager e os 5 Participantes, em cada participante será exibido o modelo gerado (mostrando a seguir o resultado de um modelo regressão logística)
```bash
Modelo final main():

Class 0 - Not severe: 0.015370522625744343
Class 1 - Severe: -0.01537051796913147
```

Os parâmetros a seguir consideram o treinamento usando regressão logística com 2 classes como desfecho do modelo:
  - O participante perguntará qual é a classe alvo (mensagem: "*Enter target class value (int):*"). Digite o número 0 ou 1 (pois o treinamento considerou 2 classes no desfecho).
  - Na sequência informe qual dispositivo deve ser usado para receber o modelo de ML global (mensagem: "*Choose the device (int):*"). O participante informa 5 possíveis dispositivos, sendo que o dispositivo 1 é o "Native". Digite o número 1.
  - Em seguida, informe o endereço do LWPubSub IoT Agent (mensagem: "*Inform the LWPubSub IoT Agent address (string):*"). Digite *localhost*.
  - Por fim informe o tempo em segundos (máximo de 999 segundos) para coleta de medições e realização de inferência (mensagem: "*Inform time interval to gather measurements with 3 digits (int):*"). Digite 5 para aguardar 5 segundos para o dispositivo realizar inferências.
  - Observe o terminal que executa o dispositivo IoT ultra-restrito, pois ele receberá as mensagens e passará a realizar inferências com base nas medições dos sensores. No caso do modo "Native", os valores de PM2.5, PM10, CO e O3 são gerados aleatoriamente.

    ```bash
    [INFO: LWPubSub  ] Received message len: 46
    [INFO: LWPubSub  ] Message AuthCheck: OK
    [INFO: LWPubSub  ] ObjectID: 32001
    [INFO: LWPubSub  ] InstanceID: 0
    ...
    [INFO: LWPubSub  ]  - CommandReceivedLen: 83
    [INFO: LWPubSub  ] -------------- FedSensor - LWPubSub[LWAIoT - Logistic regression] --------------
    [INFO: LWPubSub  ] Number of classes: 2
    [INFO: LWPubSub  ] Vetor de bias: [0.0154] [-0.0154]
    [INFO: LWPubSub  ] Matriz de weights:
    Class [0]: [-0.2898] [0.5888] [-0.1598] [0.2416]
    Class [1]: [-0.1208] [0.5722] [0.6458] [0.2240]
    [INFO: LWPubSub  ] --------------------------------------------------------------------------------
    [INFO: LWPubSub  ] New observation/measurement collected:
    33250: 15.6124
    33251: 5.9582
    33256: 1.8814
    33258: 0.5434
    [INFO: LWPubSub  ] ------------------------- Logistic Regression predict --------------------------
    [INFO: LWPubSub  ] Result of X*w + b: [-1.1702]   [2.8446]
    [INFO: LWPubSub  ] logreg_prob[0]: 0.0177
    [INFO: LWPubSub  ] logreg_prob[1]: 0.9823
    [INFO: LWPubSub  ] Class: 1
    [INFO: LWPubSub  ] Class != ACTION; class = 1, action = 0
    [INFO: LWPubSub  ] --------------------------------------------------------------------------------
    ```

---

**Vídeo de demonstração do Quick Start**:
![FedSensor - modelo de ML global regressão logística](FedSensor-demonstration.mp4)

---

## Visão geral do FedSensor framewok

Há dois pilares para a instanciar o FedSensor framework: a **Aprendizagem Federada** (Federated Learning) e os **dispositivos**.

Os dispositivos são gerenciados pelos Participantes (os Edge Servers), que por sua vez utilizam como base a plataforma [FIWARE](https://fiware.org). Portanto, primeiro devem-se inicializar os Edge Servers e seus respectivos dispositivos (a seguir apresentam-se os dispositivos disponíveis e testados).

***Depois de inicializados os Edge Servers e os dispositivos***, pode ser realizada a Aprendizagem Federada.

A seguir apresentam-se como instanciar o FedSensor: primeiro com relação aos dispositivos e a plataforma FIWARE e, depois, realizar a Aprendizagem Federada nos dispositivos provisionados nos respectivos Edge Servers. Os dispositivos receberão o modelo global gerado e realizarão a tomada de decisão considerando a mensagem MQTT recebida do seu respectivo Edge Server.

---

## Inicializando o Edge Server e os seus respectivos dispositivos

É necessário inicializar o Edge Server com a plataforma FIWARE, IoT Agent, Orion, MongoDB e MQTT Broker.
Essa estrutura pode ser inicializada usando um script gerado a partir do modelo apresentado pela própria FIWARE Foundation.

O script gerado nesta Tese está em ``` EdgeSever/docker-compose.yml ```, devendo ser usado o comando ``` edge-platform-services ```.

Primeiro devem ser baixados os containeres:
```bash
cd /IoTArchitecture/FedSensor/EdgeSever
./edge-platform-services create
./edge-platform-services start
```

Caso queira observar os logs do LWPubSub IoTAgent para MQTT desenvolvido para esta Tese, abrir em outro terminal no mesmo diretório apresentado anteriormente e executar o script ``` iotagent_view_logs.sh ```.

Uma vez inicializado o Edge Server, devem ser provisionados os dispositivos usando o script ``` provision_lwaiot_devices.sh ``` (podendo ser no mesmo terminal que inicializou a plataforma FIWARE).

O script de provisionamento, cria os seguintes dispositivos na plataforma:
* Entity name "010203060708", type "Native"
* Entity name "00124b05257a", type "Sensortag"
* Entity name "00124b4a527d", type "Remote"
* Entity name "00122518499d", type "WindowsWiFi"
* Entity name "0012ebc894cb", type "Raspberry"

É necessário customizar o script ``` provision_lwaiot_devices.sh ``` para considerar o MAC Address dos dispositivos a serem utilizados, pois os MAC Address provisionados são os utilizados na Tese.

Dessa forma é possível receber medições desses dispositivos, bem como enviar comandos para eles, usando tanto o formato apresentado nos artigos do LWPubSub [Ferraz Junior (2021a)](https://jcis.emnuvens.com.br/jcis/article/view/782), [Ferraz Junior (2021b)](https://www.sciencedirect.com/science/article/pii/S1877050921013995) e [Ferraz Junior (2022)](https://journalofcloudcomputing.springeropen.com/articles/10.1186/s13677-022-00278-6) - que enviam medições e recebem comandos da plataforma - quanto usando o formato de transmissão do modelo global para os dispositivos desenvolvido nesta Tese, customizando o LWPubSub.

Uma vez com a plataforma inicializada e os dispositivos provisionados, podem-se executar os experimentos.



---


## Experimentos relativos aos dispositivos IoT restritos

* ***Dispositivos IoT***
    * Zolertia Remote revb cc2538
    * Texas Instruments Sensortag cc2650


* ***6LBR***
    * Zolertia Firefly cc2538
    * Texas Instruments Launchpad cc2650

#### _Contiki-NG_
* Versão 4.7 (develop da versão 4.8)
* Desenvovimento do LWPubSub customizado para o FedSensor em:
    * ```contiki-ng-4.7/lwiotms/lwpubsub/```


* ***Código-fonte dos dispositivos:*** Para compilar o código usado pelos dispositivos IoT, há 3 opções (_considerando que o contiki-ng está instalado - ou que se usa o docker contiker - e que está no diretório acima indicado_):
    * **Native** (modo para testes, executado no terminal):
        * ``` make lwpubsub-fedsensor-lwaiot TARGET=native ```
    * **Remote revb**:
        * ``` make lwpubsub-fedsensor-lwaiot TARGET=zoul BOARD=remote-revb PORT=/dev/ttyUSB1 MAKE_KEYSIZE=128 MAKE_CRYPTOMODE=CCM MAKE_WITH_ENERGY=1 lwpubsub-fedsensor-lwaiot.upload ```
    * **Sensortag**:
        * ``` make lwpubsub-fedsensor-lwaiot TARGET=cc26x0-cc13x0 BOARD=sensortag/cc2650 MAKE_KEYSIZE=128 MAKE_CRYPTOMODE=CCM MAKE_WITH_ENERGY=1 lwpubsub-fedsensor-lwaiot ```
    * ***Premissas***:
        * O código compilado sempre terá a opção de apresentar o consumo de energia. Caso queira modificar essa opção, deve-se utilizar `MAKE_WITH_ENERGY=0`
        * Não há um modo de operação sem segurança no tráfego
        * A chave utilizada para teste está fixa no código e no IoT Agent: `4e6f726973504144506f6c6955535021`
        * o dispositivo _Remote revb_ deve estar conectado no Linux (ou na máquina virtual), na porta ttyUSB1, uma vez que o código será compilado e o firmware instalado no dispositivo
        * o dispositivo _Sensortag_ não precisa estar conectado, mas o firmware precisa ser instalado no dispositivo usando o UniFlash ou o Flash Programmer 2 da Texas Instruments.
        * informações completas sobre como compilar o código usando o contiki-ng podem ser obtidas na [Wiki](https://github.com/contiki-ng/contiki-ng/wiki) do Contiki-NG.
    * ***Informações adicinoais***:
        * para facilitar a compilação do código, pode-se usar o script `make_app.sh` que está dentro do diretório `contiki-ng-4.7/lwiotms/lwpubsub/`
        * recomenda-se executar o comando `make distclean` antes de executar a compilação do código independentemente da plataforma (native, sentortag, remote, launchpad, firefly).


* ***Código-fonte do 6LBR***: Para compilar o código do 6LBR, podem ser usados o _Firefly_ ou o _Launchpad_ (_tmabém considerando que o contiki-ng está instalado - ou que se usa o docker contiker - e que está no diretório `contiki-ng-4.7/lwiotms/6lbr`_):
    * Pode-se usar o comando `make_6lbr.sh` para compilar o firmware do _Firefly_ ou do _Launchpad_, usando: `make_6lbr.sh FIREFLY` ou `make_6lbr.sh LAUNCHPAD` respectivamente.
    * ***Premissas***:
        * o dispositivo _Firefly_ deve estar conectado no Linux (ou na máquina virtual) na porta ttyUSB0 - o código será compilado no dispositivo.
        * o dispositivo _Launchpad_ segue a mesma premissa apresentada anteriormente para o _Sensortag_


* ***Inicialização do ambiente com dispositivos IoT restritos***
    * Apresenta-se, a seguir, o ambiente usando um _Firefly_ como 6LBR e um _Remote revb_ como dispositivo IoT ultra-restrito.
    * O _Firefly_ deve estar conectado na interface ttyUSB0 e o _Remote revb_ deve estar na interface ttyUSB1.
    * Passos:
        * Em um novo terminal:
        ```bash
        cd /IoTArchitecture/FedSensor/contiki-ng-4.7/lwiotms/6lbr
        ./start_6lbr.sh FIREFLY
        ```

          _Isso inicializará o 6LBR, que aguardará o ingresso de dispositivos IoT na rede 6LoWPAN._
        * Também em um novo terminal:
        ```bash
        cd /IoTArchitecture/FedSensor/contiki-ng-4.7/lwiotms/lwpubsub
        ./view_device_log.sh REMOTE
        ```

          _Isso permitirá observar a saída serial do Remote revb, observando os logs (mensagens enviadas, recebidas, consumo de energia, etc.)._



---

## Experimentos relativos aos dispositivos IoT irrestritos e terminais em Windows e Linux

Há um script em python que pode ser utilizado em dispositivos IoT irrestritos, como o Raspberry Pi (pois o Raspberry utiliza uma distribuição Linux capaz de executar código em python).

Os passos a seguir executados podem ser realizados em um terminal Windows ou Linux.

* ***Execução da aplicação em python***
    * Acessar o diretório `standalone-pydevice`:

    * Executar a aplicação `lwpubsub-lwaiot-pydevice.py`:
    ```bash
    python lwpubsub-lwaiot-pydevice.py CCM 128 localhost
    ```

    * ***Premissas***:
        * Considera-se que o FedSensor está em "/IoTArchitecture/FedSensor/"
        * Uso do algoritmo criptográfico AES-CCM com chave de 128 bits (padrão quando se usa o FedSensor para o tráfego seguro fim-a-fim dos modelos de ML globais transmitidos do EdgeServer para o dispositivo IoT)
        * MQTT Broker executado localmente no EdgeServer (localhost). Também foram realizados experimentos com IPs externos, bastando trocar localhost pelo IP do MQTT Broker.

---

**Informações adicionais sobre os datasets e os argumentos da linha de comando**:
  - Os datasets disponíveis são MOTOR e AQI, mas outros podem ser customizados e utilizados.
  - Os algoritmos disponíveis são: regressão linear (linreg), regressão logística (logreg) e k-means (kmeans)
  - Para o experimento MOTOR, estão disponíveis o uso de 2 (motor_acc) ou 3 (motor_all) sensores.
    - Sensores disponíveis (valores simulados, exceto no Sensortag): 33460: 'pctid', 33130: 'x', 33131: 'y', 33132: 'z':
      - motor_acc: 33130, 33131, 33132
      - motor_all: 33130, 33131, 33132, 33460
  - Para o experimento AQI, estão disponíveis o uso de 2 (aqi_pm_only), 4 (aqi_pm_plus) ou 9 (aqi_all) sensores.
    - Sensores disponíveis (valores simulados): 33250: 'PM2.5', 33251: 'PM10', 33252: 'NO', 33253: 'NO2', 33254: 'NOx', 33255: 'NH3', 33256: 'CO', 33257: 'SO2', 33258: 'O3':
      - aqi_pm_only: 33250, 33251
      - aqi_pm_plus: 33250, 33251, 33256, 33258
      - aqi_all: 33250, 33251, 33252, 33253, 33254, 33255, 33256, 33257, 33258
  - O argumento "result" aceita três opções: multiclass (disponível para logreg e kmeans), binary (disponível para logreg e kmeans) ou regression (disponível para linreg)
  - O argumento n_classes deve ser usada apenas em combinação com logreg
  - O argumento n_groups deve ser usada em combinação com kmeans
  - O argumento anomaly_detection aceita três modos de detecção de anomalias: IForest, ECOD e LOF
  - O argumento partition aceita valores que vão de 0.1 a 1, e indicam a fração do conjunto de dados a ser utilizada pelo participante para realizar o treinamento. O valor informado também é utilizado como critério de aleatoriedade para seleção das amostras que serão utilizadas para treinamento pelo participante.


  **Informações sobre o treinamento federado**
  - O FedSensor está configurado para um mínimo de 3 participantes que devem se conectar para realizar o treinamento federado.
  - Para mudar isso, é necessário alterar os seguintes parâmetros no arquivo "manager.py":
      ```
      min_fit_clients=3,
      min_eval_clients=3,
      min_available_clients=3,
      ```
