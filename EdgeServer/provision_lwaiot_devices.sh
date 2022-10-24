#!/bin/bash


echo "###############################################################"
echo "###  Provisioning FedSensor devices on Orion and IoT Agent  ###"
echo "###############################################################"

echo "###### API-KEY in this environment: 99"

###########################################################################
####### API-KEY ###########################################################
###########################################################################
curl -iX POST \
  'http://localhost:4041/iot/services' \
  -H 'Content-Type: application/json' \
  -H 'fiware-service: school' \
  -H 'fiware-servicepath: /fd00' \
  -d '{
 "services": [
   {
     "apikey":      "99",
     "cbroker":     "http://orion:1026",
     "entity_type": "LWPUBSUB",
     "resource":    "/iot/d"
   }
 ]
}'


#sleep 1;





###########################################################################
####### Device provisioning ###############################################
###########################################################################

###########################################################################
######    CWIoT suffix: 010203060708 (Native)                       #######
###########################################################################

echo "\n###### Device provisioning on the microcloud platform:\n"

echo "\n###### NATIVE TARGET \n"

curl -iX POST \
  'http://localhost:4041/iot/devices' \
  -H 'Content-Type: application/json' \
  -H 'fiware-service: school' \
  -H 'fiware-servicepath: /fd00' \
  -d '{
 "devices": [
   {
     "device_id":   "010203060708",
     "entity_name": "010203060708",
     "entity_type": "Native",
     "protocol":    "PDI-IoTA-UltraLight",
     "transport":   "MQTT",
     "timezone":    "Europe/Berlin",
     "commands": [
       {"name": "320010","type": "command"},
       {"name": "320020","type": "command"},
       {"name": "320021","type": "command"},
       {"name": "320022","type": "command"},
       {"name": "320023","type": "command"},
       {"name": "320024","type": "command"},
       {"name": "320025","type": "command"},
       {"name": "320026","type": "command"},
       {"name": "320027","type": "command"},
       {"name": "320028","type": "command"},
       {"name": "320029","type": "command"},
       {"name": "320030","type": "command"},
       {"name": "320031","type": "command"},
       {"name": "321010","type": "command"},
       {"name": "321011","type": "command"},
       {"name": "321012","type": "command"},
       {"name": "321020","type": "command"},
       {"name": "321021","type": "command"},
       {"name": "321022","type": "command"},
       {"name": "321023","type": "command"},
       {"name": "321024","type": "command"},
       {"name": "321025","type": "command"},
       {"name": "321026","type": "command"},
       {"name": "321027","type": "command"},
       {"name": "321028","type": "command"},
       {"name": "321029","type": "command"},
       {"name": "321030","type": "command"},
       {"name": "321031","type": "command"},
       {"name": "321032","type": "command"},
       {"name": "321033","type": "command"},
       {"name": "321034","type": "command"},
       {"name": "321035","type": "command"},
       {"name": "321036","type": "command"},
       {"name": "321037","type": "command"},
       {"name": "321038","type": "command"},
       {"name": "321039","type": "command"},
       {"name": "321040","type": "command"},
       {"name": "321041","type": "command"},
       {"name": "321042","type": "command"},
       {"name": "321043","type": "command"},
       {"name": "321044","type": "command"},
       {"name": "321045","type": "command"},
       {"name": "321046","type": "command"},
       {"name": "321047","type": "command"},
       {"name": "321048","type": "command"},
       {"name": "321049","type": "command"},
       {"name": "321050","type": "command"},
       {"name": "321051","type": "command"},
       {"name": "321052","type": "command"},
       {"name": "321053","type": "command"},
       {"name": "321054","type": "command"},
       {"name": "321055","type": "command"},
       {"name": "321056","type": "command"},
       {"name": "321057","type": "command"},
       {"name": "321058","type": "command"},
       {"name": "321059","type": "command"},
       {"name": "321060","type": "command"},
       {"name": "321061","type": "command"},
       {"name": "321062","type": "command"},
       {"name": "321063","type": "command"},
       {"name": "321064","type": "command"},
       {"name": "321065","type": "command"},
       {"name": "321066","type": "command"},
       {"name": "321067","type": "command"},
       {"name": "321068","type": "command"},
       {"name": "321069","type": "command"},
       {"name": "033030","type": "command"},
       {"name": "033040","type": "command"},
       {"name": "033110","type": "command"},
       {"name": "033111","type": "command"},
       {"name": "033380","type": "command"}
      ],
      "attributes": [
       {"object_id": "033030", "name": "033030", "type": "actual_temp"},
       {"object_id": "033040", "name": "033040", "type": "actual_hum"}
      ],
      "static_attributes": [
        {"name":"refStore", "type": "Relationship","value": "urn:ngsi-ld:School:001"}
     ]
   }
 ]
}
'

#sleep 1;


echo "\n###### SENSORTAG CC2650 TARGET\n"

curl -iX POST \
  'http://localhost:4041/iot/devices' \
  -H 'Content-Type: application/json' \
  -H 'fiware-service: school' \
  -H 'fiware-servicepath: /fd00' \
  -d '{
 "devices": [
   {
     "device_id":   "00124b05257a",
     "entity_name": "00124b05257a",
     "entity_type": "Sensortag",
     "protocol":    "PDI-IoTA-UltraLight",
     "transport":   "MQTT",
     "timezone":    "Europe/Berlin",
     "commands": [
     {"name": "320010","type": "command"},
     {"name": "320020","type": "command"},
     {"name": "320021","type": "command"},
     {"name": "320022","type": "command"},
     {"name": "320023","type": "command"},
     {"name": "320024","type": "command"},
     {"name": "320025","type": "command"},
     {"name": "320026","type": "command"},
     {"name": "320027","type": "command"},
     {"name": "320028","type": "command"},
     {"name": "320029","type": "command"},
     {"name": "320030","type": "command"},
     {"name": "320031","type": "command"},
     {"name": "321010","type": "command"},
     {"name": "321011","type": "command"},
     {"name": "321012","type": "command"},
     {"name": "321020","type": "command"},
     {"name": "321021","type": "command"},
     {"name": "321022","type": "command"},
     {"name": "321023","type": "command"},
     {"name": "321024","type": "command"},
     {"name": "321025","type": "command"},
     {"name": "321026","type": "command"},
     {"name": "321027","type": "command"},
     {"name": "321028","type": "command"},
     {"name": "321029","type": "command"},
     {"name": "321030","type": "command"},
     {"name": "321031","type": "command"},
     {"name": "321032","type": "command"},
     {"name": "321033","type": "command"},
     {"name": "321034","type": "command"},
     {"name": "321035","type": "command"},
     {"name": "321036","type": "command"},
     {"name": "321037","type": "command"},
     {"name": "321038","type": "command"},
     {"name": "321039","type": "command"},
     {"name": "321040","type": "command"},
     {"name": "321041","type": "command"},
     {"name": "321042","type": "command"},
     {"name": "321043","type": "command"},
     {"name": "321044","type": "command"},
     {"name": "321045","type": "command"},
     {"name": "321046","type": "command"},
     {"name": "321047","type": "command"},
     {"name": "321048","type": "command"},
     {"name": "321049","type": "command"},
     {"name": "321050","type": "command"},
     {"name": "321051","type": "command"},
     {"name": "321052","type": "command"},
     {"name": "321053","type": "command"},
     {"name": "321054","type": "command"},
     {"name": "321055","type": "command"},
     {"name": "321056","type": "command"},
     {"name": "321057","type": "command"},
     {"name": "321058","type": "command"},
     {"name": "321059","type": "command"},
     {"name": "321060","type": "command"},
     {"name": "321061","type": "command"},
     {"name": "321062","type": "command"},
     {"name": "321063","type": "command"},
     {"name": "321064","type": "command"},
     {"name": "321065","type": "command"},
     {"name": "321066","type": "command"},
     {"name": "321067","type": "command"},
     {"name": "321068","type": "command"},
     {"name": "321069","type": "command"},
     {"name": "033030","type": "command"},
     {"name": "033040","type": "command"},
     {"name": "033110","type": "command"},
     {"name": "033111","type": "command"},
     {"name": "033380","type": "command"}
      ],
      "attributes": [
       {"object_id": "033030", "name": "033030", "type": "actual_temp"},
       {"object_id": "033040", "name": "033040", "type": "actual_hum"}
      ],
      "static_attributes": [
        {"name":"refStore", "type": "Relationship","value": "urn:ngsi-ld:School:001"}
     ]
   }
 ]
}
'


#sleep 1;


echo "\n###### REMOTE CC2538 TARGET\n"

curl -iX POST \
  'http://localhost:4041/iot/devices' \
  -H 'Content-Type: application/json' \
  -H 'fiware-service: school' \
  -H 'fiware-servicepath: /fd00' \
  -d '{
 "devices": [
   {
     "device_id":   "00124b4a527d",
     "entity_name": "00124b4a527d",
     "entity_type": "Remote",
     "protocol":    "PDI-IoTA-UltraLight",
     "transport":   "MQTT",
     "timezone":    "Europe/Berlin",
     "commands": [
     {"name": "320010","type": "command"},
     {"name": "320020","type": "command"},
     {"name": "320021","type": "command"},
     {"name": "320022","type": "command"},
     {"name": "320023","type": "command"},
     {"name": "320024","type": "command"},
     {"name": "320025","type": "command"},
     {"name": "320026","type": "command"},
     {"name": "320027","type": "command"},
     {"name": "320028","type": "command"},
     {"name": "320029","type": "command"},
     {"name": "320030","type": "command"},
     {"name": "320031","type": "command"},
     {"name": "321010","type": "command"},
     {"name": "321011","type": "command"},
     {"name": "321012","type": "command"},
     {"name": "321020","type": "command"},
     {"name": "321021","type": "command"},
     {"name": "321022","type": "command"},
     {"name": "321023","type": "command"},
     {"name": "321024","type": "command"},
     {"name": "321025","type": "command"},
     {"name": "321026","type": "command"},
     {"name": "321027","type": "command"},
     {"name": "321028","type": "command"},
     {"name": "321029","type": "command"},
     {"name": "321030","type": "command"},
     {"name": "321031","type": "command"},
     {"name": "321032","type": "command"},
     {"name": "321033","type": "command"},
     {"name": "321034","type": "command"},
     {"name": "321035","type": "command"},
     {"name": "321036","type": "command"},
     {"name": "321037","type": "command"},
     {"name": "321038","type": "command"},
     {"name": "321039","type": "command"},
     {"name": "321040","type": "command"},
     {"name": "321041","type": "command"},
     {"name": "321042","type": "command"},
     {"name": "321043","type": "command"},
     {"name": "321044","type": "command"},
     {"name": "321045","type": "command"},
     {"name": "321046","type": "command"},
     {"name": "321047","type": "command"},
     {"name": "321048","type": "command"},
     {"name": "321049","type": "command"},
     {"name": "321050","type": "command"},
     {"name": "321051","type": "command"},
     {"name": "321052","type": "command"},
     {"name": "321053","type": "command"},
     {"name": "321054","type": "command"},
     {"name": "321055","type": "command"},
     {"name": "321056","type": "command"},
     {"name": "321057","type": "command"},
     {"name": "321058","type": "command"},
     {"name": "321059","type": "command"},
     {"name": "321060","type": "command"},
     {"name": "321061","type": "command"},
     {"name": "321062","type": "command"},
     {"name": "321063","type": "command"},
     {"name": "321064","type": "command"},
     {"name": "321065","type": "command"},
     {"name": "321066","type": "command"},
     {"name": "321067","type": "command"},
     {"name": "321068","type": "command"},
     {"name": "321069","type": "command"},
     {"name": "033030","type": "command"},
     {"name": "033040","type": "command"},
     {"name": "033110","type": "command"},
     {"name": "033111","type": "command"},
     {"name": "033380","type": "command"}
     ],
      "attributes": [
       {"object_id": "033030", "name": "033030", "type": "actual_temp"},
       {"object_id": "033040", "name": "033040", "type": "actual_hum"}
      ],
      "static_attributes": [
        {"name":"refStore", "type": "Relationship","value": "urn:ngsi-ld:School:001"}
     ]
   }
 ]
}
'



#sleep 1;


echo "\n###### LAUNCHPAD CC1352P1 TARGET\n"

curl -iX POST \
  'http://localhost:4041/iot/devices' \
  -H 'Content-Type: application/json' \
  -H 'fiware-service: school' \
  -H 'fiware-servicepath: /fd00' \
  -d '{
 "devices": [
   {
     "device_id":   "00124ba1ad06",
     "entity_name": "00124ba1ad06",
     "entity_type": "Launchpad",
     "protocol":    "PDI-IoTA-UltraLight",
     "transport":   "MQTT",
     "timezone":    "Europe/Berlin",
     "commands": [
     {"name": "320010","type": "command"},
     {"name": "320020","type": "command"},
     {"name": "320021","type": "command"},
     {"name": "320022","type": "command"},
     {"name": "320023","type": "command"},
     {"name": "320024","type": "command"},
     {"name": "320025","type": "command"},
     {"name": "320026","type": "command"},
     {"name": "320027","type": "command"},
     {"name": "320028","type": "command"},
     {"name": "320029","type": "command"},
     {"name": "320030","type": "command"},
     {"name": "320031","type": "command"},
     {"name": "321010","type": "command"},
     {"name": "321011","type": "command"},
     {"name": "321012","type": "command"},
     {"name": "321020","type": "command"},
     {"name": "321021","type": "command"},
     {"name": "321022","type": "command"},
     {"name": "321023","type": "command"},
     {"name": "321024","type": "command"},
     {"name": "321025","type": "command"},
     {"name": "321026","type": "command"},
     {"name": "321027","type": "command"},
     {"name": "321028","type": "command"},
     {"name": "321029","type": "command"},
     {"name": "321030","type": "command"},
     {"name": "321031","type": "command"},
     {"name": "321032","type": "command"},
     {"name": "321033","type": "command"},
     {"name": "321034","type": "command"},
     {"name": "321035","type": "command"},
     {"name": "321036","type": "command"},
     {"name": "321037","type": "command"},
     {"name": "321038","type": "command"},
     {"name": "321039","type": "command"},
     {"name": "321040","type": "command"},
     {"name": "321041","type": "command"},
     {"name": "321042","type": "command"},
     {"name": "321043","type": "command"},
     {"name": "321044","type": "command"},
     {"name": "321045","type": "command"},
     {"name": "321046","type": "command"},
     {"name": "321047","type": "command"},
     {"name": "321048","type": "command"},
     {"name": "321049","type": "command"},
     {"name": "321050","type": "command"},
     {"name": "321051","type": "command"},
     {"name": "321052","type": "command"},
     {"name": "321053","type": "command"},
     {"name": "321054","type": "command"},
     {"name": "321055","type": "command"},
     {"name": "321056","type": "command"},
     {"name": "321057","type": "command"},
     {"name": "321058","type": "command"},
     {"name": "321059","type": "command"},
     {"name": "321060","type": "command"},
     {"name": "321061","type": "command"},
     {"name": "321062","type": "command"},
     {"name": "321063","type": "command"},
     {"name": "321064","type": "command"},
     {"name": "321065","type": "command"},
     {"name": "321066","type": "command"},
     {"name": "321067","type": "command"},
     {"name": "321068","type": "command"},
     {"name": "321069","type": "command"},
     {"name": "033030","type": "command"},
     {"name": "033040","type": "command"},
     {"name": "033110","type": "command"},
     {"name": "033111","type": "command"},
     {"name": "033380","type": "command"}
      ],
      "attributes": [
       {"object_id": "033030", "name": "033030", "type": "actual_temp"},
       {"object_id": "033040", "name": "033040", "type": "actual_hum"}
      ],
      "static_attributes": [
        {"name":"refStore", "type": "Relationship","value": "urn:ngsi-ld:School:001"}
     ]
   }
 ]
}
'



#sleep 1;


echo "\n###### LAUNCHPAD1 CC2650 TARGET\n"

curl -iX POST \
  'http://localhost:4041/iot/devices' \
  -H 'Content-Type: application/json' \
  -H 'fiware-service: school' \
  -H 'fiware-servicepath: /fd00' \
  -d '{
 "devices": [
   {
     "device_id":   "00124b82ad03",
     "entity_name": "00124b82ad03",
     "entity_type": "Launchpad",
     "protocol":    "PDI-IoTA-UltraLight",
     "transport":   "MQTT",
     "timezone":    "Europe/Berlin",
     "commands": [
       {"name": "033030","type": "command"},
       {"name": "033040","type": "command"},
       {"name": "033110","type": "command"},
       {"name": "033111","type": "command"},
       {"name": "033380","type": "command"}
      ],
      "attributes": [
       {"object_id": "033030", "name": "033030", "type": "actual_temp"},
       {"object_id": "033040", "name": "033040", "type": "actual_hum"}
      ],
      "static_attributes": [
        {"name":"refStore", "type": "Relationship","value": "urn:ngsi-ld:School:001"}
     ]
   }
 ]
}
'



#sleep 1;



echo "\n###### LAUNCHPAD2 CC2650 TARGET CWIoT\n"

curl -iX POST \
  'http://localhost:4041/iot/devices' \
  -H 'Content-Type: application/json' \
  -H 'fiware-service: school' \
  -H 'fiware-servicepath: /fd00' \
  -d '{
 "devices": [
   {
     "device_id":   "00124b82b206",
     "entity_name": "00124b82b206",
     "entity_type": "Launchpad",
     "protocol":    "PDI-IoTA-UltraLight",
     "transport":   "MQTT",
     "timezone":    "Europe/Berlin",
     "commands": [
       {"name": "033030","type": "command"},
       {"name": "033040","type": "command"},
       {"name": "033110","type": "command"},
       {"name": "033111","type": "command"},
       {"name": "033380","type": "command"}
      ],
      "attributes": [
       {"object_id": "033030", "name": "033030", "type": "actual_temp"},
       {"object_id": "033040", "name": "033040", "type": "actual_hum"}
      ],
      "static_attributes": [
        {"name":"refStore", "type": "Relationship","value": "urn:ngsi-ld:School:001"}
     ]
   }
 ]
}
'
#sleep 1;



echo "\n###### Windows WiFi standalone app\n"

curl -iX POST \
  'http://localhost:4041/iot/devices' \
  -H 'Content-Type: application/json' \
  -H 'fiware-service: school' \
  -H 'fiware-servicepath: /fd00' \
  -d '{
 "devices": [
   {
     "device_id":   "00122518499d",
     "entity_name": "00122518499d",
     "entity_type": "WindowsWiFi",
     "protocol":    "PDI-IoTA-UltraLight",
     "transport":   "MQTT",
     "timezone":    "Europe/Berlin",
     "commands": [
       {"name": "320010","type": "command"},
       {"name": "320020","type": "command"},
       {"name": "320021","type": "command"},
       {"name": "320022","type": "command"},
       {"name": "320023","type": "command"},
       {"name": "320024","type": "command"},
       {"name": "320025","type": "command"},
       {"name": "320026","type": "command"},
       {"name": "320027","type": "command"},
       {"name": "320028","type": "command"},
       {"name": "320029","type": "command"},
       {"name": "320030","type": "command"},
       {"name": "320031","type": "command"},
       {"name": "321010","type": "command"},
       {"name": "321011","type": "command"},
       {"name": "321012","type": "command"},
       {"name": "321020","type": "command"},
       {"name": "321021","type": "command"},
       {"name": "321022","type": "command"},
       {"name": "321023","type": "command"},
       {"name": "321024","type": "command"},
       {"name": "321025","type": "command"},
       {"name": "321026","type": "command"},
       {"name": "321027","type": "command"},
       {"name": "321028","type": "command"},
       {"name": "321029","type": "command"},
       {"name": "321030","type": "command"},
       {"name": "321031","type": "command"},
       {"name": "321032","type": "command"},
       {"name": "321033","type": "command"},
       {"name": "321034","type": "command"},
       {"name": "321035","type": "command"},
       {"name": "321036","type": "command"},
       {"name": "321037","type": "command"},
       {"name": "321038","type": "command"},
       {"name": "321039","type": "command"},
       {"name": "321040","type": "command"},
       {"name": "321041","type": "command"},
       {"name": "321042","type": "command"},
       {"name": "321043","type": "command"},
       {"name": "321044","type": "command"},
       {"name": "321045","type": "command"},
       {"name": "321046","type": "command"},
       {"name": "321047","type": "command"},
       {"name": "321048","type": "command"},
       {"name": "321049","type": "command"},
       {"name": "321050","type": "command"},
       {"name": "321051","type": "command"},
       {"name": "321052","type": "command"},
       {"name": "321053","type": "command"},
       {"name": "321054","type": "command"},
       {"name": "321055","type": "command"},
       {"name": "321056","type": "command"},
       {"name": "321057","type": "command"},
       {"name": "321058","type": "command"},
       {"name": "321059","type": "command"},
       {"name": "321060","type": "command"},
       {"name": "321061","type": "command"},
       {"name": "321062","type": "command"},
       {"name": "321063","type": "command"},
       {"name": "321064","type": "command"},
       {"name": "321065","type": "command"},
       {"name": "321066","type": "command"},
       {"name": "321067","type": "command"},
       {"name": "321068","type": "command"},
       {"name": "321069","type": "command"},
       {"name": "033030","type": "command"},
       {"name": "033040","type": "command"},
       {"name": "033110","type": "command"},
       {"name": "033111","type": "command"},
       {"name": "033112","type": "command"},
       {"name": "033380","type": "command"}
      ],
      "attributes": [
       {"object_id": "033030", "name": "033030", "type": "actual_temp"},
       {"object_id": "033040", "name": "033040", "type": "actual_hum"}
      ],
      "static_attributes": [
        {"name":"refStore", "type": "Relationship","value": "urn:ngsi-ld:School:001"}
     ]
   }
 ]
}
'
#sleep 1;



# Raspberry pi: 'b8:27:eb:00:f6:d0'
#client_id: 0012eb00f6d0

curl -iX POST \
  'http://localhost:4041/iot/devices' \
  -H 'Content-Type: application/json' \
  -H 'fiware-service: school' \
  -H 'fiware-servicepath: /fd00' \
  -d '{
 "devices": [
   {
     "device_id":   "0012eb00f6d0",
     "entity_name": "0012eb00f6d0",
     "entity_type": "Raspberry",
     "protocol":    "PDI-IoTA-UltraLight",
     "transport":   "MQTT",
     "timezone":    "Europe/Berlin",
     "commands": [
       {"name": "033030","type": "command"},
       {"name": "033040","type": "command"},
       {"name": "033110","type": "command"},
       {"name": "033111","type": "command"},
       {"name": "033112","type": "command"},
       {"name": "033380","type": "command"}
      ],
      "attributes": [
       {"object_id": "033030", "name": "033030", "type": "actual_temp"},
       {"object_id": "033040", "name": "033040", "type": "actual_hum"}
      ],
      "static_attributes": [
        {"name":"refStore", "type": "Relationship","value": "urn:ngsi-ld:School:001"}
     ]
   }
 ]
}
'

#sleep 1;





#Raspberry Pi Zero W
#0012ebc894cb
curl -iX POST \
  'http://localhost:4041/iot/devices' \
  -H 'Content-Type: application/json' \
  -H 'fiware-service: school' \
  -H 'fiware-servicepath: /fd00' \
  -d '{
 "devices": [
   {
     "device_id":   "0012ebc894cb",
     "entity_name": "0012ebc894cb",
     "entity_type": "Raspberry",
     "protocol":    "PDI-IoTA-UltraLight",
     "transport":   "MQTT",
     "timezone":    "Europe/Berlin",
     "commands": [
       {"name": "320010","type": "command"},
       {"name": "320020","type": "command"},
       {"name": "320021","type": "command"},
       {"name": "320022","type": "command"},
       {"name": "320023","type": "command"},
       {"name": "320024","type": "command"},
       {"name": "320025","type": "command"},
       {"name": "320026","type": "command"},
       {"name": "320027","type": "command"},
       {"name": "320028","type": "command"},
       {"name": "320029","type": "command"},
       {"name": "320030","type": "command"},
       {"name": "320031","type": "command"},
       {"name": "321010","type": "command"},
       {"name": "321011","type": "command"},
       {"name": "321012","type": "command"},
       {"name": "321020","type": "command"},
       {"name": "321021","type": "command"},
       {"name": "321022","type": "command"},
       {"name": "321023","type": "command"},
       {"name": "321024","type": "command"},
       {"name": "321025","type": "command"},
       {"name": "321026","type": "command"},
       {"name": "321027","type": "command"},
       {"name": "321028","type": "command"},
       {"name": "321029","type": "command"},
       {"name": "321030","type": "command"},
       {"name": "321031","type": "command"},
       {"name": "321032","type": "command"},
       {"name": "321033","type": "command"},
       {"name": "321034","type": "command"},
       {"name": "321035","type": "command"},
       {"name": "321036","type": "command"},
       {"name": "321037","type": "command"},
       {"name": "321038","type": "command"},
       {"name": "321039","type": "command"},
       {"name": "321040","type": "command"},
       {"name": "321041","type": "command"},
       {"name": "321042","type": "command"},
       {"name": "321043","type": "command"},
       {"name": "321044","type": "command"},
       {"name": "321045","type": "command"},
       {"name": "321046","type": "command"},
       {"name": "321047","type": "command"},
       {"name": "321048","type": "command"},
       {"name": "321049","type": "command"},
       {"name": "321050","type": "command"},
       {"name": "321051","type": "command"},
       {"name": "321052","type": "command"},
       {"name": "321053","type": "command"},
       {"name": "321054","type": "command"},
       {"name": "321055","type": "command"},
       {"name": "321056","type": "command"},
       {"name": "321057","type": "command"},
       {"name": "321058","type": "command"},
       {"name": "321059","type": "command"},
       {"name": "321060","type": "command"},
       {"name": "321061","type": "command"},
       {"name": "321062","type": "command"},
       {"name": "321063","type": "command"},
       {"name": "321064","type": "command"},
       {"name": "321065","type": "command"},
       {"name": "321066","type": "command"},
       {"name": "321067","type": "command"},
       {"name": "321068","type": "command"},
       {"name": "321069","type": "command"},
       {"name": "033030","type": "command"},
       {"name": "033040","type": "command"},
       {"name": "033110","type": "command"},
       {"name": "033111","type": "command"},
       {"name": "033112","type": "command"},
       {"name": "033380","type": "command"}
      ],
      "attributes": [
       {"object_id": "033030", "name": "033030", "type": "actual_temp"},
       {"object_id": "033040", "name": "033040", "type": "actual_hum"}
      ],
      "static_attributes": [
        {"name":"refStore", "type": "Relationship","value": "urn:ngsi-ld:School:001"}
     ]
   }
 ]
}
'

sleep 1;


###########################################################################
###########################################################################
#### Dummy test measurements (for testing purposes): ######################
###########################################################################
###########################################################################

echo "#### Mosquitto dummy temperatura sensortag ####"
mosquitto_pub -h localhost -t /99/00124b05257a -m '0033030|23.123'

sleep 1;

echo "#### Mosquitto dummy umidade ####"
mosquitto_pub -h localhost -t /99/00124b05257a -m '0033040|80.789'

sleep 1;

echo "#### Mosquitto dummy RedLed ####"
mosquitto_pub -h localhost -t /99/00124b05257a -m '0033110|1'

sleep 1;

echo "#### Mosquitto dummy GreenLed ####"
mosquitto_pub -h localhost -t /99/00124b05257a -m '0033111|1'




echo "#### Mosquitto dummy temperatura remote ####"
mosquitto_pub -h localhost -t /99/00124b4a527d -m '0033030|23.123'

sleep 1;

echo "#### Mosquitto dummy temperatura native ####"
mosquitto_pub -h localhost -t /99/010203060708 -m '0033030|23.123'

sleep 1;

echo "#### Mosquitto dummy temperatura launchpad1 ####"
mosquitto_pub -h localhost -t /99/00124b82ad03 -m '0033030|23.123'

sleep 1;

echo "#### Mosquitto dummy temperatura launchpad2 ####"
mosquitto_pub -h localhost -t /99/00124b82b206 -m '0033030|23.123'


sleep 1;


echo "############################################################"
echo "############                                    ############"
echo "###########                                      ###########"
echo "##########          RESUMED DEVICES INFO          ##########"
echo "###########                                      ###########"
echo "############                                    ############"
echo "############################################################"


sleep 1;

curl -X GET \
  'http://localhost:1026/v2/entities/010203060708?type=Native&options=keyValues' \
  -H 'fiware-service: school' \
  -H 'fiware-servicepath: /fd00' | python3 -m json.tool

sleep 1;

curl -X GET \
  'http://localhost:1026/v2/entities/00124b05257a?type=Sensortag&options=keyValues' \
  -H 'fiware-service: school' \
  -H 'fiware-servicepath: /fd00' | python3 -m json.tool

sleep 1;

curl -X GET \
  'http://localhost:1026/v2/entities/0012e3034d8c?type=WindowsWiFi&options=keyValues' \
  -H 'fiware-service: school' \
  -H 'fiware-servicepath: /fd00' | python3 -m json.tool

sleep 1;


curl -X GET \
  'http://localhost:1026/v2/entities/0012ebc894cb?type=Raspberry&options=keyValues' \
  -H 'fiware-service: school' \
  -H 'fiware-servicepath: /fd00' | python -m json.tool

sleep 1;
