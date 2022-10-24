#!/bin/bash

echo "#############################################################"
echo "###                  FedSensor - MESSAGES                 ###"
echo "#############################################################"

if [[ $# -eq 0 ]] ; then
    echo 'usage: lwaiot_msgs_msg.sh <IP> <message_code> <value> <device_name/type> <deviceID>'
    echo 'IP: localhost'
    echo 'IP ProjectHuaweiUSP: 159.138.214.207'
    echo 'IoT devices (Name/Type - deviceID):'
    echo '"Native" "010203060708"'
    echo '"Sensortag" "00124b05257a"'
    echo '"Remote" "00124b4a527d"'
    echo '"Launchpad CC1352P1" "00124ba1ad06"'
    echo '"WindowsWiFi" "0012e3034d8c"'
    echo '"Raspberry" "0012eb00f6d0" - Raspberry Pi 3B'
    echo '"Raspberry" "0012ebc894cb" - Raspberry Pi Zero W'
    exit 0
fi

echo $1

if   [ -z "$2" ] ; then
  echo 'error: provide LWML4AIoT message'
  exit 0
fi

if   [ -z "$3" ] ; then
  echo 'error: provide value'
  exit 0
fi

if   [ -z "$4" ] ; then
  echo 'error: provide device_name/type'
  exit 0
fi

if   [ -z "$5" ] ; then
  echo 'error: provide deviceID'
  exit 0
fi

# if   [ "$1" == 'red' ] && [ "$1" != 'green' ] ; then
#   echo 'error: only \"red\" of \"green\" allowed modes'
#   exit 0
# fi

IP=$1

algorithm=$2

value=$3

devicename=$4

deviceID=$5

#text=$(tr -d ' ' <<< "$text")

echo -e "FedSensor-LWAIoT:"
curl -iX POST \
  http://$IP:4041/v2/op/update \
  -H 'Content-Type: application/json' \
  -H 'fiware-service: school' \
  -H 'fiware-servicepath: /fd00' \
  -d '{
    "actionType": "update",
    "entities": [
        {
            "type": "'"$devicename"'",
            "id": "'"$deviceID"'",
            "'"$algorithm"'0" : {
                "type": "command",
                "value": "'"$value"'"
            }
        }
    ]
}'

echo $IP
echo $algorithm
echo $value
echo $devicename
echo $deviceID
