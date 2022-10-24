#!/bin/bash

echo "#############################################################"
echo "###                      Toggle leds                      ###"
echo "#############################################################"

if [[ $# -eq 0 ]] ; then
    echo 'usage: toggle-leds <IP> <led> <on/off> <device_name/type> <deviceID>\n\n'
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

if   [ -z "$4" ] ; then
  echo 'error: provide device_name/type'
  exit 0
fi

if   [ -z "$5" ] ; then
  echo 'error: provide deviceID'
  exit 0
fi

if   [ "$2" != 'red' ] && [ "$2" != 'green' ] && [ "$2" != 'blue' ] ; then
  echo 'error: only \"red\", \"green\", \"blue\" allowed modes'
  exit 0
fi

if   [ "$3" != 'on' ] && [ "$3" != 'off' ] ; then
  echo 'error: only \"red\" of \"green\" allowed modes'
  exit 0
fi

#Default = turn on
IP=$1

action=1

devicename=$4

deviceID=$5

#text=$(tr -d ' ' <<< "$text")

redled () {
  if [ "$1" == 'on' ] ; then
  	action=1
  else
  	action=0
  fi
  echo "red" $action

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
              "033110" : {
                  "type": "command",
                  "value": "'$action'"
              }
          }
      ]
  }'

}



greenled () {
  if [ "$1" == 'on' ] ; then
  	action=1
  else
  	action=0
  fi
  echo "green" $action

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
              "033111" : {
                  "type": "command",
                  "value": "'$action'"
              }
          }
      ]
  }'

}

blueled () {
  if [ "$1" == 'on' ] ; then
  	action=1
  else
  	action=0
  fi
  echo "blue" $action

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
              "033112" : {
                  "type": "command",
                  "value": "'$action'"
              }
          }
      ]
  }'

}


case "$2" in
    'red')   redled $3 ;;
    'green')  greenled $3 ;;
    'blue')  blueled $3 ;;
    *) echo 'error: only \"red\", \"green\", or \"blue\" and \"on\" of \"off\" allowed modes' ;;
esac
