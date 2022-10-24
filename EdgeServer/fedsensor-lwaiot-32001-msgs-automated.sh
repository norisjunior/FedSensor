#!/bin/bash

echo "#############################################################"
echo "### FedSensor-32001-Automated messages ###"
echo "#############################################################"

if [[ $# -eq 0 ]] ; then
    echo 'usage: fedsensor-lwaiot-32001-msgs-automated <IP> <message_code> <value>'
    echo 'Automated message to Sensortag, Remote, and Launchpad CC1352P1 (for experiments of the Thesis)'
    echo 'IP: localhost'
    echo 'IoT devices (Name/Type - deviceID):'
    echo '"Native" "010203060708"'
    echo '"Sensortag" "00124b05257a"'
    echo '"Remote" "00124b4a527d"'
    echo '"Launchpad CC1352P1" "00124ba1ad06"'
    echo '"WindowsWiFi" "00122518499d"'
    exit 0
fi


echo $1

if   [ -z "$2" ] ; then
  echo 'error: provide FedSensor-LWAIoT message'
  exit 0
fi

if   [ -z "$3" ] ; then
  echo 'error: provide value'
  exit 0;
fi

IP=$1

algorithm=$2

value=$3

iterations=0

echo "Time Now: `date +%H:%M:%S`"
echo ""

while [ $iterations -lt 50 ];
do
  echo "sleep ..."
  sleep 10
  echo " "
  echo "Request Sensortag:"
  ./lwaiot_msgs.sh "$IP" "$algorithm" "$value" "Sensortag" "00124b05257a"
  echo "Request Remote:"
  ./lwaiot_msgs.sh "$IP" "$algorithm" "$value" "Remote" "00124b4a527d"
  echo "Request Launchpad:"
  ./lwaiot_msgs.sh "$IP" "$algorithm" "$value" "Launchpad" "00124ba1ad06"
  echo "Request Native:"
  ./lwaiot_msgs.sh "$IP" "$algorithm" "$value" "Native" "010203060708"
  echo "Request Windows WiFi:"
  ./lwaiot_msgs.sh "$IP" "$algorithm" "$value" "WindowsWiFi" "00122518499d"
  iterations=$(( $iterations + 1 ))
done
