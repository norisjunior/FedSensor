#!/bin/bash

echo "#############################################################"
echo "### FedSensor-MLModels-Automated messages ###"
echo "#############################################################"

if [[ $# -eq 0 ]] ; then
    echo 'usage: fedsensor-lwaiot-MLModels-msgs-automated <IP> <32001 message> <ML algorithm> <MLModel message>'
    echo 'Automated message to Sensortag, Remote, and Launchpad CC1352P1 (for experiments of the Thesis)'
    echo 'IP: localhost'
    echo 'IoT devices (Name/Type - deviceID):'
    #echo '"Native" "010203060708"'
    echo '"Sensortag" "00124b05257a"'
    echo '"Remote" "00124b4a527d"'
    echo '"Launchpad CC1352P1" "00124ba1ad06"'
    #echo '"WindowsWiFi" "0012e3034d8c"'
    #echo '"Raspberry" "0012eb00f6d0" - Raspberry Pi 3B'
    #echo '"Raspberry" "0012ebc894cb" - Raspberry Pi Zero W'
    exit 0
fi


echo $1

if   [ -z "$2" ] ; then
  echo 'error: provide FedSensor-LWAIoT-32001 message'
  exit 0
fi

if   [ -z "$3" ] ; then
  echo 'error: provide ML Model'
  exit 0;
fi

if   [ -z "$4" ] ; then
  echo 'error: provide ML Model message'
  exit 0;
fi

IP=$1

provisionMessage=$2

algorithm=$3

mlModelMessage=$4

iterations=0

echo "Time Now: `date +%H:%M:%S`"
echo ""


echo "Provision model - 32001 message:"
echo " "

# echo "Provision 32001 on Native:"
# ./lwaiot_msgs.sh "$IP" "32001" "$provisionMessage" "Native" "010203060708"
# sleep 2


echo "Provision 32001 on Sensortag:"
./lwaiot_msgs.sh "$IP" "32001" "$provisionMessage" "Sensortag" "00124b05257a"
sleep 2

echo "Provision 32001 on Remote:"
./lwaiot_msgs.sh "$IP" "32001" "$provisionMessage" "Remote" "00124b4a527d"
sleep 2

echo "Provision 32001 on CC1352P1:"
./lwaiot_msgs.sh "$IP" "32001" "$provisionMessage" "Launchpad" "00124ba1ad06"
sleep 2


while [ $iterations -lt 35 ];
do
  echo "wait 10 seconds ..."
  sleep 10
  echo " "

  # echo "Request Native:"
  # ./lwaiot_msgs.sh "$IP" "$algorithm" "$mlModelMessage" "Native" "010203060708"

  echo "Request Sensortag:"
  ./lwaiot_msgs.sh "$IP" "$algorithm" "$mlModelMessage" "Sensortag" "00124b05257a"
  sleep 1

  echo "Request Remote:"
  ./lwaiot_msgs.sh "$IP" "$algorithm" "$mlModelMessage" "Remote" "00124b4a527d"
  sleep 1

  echo "Request CC1352P1:"
  ./lwaiot_msgs.sh "$IP" "$algorithm" "$mlModelMessage" "Launchpad" "00124ba1ad06"

  iterations=$(( $iterations + 1 ))
done

echo "Finished..."
play -v 1 /usr/share/sounds/sound-icons/xylofon.wav
