#!/bin/bash

echo "#############################################################"
echo "### CWIoT-Automated CommandRequest-Sensortag CC2650 target###"
echo "#############################################################"

if [[ $# -eq 0 ]] ; then
    echo 'usage: commandReceived-automated <board/devicename/type>'
    exit 0
fi

#Our boards:
#Sensortag CC2650 - devicename/type = Sensortag
#                   deviceID        = 00124b05257a

#Remote CC2538 - devicename/type = Remote
#                deviceID        = 00124b4a527d

#"entity_name": "0012eb00f6d0",
#"entity_type": "Raspberry",


if   [ "$1" != 'SENSORTAG' ] && [ "$1" != 'REMOTE' ] && [ "$1" != 'Raspberry' ] && [ "$1" != 'Native' ] ; then
  echo 'error: only \"SENSORTAG\", \"REMOTE\", or \"Raspberry\" allowed boards'
  exit 0
fi

if [ "$1" == 'SENSORTAG' ] ; then
  devicename="Sensortag"
  deviceID="00124b05257a"
elif [ "$1" == 'REMOTE' ] ; then
  devicename="Remote"
  deviceID="00124b4a527d"
elif [ "$1" == 'Raspberry' ] ; then
  devicename="Raspberry"
  deviceID="0012eb00f6d0"
elif [ "$1" == 'Native' ] ; then
  devicename="Native"
  deviceID="010203060708"
fi

wait_period=0
switch_leds=0

while true
do
  echo "Time Now: `date +%H:%M:%S`"
  echo ""
  # 720 seconds = 12 minutes
  wait_period=$(($wait_period+30))
  if [ $wait_period -gt 3000 ];then
    echo "The script successfully ran for 50 minutes, exiting now ... beep!"
    ./toggle-buzzer.sh "$devicename" "$deviceID"
    break
  else
    sleep 10
    echo "CommandReceived - Execution - Red Led (3311 0) On (1) and Off (0)"
    if [ $switch_leds == 0 ]; then #leds off
      ./toggle-leds.sh red on "$devicename" "$deviceID"
      switch_leds=1
    else
      ./toggle-leds.sh red off "$devicename" "$deviceID"
      switch_leds=0
    fi
    sleep 10
    echo ""
    echo "CommandReceived - Request - temperature (3303)"
    ./request-temperature.sh "$devicename" "$deviceID"
    echo ""
    sleep 10
  fi
done
