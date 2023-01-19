#!/bin/bash

USERID=twkim
sudo chmod g+rw /dev/hidraw4 /dev/hidraw5 /dev/hidraw6 /dev/hidraw7 /dev/hidraw8 /dev/hidraw9 /dev/hidraw10
sudo chgrp $USERID /dev/hidraw4 /dev/hidraw5 /dev/hidraw6 /dev/hidraw7 /dev/hidraw8 /dev/hidraw9 /dev/hidraw10

ls -al /dev/hidraw*
