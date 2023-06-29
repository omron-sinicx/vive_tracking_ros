# vive_tracking_ros
pyOpenVR based vive controller tracking for teleoperation.

For now, only the twist and input feedback is published to ROS.

In the future, I may add the pose.

Inspired by [robosavvy/vive_ros](https://github.com/robosavvy/vive_ros) and [triad_openvr](https://github.com/TriadSemi/triad_openvr) 

This was tested on ROS Noetic / Ubuntu 20.04

# Getting started

After installing dependencies and this package

Start the VR server
```
roslaunch vive_tracking_ros server_vr.launch
```

To close the node you can `Ctrl+C`. To close the vr server you have to kill the process.
```
rosrun vive_tracking_ros close_server.sh
```

To start tracking the controllers run (added part from original package):  
```
roslaunch vive_tracking_ros tracking_controller.launch
```


## Installation instructions

### Download and build Valve's OpenVR SDK (most recently tested version):

``` shell
cd ~
mkdir libraries
cd libraries
git clone https://github.com/ValveSoftware/openvr.git -b v1.3.22
cd openvr
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ../
make
```

### Allow hardware access
Then plug-in VIVE to your computer and make sure you can see the devices on `/dev/hidraw[1-6]`.

Copy the file `60-HTC-Vive-perms.rules` to the folder `/etc/udev/rules.d`. Then run:

      sudo udevadm control --reload-rules && sudo udevadm trigger

### Install Steam and SteamVR

Install Steam:
      
      sudo apt install steam

Run Steam:
      
      steam

Setup or log in into your Steam account and install SteamVR app from Steam store.

Steam files should be located in: `~/.steam/steam`

SteamVR files should be located in: `~/.steam/steam/steamapps/common/SteamVR`

#### To work without the Headset HMD

See here for [instructions](https://github.com/moon-wreckers/vive_tracker) 
