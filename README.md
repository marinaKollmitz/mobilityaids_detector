# mobilityaids_detector
ROS node for 3D perception of people and their mobility aids, using the Faster R-CNN framework.  

## Setup
1. Mobilityaids Detector Code

   Clone mobilityaids_detector code in your catkin_ws and compile
   ```
   cd ~/catkin_ws/src
   git clone https://github.com/marinaKollmitz/mobilityaids_detector.git
   cd ..
   catkin_make
   ```

2. Multiclass Tracking Code

   Additionally you need the [multiclass-people-tracking](https://github.com/marinaKollmitz/multiclass-people-tracking) code. Clone    it into a directory of your choice which we will refer to as `$TRACKING_ROOT`:
   ```
   cd $TRACKING_ROOT
   git clone https://github.com/marinaKollmitz/multiclass-people-tracking
   ```
   To make sure the mobilityaids_detector can find the tracking code you can add it to your `$PYTHONPATH` by adding the following to your `.bashrc`:
   ```
   export PYTHONPATH=$PYTHONPATH:$TRACKING_ROOT/multiclass-people-tracking/
   ```

3. Detectron Code

   Lastly, you need our adapted Detectron Code for people detection. Please follow the points "Installation" and "Get Trained Mobility Aids Models" from the [Mobilityaids Howto](https://github.com/marinaKollmitz/DetectronDistance/blob/master/MOBILITYAIDS_HOWTO.md).

## Mobilityaids Detector Demo

To run the mobilityaids detection demo, first download the `test_mobilityaids.bag` file:
``` 
wget http://mobility-aids.informatik.uni-freiburg.de/test_mobilityaids.bag
```
Start the mobilityaids_detector ROS node and play the bag file:
``` 
roslaunch mobilityaids_detector detection.launch
```
You should see an rviz window coming up and the console output should print "waiting for images". Now play the bag file
```
rosbag play test_mobilityaids.bag
```
You should see tracking visualizations in rviz, like this:
<div align="center">
  <img src="mobilityaids_demo.png" width="500px" />
  <p>Mobilityaids Detector Demo</p>
</div>

## Tracking Configuration and ROS parameters
