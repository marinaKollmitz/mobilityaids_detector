# mobilityaids_detector
ROS node for 3D perception of people and their mobility aids, using the Faster R-CNN framework  

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
