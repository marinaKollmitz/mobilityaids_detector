#! /usr/bin/env python

import detectron #for finding detectron path
import detectron.core.test_engine as infer_engine
import detectron.utils.c2 as c2_utils
from detectron.core.config import cfg, merge_cfg_from_file, assert_and_infer_cfg
from detectron.core.track_engine import validate_tracking_params
from detectron.utils.vis import convert_from_cls_format

import multiclass_tracking
from multiclass_tracking.tracker import Tracker
from multiclass_tracking.viz import Visualizer

from sensor_msgs.msg import CameraInfo, Image
from dynamic_reconfigure.server import Server
from mobilityaids_detector.cfg import TrackingParamsConfig

from publisher import Publisher
from image_handler import ImageHandler
from inside_box_filter import filter_inside_boxes

from cv_bridge import CvBridge
import numpy as np
import rospy
import tf
import os
import inspect

#dirty hack to fix python logging with ROS
import logging
detectron.core.track_engine.logger.addHandler(logging.StreamHandler())
detectron.core.test_engine.logger.addHandler(logging.StreamHandler())
multiclass_tracking.tracker.logger.addHandler(logging.StreamHandler())

class Detector:
    
    def __init__(self):
        
        detectron_root = os.path.join(os.path.dirname(inspect.getfile(detectron)), os.pardir)
        
        #model config file - mandatory
        config_file = rospy.get_param('~model_config', "")
        
        if not os.path.exists(config_file):
            rospy.logerr("config file '{}' does not exist. ".format(config_file) + 
                         "Please specify a valid model config file for the " +
                         "model_config ros param. See " +
                         "https://github.com/marinaKollmitz/mobilityaids_detector " +
                         "for setup instructions")
            exit(0)
        
        merge_cfg_from_file(config_file)
        
        #absolute output dir path
        cfg.OUTPUT_DIR = os.path.join(detectron_root, cfg.OUTPUT_DIR)
        
        weights_file = os.path.join(detectron_root, cfg.TEST.WEIGHTS)
        val_dataset = cfg.TRACK.VALIDATION_DATASET

        assert_and_infer_cfg()
        self.model = infer_engine.initialize_model_from_cfg(weights_file)
        
        #initialize tracker
        class_thresh, obs_model, meas_cov  = validate_tracking_params(weights_file, 
                                                                      val_dataset)
        self.tracker = Tracker(meas_cov, obs_model, use_hmm=True)
        self.classnames = ["background", "person", "crutches", "walking_frame", "wheelchair", "push_wheelchair"]
        self.cla_thresholds = class_thresh
        
        self.last_received_image = None #set from image topic
        self.last_processed_image = None #set from image topic
        self.new_image = False
        
        self.cam_calib = None #set from camera info
        self.camera_frame = None #set from camera info
        
        #read rosparams
        self.fixed_frame = rospy.get_param('~fixed_frame', 'odom')
        self.tracking = rospy.get_param('~tracking', True)
        self.filter_detections = rospy.get_param('~filter_inside_boxes', True)
        self.inside_box_ratio = rospy.get_param('~inside_box_ratio', 0.8)
        camera_topic = rospy.get_param('~camera_topic', '/kinect2/qhd/image_color_rect')
        camera_info_topic = rospy.get_param('~camera_info_topic', '/kinect2/qhd/camera_info')
        
        rospy.Subscriber(camera_topic, Image, self.image_callback, queue_size=1) 
        rospy.Subscriber(camera_info_topic, CameraInfo, self.cam_info_callback, queue_size=1)
        
        #dynamic reconfigure server
        Server(TrackingParamsConfig, self.reconfigure_callback)
        
        bridge = CvBridge()
        im_width = cfg.TEST.MAX_SIZE
        im_height = cfg.TEST.SCALE
        self.viz_helper = Visualizer(len(self.classnames))
        self.publisher = Publisher(self.classnames, bridge)
        self.image_handler = ImageHandler(bridge, im_width, im_height)
        
        self.tfl = tf.TransformListener()
    
    def reconfigure_callback(self, config, level):
        
        pos_cov_threshold = config["pos_cov_threshold"]
        mahalanobis_threshold = config["mahalanobis_max_dist"]
        euclidean_threshold = config["euclidean_max_dist"]
        
        accel_noise = config["accel_noise"]
        height_noise = config["height_noise"]
        init_vel_sigma = config["init_vel_sigma"]
        hmm_transition_prob = config["hmm_transition_prob"]
        
        use_hmm = config["use_hmm"]
        
        self.tracker.set_thresholds(pos_cov_threshold, mahalanobis_threshold, 
                                    euclidean_threshold)
        
        self.tracker.set_tracking_config(accel_noise, height_noise,
                                         init_vel_sigma, hmm_transition_prob,
                                         use_hmm)
        
        return config
    
    def get_trafo_odom_in_cam(self):
        
        trafo_odom_in_cam = None
        
        if self.camera_frame is not None:
            
            try:
                time = self.last_processed_image.header.stamp
                self.tfl.waitForTransform(self.camera_frame, self.fixed_frame, time, rospy.Duration(0.5))
                pos, quat = self.tfl.lookupTransform(self.camera_frame, self.fixed_frame, time)
                
                trans = tf.transformations.translation_matrix(pos)
                rot = tf.transformations.quaternion_matrix(quat)
                
                trafo_odom_in_cam = np.dot(trans, rot)
            
            except (Exception) as e:
                print e
        
        else:
            rospy.logerr("camera frame not set, cannot get trafo between camera and fixed frame")
        
        return trafo_odom_in_cam

    def get_detections(self, image):
        
        cls_boxes, cls_depths, cls_segms, cls_keyps = infer_engine.im_detect_all(
                    self.model, image, None)
        
        boxes, depths, _segms, _keyps, classes = convert_from_cls_format(cls_boxes, 
                                                                         cls_depths, 
                                                                         None, 
                                                                         None)
        detections = []
        
        for i in range(len(classes)):
            detection = {}
            
            detection["bbox"] = boxes[i, :4]
            detection["score"] = boxes[i, -1]
            detection["depth"] = depths[i]
            detection["category_id"] = classes[i]
            
            if detection["score"] > self.cla_thresholds[self.classnames[detection["category_id"]]]:
                detections.append(detection)
        
        if self.filter_detections:
            filter_inside_boxes(detections, inside_ratio_thresh = self.inside_box_ratio)
        
        return detections
    
    def update_tracker(self, detections, trafo_odom_in_cam, dt):
        
        if dt is not None:
            self.tracker.predict(dt)
            
        if (trafo_odom_in_cam is not None) and (self.cam_calib is not None):
            self.tracker.update(detections, trafo_odom_in_cam, self.cam_calib)
        
    def process_detections(self, image, detections, trafo_odom_in_cam, dt):
        
        if self.tracking:
            self.update_tracker(detections, trafo_odom_in_cam, dt)
        
        #publish messages
        self.publisher.publish_results(image, self.last_processed_image.header, 
                                       detections, self.tracker, self.cam_calib,
                                       trafo_odom_in_cam, self.fixed_frame, 
                                       tracking=self.tracking)
    
    def process_last_image(self):
        
        if self.new_image:
            
            dt = None
            if self.last_processed_image is not None:
                dt = (self.last_received_image.header.stamp - self.last_processed_image.header.stamp).to_sec()
            self.last_processed_image = self.last_received_image
            
            image = self.image_handler.get_image(self.last_processed_image)
            
            with c2_utils.NamedCudaScope(0):
                detections = self.get_detections(image)
            
            trafo_odom_in_cam = self.get_trafo_odom_in_cam()
            
            self.process_detections(image, detections, trafo_odom_in_cam, dt)
            self.new_image = False

    def get_cam_calib(self, camera_info):
        
        cam_calib = {}

        #camera calibration
        cam_calib["fx"] = camera_info.K[0]
        cam_calib["cx"] = camera_info.K[2]
        cam_calib["fy"] = camera_info.K[4]
        cam_calib["cy"] = camera_info.K[5]
        
        return cam_calib

    def cam_info_callback(self, camera_info):
        
        if self.cam_calib is None:
            rospy.loginfo("camera info received")
            self.cam_calib = self.get_cam_calib(camera_info)
            self.camera_frame = camera_info.header.frame_id

    def image_callback(self, image):
        
        self.last_received_image = image
        self.new_image = True

if __name__ == '__main__':

    rospy.init_node('mobilityaids_detector')
    det = Detector()
    
    rospy.loginfo("waiting for images ...")
    rate = rospy.Rate(30)
    
    while not rospy.is_shutdown():
        det.process_last_image()
        rate.sleep()