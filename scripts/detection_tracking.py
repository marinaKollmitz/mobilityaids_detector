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
from multiclass_tracking.image_projection import ImageProjection

from dynamic_reconfigure.server import Server
from mobilityaids_detector.cfg import TrackingParamsConfig
from mobilityaids_detector.msg import Detection, Detections

from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker, MarkerArray

from cv_bridge import CvBridge
import cv2
import numpy as np
import rospy
import tf
import os
import inspect
import logging

#dirty hack to fix python logging with ROS
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
        self.dt = None #set from image topic timestamps
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
        camera_topic = rospy.get_param('~camera_topic', '/kinect2/qhd/image_color_rect')
        camera_info_topic = rospy.get_param('~camera_info_topic', '/kinect2/qhd/camera_info')
        
        rospy.Subscriber(camera_topic, Image, self.image_callback, queue_size=1) 
        rospy.Subscriber(camera_info_topic, CameraInfo, self.cam_info_callback, queue_size=1)
        
        #dynamic reconfigure server
        Server(TrackingParamsConfig, self.reconfigure_callback)
        
        #initialize publisher
        self.dets_image_pub = rospy.Publisher("~dets_image", Image, queue_size=1)
        self.tracks_image_pub = rospy.Publisher("~tracks_image", Image, queue_size=1)
        self.rviz_dets_pub = rospy.Publisher("~dets_vis", MarkerArray, queue_size=1)
        self.rviz_tracks_pub = rospy.Publisher("~tracks_vis", MarkerArray, queue_size=1)
        self.det_pub = rospy.Publisher("~detections", Detections, queue_size=1)
        self.track_pub = rospy.Publisher("~tracks", Detections, queue_size=1)
        
        self.viz_helper = Visualizer(len(self.classnames))
        self.bridge = CvBridge()
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
            self.filter_inside_boxes(detections)
        
        return detections
    
    def mark_detections(self, image, detections):
        
        for detection in detections:
            bbox = detection["bbox"]
            cla = detection["category_id"]
            color_box = self.viz_helper.colors_box[cla]
            color = [255*color_box[2], 255*color_box[1], 255*color_box[0]]
            cv2.rectangle(image, 
                          (int(bbox[0]), int(bbox[1])), 
                          (int(bbox[2]), int(bbox[3])), 
                          color, 3)
    
    def update_tracker(self, detections):
        
        if self.dt is not None:
            self.tracker.predict(self.dt)
            
            trafo_odom_in_cam = self.get_trafo_odom_in_cam()
            
            if (trafo_odom_in_cam is not None) and (self.cam_calib is not None):
                self.tracker.update(detections, trafo_odom_in_cam, self.cam_calib)
    
    def publish_image_vis(self, image, detections, publisher):
        
        dets_image = image.copy()
        self.mark_detections(dets_image, detections)
        
        dets_image = self.bridge.cv2_to_imgmsg(dets_image)
        dets_image.header = self.last_processed_image.header
        publisher.publish(dets_image)
    
    def publish_rviz_marker(self,  publisher, classes, positions, pos_covariances=None, track_ids=None):
        
        markers = MarkerArray()

        for i in range(len(classes)):
            
            cla = classes[i]
            
            #setup marker
            marker = Marker()
            marker.header.stamp = self.last_processed_image.header.stamp
            marker.header.frame_id = positions[i]["frame_id"]
            
            if track_ids is not None:
                marker.id = track_ids[i]
            else:
                marker.id = i
                
            marker.ns = "mobility_aids"
            marker.type = Marker.SPHERE
            marker.action = Marker.MODIFY
            if self.dt is not None:
                marker.lifetime = rospy.Duration(self.dt)
            else:
                marker.lifetime = rospy.Duration(0.1)
            #maker position
            marker.pose.position.x = positions[i]["x"]
            marker.pose.position.y = positions[i]["y"]
            marker.pose.position.z = positions[i]["z"]
            
            #marker color
            color_box = self.viz_helper.colors_box[cla]
            marker.color.b = float(color_box[2])
            marker.color.g = float(color_box[1])
            marker.color.r = float(color_box[0])
            marker.color.a = 1.0
            
            #get error ellipse
            width, height, scale, angle = 0.5, 0.5, 0.5, 0.0
            
            #if a pose covariance is given, like for tracking, plot ellipse marker
            if pos_covariances is not None:
                width, height, angle = Visualizer.get_error_ellipse(pos_covariances[i])
                angle = angle + np.pi/2
                scale = 0.1
            
            quat = tf.transformations.quaternion_from_euler(0, 0, angle)
            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]
            marker.scale.x = height
            marker.scale.y = width
            marker.scale.z = scale
            
            markers.markers.append(marker)
            
        publisher.publish(markers)
    
    def publish_detection_msg(self, im_detections, publisher, positions=None, velocities=None, track_ids=None):
        
        detections = Detections()
        
        detections.header = self.last_processed_image.header
        
        for i in range(len(im_detections)):
            detection = Detection()
            detection.category = self.classnames[im_detections[i]["category_id"]]
            detection.confidence = im_detections[i]["score"]
            detection.image_bbox.x_min = im_detections[i]["bbox"][0]
            detection.image_bbox.y_min = im_detections[i]["bbox"][1]
            detection.image_bbox.x_max = im_detections[i]["bbox"][2]
            detection.image_bbox.y_max = im_detections[i]["bbox"][3]
            detection.depth = im_detections[i]["depth"]
            
            if positions is not None:
                detection.position.header.stamp = self.last_processed_image.header.stamp
                detection.position.header.frame_id = positions[i]["frame_id"]
                detection.position.point.x = positions[i]["x"]
                detection.position.point.y = positions[i]["y"]
                detection.position.point.z = positions[i]["z"]
                
            if velocities is not None:
                detection.velocity.header.stamp = self.last_processed_image.header.stamp
                detection.velocity.header.frame_id = velocities[i]["frame_id"]
                detection.velocity.point.x = velocities[i]["x"]
                detection.velocity.point.y = velocities[i]["y"]
                detection.velocity.point.z = velocities[i]["z"]
            
            if track_ids is not None:
                detection.track_id = track_ids[i]
            
            else:
                detection.track_id = -1
                
            detections.detections.append(detection)
        
        publisher.publish(detections)
    
    def get_inside_ratio(self, bbox_out, bbox_in):
        
        overlap_bbox=[0,0,0,0]
        overlap_bbox[0] = max(bbox_out[0], bbox_in[0]);
        overlap_bbox[1] = max(bbox_out[1], bbox_in[1]);
        overlap_bbox[2] = min(bbox_out[2], bbox_in[2]);
        overlap_bbox[3] = min(bbox_out[3], bbox_in[3]);	
        #print bi
        overlap_width = overlap_bbox[2] - overlap_bbox[0] + 1;
        overlap_height = overlap_bbox[3] - overlap_bbox[1] + 1;
        inside_ratio = 0.0;
        if (overlap_width>0 and overlap_height>0):
            overlap_area = overlap_width*overlap_height
            bbox_in_area = (bbox_in[2] - bbox_in[0] + 1) * (bbox_in[3] - bbox_in[1] + 1)
            inside_ratio = float(overlap_area)/float(bbox_in_area);
        return inside_ratio
    
    def filter_inside_boxes(self, detections, inside_ratio_thres = 0.8):
        #filter pedestrian bounding box inside mobilityaids bounding box
        
        for outside_det in detections:
            
            # check for mobility aids bboxes
            if outside_det['category_id'] > 1:
                for inside_det in detections:
                    #check all pedestrian detections against mobility aids detection
                    if inside_det['category_id'] is 1:
                        inside_ratio = self.get_inside_ratio(outside_det['bbox'], inside_det['bbox'])
                        if inside_ratio > inside_ratio_thres:
                            rospy.logdebug("filtering pedestrian bbox inside %s bbox" % self.classnames[outside_det['category_id']])
                            detections.remove(inside_det)
    
    def publish_results(self, image, detections):
        
        #image detections projected into cartesian space
        projected_det_positions = []
        for detection in detections:
            cart_det = ImageProjection.get_cart_detection(detection, self.cam_calib)
            cart_det["frame_id"] = self.last_processed_image.header.frame_id
            projected_det_positions.append(cart_det)
        detection_classes = [detection["category_id"] for detection in detections]
        
        #publish detection images
        self.publish_image_vis(image, detections, self.dets_image_pub)
        
        #publish detection markers
        self.publish_rviz_marker(self.rviz_dets_pub, detection_classes, 
                                 projected_det_positions)
        
        #publish detection messages
        self.publish_detection_msg(detections, self.det_pub, positions=projected_det_positions)
        
        if self.tracking:
            
            trafo_odom_in_cam = self.get_trafo_odom_in_cam()
            
            if trafo_odom_in_cam is not None:
                #tracks projected into image space
                track_detections = self.tracker.get_track_detections(trafo_odom_in_cam)
        
                #positions, velocities, ids and covariance of tracks
                track_positions = self.tracker.get_track_positions()
                track_vels = self.tracker.get_track_velocities()
                track_ids = self.tracker.get_track_ids()
                track_covs = self.tracker.get_track_covariances()
                track_classes = [detection["category_id"] for detection in track_detections]
                
                for track_position in track_positions:
                    track_position["frame_id"] = self.fixed_frame
                
                for track_vel in track_vels:
                    track_vel["frame_id"] = self.fixed_frame
                
                
                #publish detection images
                self.publish_image_vis(image, track_detections, self.tracks_image_pub)
                
                #publish detection markers
                self.publish_rviz_marker(self.rviz_tracks_pub, 
                                         track_classes, track_positions,
                                         pos_covariances=track_covs, track_ids=track_ids)
                
                #publish detection messages
                self.publish_detection_msg(track_detections, self.track_pub, 
                                           positions=track_positions, 
                                           velocities=track_vels, 
                                           track_ids=track_ids)
        
    def process_detections(self, image, detections):
        
        if self.tracking:
            self.update_tracker(detections)
        
        #publish messages
        self.publish_results(image, detections)
    
    def convert_to_DepthJet(self, depth_image):
        
        min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(depth_image)
        
        depthJet_image = cv2.convertScaleAbs(depth_image, None, 255 / (max_val-min_val), -min_val); 
        depthJet_image = cv2.applyColorMap(depthJet_image, cv2.COLORMAP_JET)
        
        return depthJet_image
    
    def get_image(self, image_msg):
        
        image = self.bridge.imgmsg_to_cv2(self.last_processed_image, desired_encoding="passthrough")
        
        if len(image.shape) < 3:
            #it is a 1-channel image, we assume it is a depth image
            image = self.convert_to_DepthJet(image)
        
        if "rgb" in image_msg.encoding:
            #if the encoding is rgb, change it to bgr
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #resize to network input size
        (h, w) = image.shape[:2]
        
        ratio_w = 960./w
        ratio_h = 540./h
        
        im_ratio = np.min([ratio_h, ratio_w])
        
        h_new = int(h*im_ratio)
        w_new = int(w*im_ratio)
        
        image = cv2.resize(image, (w_new, h_new))
        
        #add border to make sure the image has the correct input size while keeping the original aspect ratio
        cv2.copyMakeBorder(image, int(float(540-h_new)/2), int(float(540-h_new)/2), int(float(960-w_new)/2), int(float(960-w_new)/2), cv2.BORDER_CONSTANT)
        
        return image
    
    def process_last_image(self):
        
        if self.new_image:
            if self.last_processed_image is not None:
                self.dt = (self.last_received_image.header.stamp - self.last_processed_image.header.stamp).to_sec()
            self.last_processed_image = self.last_received_image
            
            image = self.get_image(self.last_processed_image)
            
            with c2_utils.NamedCudaScope(0):
                detections = self.get_detections(image)
            
            self.process_detections(image, detections)
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