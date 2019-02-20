# -*- coding: utf-8 -*-

import cv2
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image
from multiclass_tracking.viz import Visualizer
from multiclass_tracking.image_projection import ImageProjection
from mobilityaids_detector.msg import Detection, Detections

import rospy
import tf
import numpy as np

class Publisher:
    
    def __init__(self, classnames, cv_bridge):
        self.viz_helper = Visualizer(len(classnames))
        self.classnames = classnames
        self.bridge = cv_bridge
        
        #initialize publisher
        self.dets_image_pub = rospy.Publisher("~dets_image", Image, queue_size=1)
        self.tracks_image_pub = rospy.Publisher("~tracks_image", Image, queue_size=1)
        self.rviz_dets_pub = rospy.Publisher("~dets_vis", MarkerArray, queue_size=1)
        self.rviz_tracks_pub = rospy.Publisher("~tracks_vis", MarkerArray, queue_size=1)
        self.det_pub = rospy.Publisher("~detections", Detections, queue_size=1)
        self.track_pub = rospy.Publisher("~tracks", Detections, queue_size=1)
        
        self.dt = 0.1
        
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
    
    def publish_image_vis(self, image, header, detections, ros_publisher):
        
        dets_image = image.copy()
        self.mark_detections(dets_image, detections)
        
        dets_image = self.bridge.cv2_to_imgmsg(dets_image)
        dets_image.header = header
        ros_publisher.publish(dets_image)
        
    def publish_detection_msg(self, header, im_detections, publisher, 
                              positions=None, velocities=None, track_ids=None):
        
        detections = Detections()
        
        detections.header = header
        
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
                detection.position.header.stamp = header
                detection.position.header.frame_id = positions[i]["frame_id"]
                detection.position.point.x = positions[i]["x"]
                detection.position.point.y = positions[i]["y"]
                detection.position.point.z = positions[i]["z"]
                
            if velocities is not None:
                detection.velocity.header.stamp = header
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
        
    def publish_rviz_marker(self, stamp, lifetime, ros_publisher, classes, positions, 
                            pos_covariances=None, track_ids=None):
        
        markers = MarkerArray()

        for i in range(len(classes)):
            
            cla = classes[i]
            
            #setup marker
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = positions[i]["frame_id"]
            
            if track_ids is not None:
                marker.id = track_ids[i]
            else:
                marker.id = i
                
            marker.ns = "mobility_aids"
            marker.type = Marker.SPHERE
            marker.action = Marker.MODIFY
            if lifetime is not None:
                marker.lifetime = rospy.Duration(lifetime)
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
            
        ros_publisher.publish(markers)
        
    def publish_results(self, image, header, detections, tracker, cam_calib, 
                        trafo_odom_in_cam, fixed_frame, tracking = True):
        
        #image detections projected into cartesian space
        projected_det_positions = []
        for detection in detections:
            cart_det = ImageProjection.get_cart_detection(detection, cam_calib)
            cart_det["frame_id"] = header.frame_id
            projected_det_positions.append(cart_det)
        detection_classes = [detection["category_id"] for detection in detections]
        
        #publish detection images
        self.publish_image_vis(image, header, detections, self.dets_image_pub)
        
        #publish detection markers
        self.publish_rviz_marker(header, self.dt, self.rviz_dets_pub, 
                                 detection_classes, projected_det_positions)
        
        #publish detection messages
        self.publish_detection_msg(header, detections, self.det_pub, 
                                   positions=projected_det_positions)
        
        if tracking:
            
            if trafo_odom_in_cam is not None:
                #tracks projected into image space
                track_detections = tracker.get_track_detections(trafo_odom_in_cam)
        
                #positions, velocities, ids and covariance of tracks
                track_positions = tracker.get_track_positions()
                track_vels = tracker.get_track_velocities()
                track_ids = tracker.get_track_ids()
                track_covs = tracker.get_track_covariances()
                track_classes = [detection["category_id"] for detection in track_detections]
                
                for track_position in track_positions:
                    track_position["frame_id"] = fixed_frame
                
                for track_vel in track_vels:
                    track_vel["frame_id"] = fixed_frame
                
                
                #publish detection images
                self.publish_image_vis(image, header, track_detections, 
                                       self.tracks_image_pub)
                
                #publish detection markers
                self.publish_rviz_marker(header.stamp, self.dt, self.rviz_tracks_pub, 
                                         track_classes, track_positions,
                                         pos_covariances=track_covs, track_ids=track_ids)
                
                #publish detection messages
                self.publish_detection_msg(header, track_detections, 
                                           self.track_pub, positions=track_positions, 
                                           velocities=track_vels, track_ids=track_ids)