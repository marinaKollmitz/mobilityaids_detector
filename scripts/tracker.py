# -*- coding: utf-8 -*-
import numpy as np
from EKF_with_HMM import EKF_with_HMM
from scipy.stats import multivariate_normal

class Tracker:
    def __init__(self, hmm_observation_model):
        self.tracks = []
        self.pos_cov_limit = 1.0
        self.chi2_thresh = 7.815 #threshold for Mahalanobis distance
        self.eucl_thresh = 1.25 #threshold for Eucledian distance
        self.HMM_observation_model = hmm_observation_model
        self.curr_id = 0
    
    def predict(self, dt):
        for track in self.tracks:
            if track.sigma[0,0] > self.pos_cov_limit or track.sigma[1,1] > self.pos_cov_limit or track.hmm.get_max_class() is 0:
                print "deleting track", track.track_id
                self.tracks.remove(track)
            else:
                track.predict(dt)
    
    def update(self, detections, trafo_odom_in_cam, cam_calib):
        #calculate pairwise mahalanobis distance
        assignment_profit = np.zeros([len(self.tracks), len(detections)])
        trafo_cam_in_odom = np.linalg.inv(trafo_odom_in_cam)
        
        for i, detection in enumerate(detections):
            for j,track in enumerate(self.tracks):
                z_exp = track.get_z_exp(trafo_odom_in_cam, cam_calib)
                H = track.get_H(trafo_odom_in_cam, cam_calib)

                z =  np.array([[detection['im_x']], [detection['im_y']], [detection['depth']]])
                v = z - z_exp
                
                S = H.dot(track.sigma).dot(np.transpose(H)) + track.R
                mahalanobis_d = np.transpose(v).dot(np.linalg.inv(S)).dot(v)
                
                x = np.squeeze(v)
                mu = np.array([0.0, 0.0, 0.0])
                pdf = multivariate_normal.pdf(x, mu, S)
                
                assignment_profit[j,i] = pdf
                
                odom_det = EKF_with_HMM.get_odom_detection(detection, trafo_cam_in_odom, cam_calib)
                eucl_distance = np.hypot(odom_det["x"] - track.mu[0], odom_det["y"] - track.mu[1])
 
#                print "mahalanobis_d", mahalanobis_d[0,0]
#                print "eucl distance", eucl_distances[j,i]
#                
#                mu = np.array([[track.mu[0,0]], [track.mu[1,0]], [track.mu[2,0]]])
#                prob_dens = get_prob_density(mu, np.array([odom_det["x"], odom_det["y"], odom_det["z"]]), track.sigma[0:3,0:3])
                
                if mahalanobis_d >  self.chi2_thresh:
                    print "mahalanobis too big: ", mahalanobis_d
                    assignment_profit[j,i] = -1

                if eucl_distance > self.eucl_thresh:
                    print "eucl too big: ", eucl_distance
                    assignment_profit[j,i] = -1

                    
        detection_assignments = -1 * np.ones(len(detections), np.int)

        #pair each detection to the closest track
        for i,odom_det in enumerate(detections):
            max_profit = 0
            for j,track in enumerate(self.tracks):
                if assignment_profit[j,i] > max_profit:
                    detection_assignments[i] = j
                    max_profit = assignment_profit[j,i]
        
        for i,detection in enumerate(detections):
            #if detection was paired, update tracker
            if detection_assignments[i] != -1:
                #detection was paired, update tracker
                tracker = self.tracks[detection_assignments[i]]
                tracker.update(detection, trafo_odom_in_cam, cam_calib)
                print("update tracker", tracker.track_id)
            
            else: 
                #start new tracker
                print("detection not matched, start new KF")
                track = EKF_with_HMM(detection, trafo_odom_in_cam, cam_calib, self.HMM_observation_model, self.curr_id)
                self.curr_id += 1
                self.tracks.append(track)
        
        for track in self.tracks:
            #apply background detection if not detected
            if not track.updated:
                track.hmm.update(0)