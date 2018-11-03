import numpy as np
from HMM import HMM

class EKF_with_HMM:

    def __init__(self, measurement, trafo_odom_in_cam, cam_calib, dt, hmm_observation_model, track_id):

        self.track_id = track_id
        
        accel_noise = 0.5 #noise in acceleration: meters/sec^2
        
        trafo_cam_in_odom = np.linalg.inv(trafo_odom_in_cam)
        odom_det = self.get_odom_detection(measurement, trafo_cam_in_odom, cam_calib)
        
        #state: x_odom, y_odom, z_odom, vel_x_odom, vel_y_odom
        self.mu = np.array([odom_det["x"], odom_det["y"], odom_det["z"], [0.0], [0.0]])
        
        #motion noise
        self.Q = np.array( [[(dt**2),     0.0,     0.0, (dt),  0.0],
                             [   0.0, (dt**2),     0.0,  0.0, (dt)],
                             [   0.0,     0.0, (dt**2),  0.0,  0.0],
                             [  (dt),     0.0,     0.0,  1.0,  0.0],
                             [   0.0,    (dt),     0.0,  0.0,  1.0]])*(accel_noise**2);
        
        #motion model
        self.A = np.matrix( [ [1, 0, 0, dt, 0], 
                              [0, 1, 0, 0, dt], 
                              [0, 0, 1, 0,  0],
                              [0, 0, 0, 1,  0], 
                              [0, 0, 0, 0,  1] ])
        
        #measurement_noise
        self.R = np.array([[104.6025,  0.9303, 0.0139],
                            [  0.9303, 41.2974, 0.0323],
                            [  0.0139,  0.0323, 0.1177]])
        
        #state uncertainty
        self.sigma = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.1, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.1]])
        
        #initialize state uncertainty of pose from measurement uncertainty
        H = self.get_H(trafo_odom_in_cam, cam_calib)
        H_inv = np.linalg.inv(H[0:3,0:3])
        self.sigma[0:3,0:3] = H_inv.dot(self.R).dot(H_inv.T)
        
        #initialize HMM
        self.hmm = HMM(hmm_observation_model)
        self.hmm.update(measurement['class'])
        
        self.updated = True
        
    def get_H(self, trafo_odom_in_cam, cam_calib):
        
        fx = cam_calib['fx']
        fy = cam_calib['fy']
        
        #get the partial derivatives
        T = trafo_odom_in_cam
        T11 = T[0,0]
        T12 = T[0,1]
        T13 = T[0,2]
        T14 = T[0,3]
        T21 = T[1,0]
        T22 = T[1,1]
        T23 = T[1,2]
        T24 = T[1,3]
        T31 = T[2,0]
        T32 = T[2,1]
        T33 = T[2,2]
        T34 = T[2,3]
        
        xw = self.mu[0,0]
        yw = self.mu[1,0]
        zw = self.mu[2,0]
        
        d_imx_dx = (T11*fx)/(T34 + T31*xw + T32*yw + T33*zw) - (T31*fx*(T14 + T11*xw + T12*yw + T13*zw))/(T34 + T31*xw + T32*yw + T33*zw)**2
        d_imx_dy = (T12*fx)/(T34 + T31*xw + T32*yw + T33*zw) - (T32*fx*(T14 + T11*xw + T12*yw + T13*zw))/(T34 + T31*xw + T32*yw + T33*zw)**2
        d_imx_dz = (T13*fx)/(T34 + T31*xw + T32*yw + T33*zw) - (T33*fx*(T14 + T11*xw + T12*yw + T13*zw))/(T34 + T31*xw + T32*yw + T33*zw)**2
        
        d_imy_dx = (T21*fy)/(T34 + T31*xw + T32*yw + T33*zw) - (T31*fy*(T24 + T21*xw + T22*yw + T23*zw))/(T34 + T31*xw + T32*yw + T33*zw)**2
        d_imy_dy = (T22*fy)/(T34 + T31*xw + T32*yw + T33*zw) - (T32*fy*(T24 + T21*xw + T22*yw + T23*zw))/(T34 + T31*xw + T32*yw + T33*zw)**2
        d_imy_dz = (T23*fy)/(T34 + T31*xw + T32*yw + T33*zw) - (T33*fy*(T24 + T21*xw + T22*yw + T23*zw))/(T34 + T31*xw + T32*yw + T33*zw)**2
        
        d_depth_dx = T31
        d_depth_dy = T32
        d_depth_dz = T33
        
        #Jacobian of h(x)
        H = np.array([[  d_imx_dx,   d_imx_dy,   d_imx_dz, 0.0, 0.0],
                      [  d_imy_dx,   d_imy_dy,   d_imy_dz, 0.0, 0.0],
                      [d_depth_dx, d_depth_dy, d_depth_dz, 0.0, 0.0]])
        
        return H
    
    @staticmethod
    def get_odom_detection(detection, trafo_cam_in_odom, cam_calib):
        
        fx = cam_calib['fx']
        fy = cam_calib['fy']
        cx = cam_calib['cx']
        cy = cam_calib['cy']
        
        #project detection into camera frame
        xc = (detection["im_x"] - cx)/fx * detection["depth"]
        yc = (detection["im_y"] - cy)/fy * detection["depth"]
        zc = detection["depth"]
    
        #transform into world frame
        Xc = np.array([[xc],[yc],[zc],[1]])
        Xw = trafo_cam_in_odom.dot(Xc)
        
        odom_det = {}
        odom_det["x"] = Xw[0]
        odom_det["y"] = Xw[1]
        odom_det["z"] = Xw[2]
        
        return odom_det
    
    def get_z_exp(self, trafo_odom_in_cam, cam_calib):
        
        fx = cam_calib['fx']
        fy = cam_calib['fy']
        cx = cam_calib['cx']
        cy = cam_calib['cy']
        
        #get the partial derivatives
        T = trafo_odom_in_cam
        T11 = T[0,0]
        T12 = T[0,1]
        T13 = T[0,2]
        T14 = T[0,3]
        T21 = T[1,0]
        T22 = T[1,1]
        T23 = T[1,2]
        T24 = T[1,3]
        T31 = T[2,0]
        T32 = T[2,1]
        T33 = T[2,2]
        T34 = T[2,3]
        
        xw = self.mu[0,0]
        yw = self.mu[1,0]
        zw = self.mu[2,0]
        
        im_x = cx + (fx*(T14 + T11*xw + T12*yw + T13*zw))/(T34 + T31*xw + T32*yw + T33*zw)
        im_y = cy + (fy*(T24 + T21*xw + T22*yw + T23*zw))/(T34 + T31*xw + T32*yw + T33*zw)
        depth = T34 + T31*xw + T32*yw + T33*zw
        
        return np.array([[im_x], [im_y], [depth]])
    
    def predict(self):
        
        #this is linear in our case, can update like KF
        self.mu = self.A.dot(self.mu)
        self.sigma = self.A.dot(self.sigma).dot(np.transpose(self.A)) + self.Q
        
        self.hmm.predict()
        self.updated = False

    def update(self, measurement, trafo_odom_in_cam, cam_calib):
        
        #measurement for Kalman filter
        z = np.array([[measurement['im_x']], [measurement['im_y']], [measurement['depth']]])
        H = self.get_H(trafo_odom_in_cam, cam_calib)
        
#        print "H", H
        
        # Kalman Gain
        K_tmp = H.dot(self.sigma).dot(np.transpose(H)) + self.R
        K = self.sigma.dot(np.transpose(H)).dot(np.linalg.inv(K_tmp))
        
#        print "Kalman Gain: ", K
        
        #update mu
        z_exp = self.get_z_exp(trafo_odom_in_cam, cam_calib) #h(mu)
        self.mu = self.mu + K.dot(z-z_exp)
        
        # update covariance
        self.sigma = ( np.eye(5) - K.dot(H) ).dot(self.sigma)
        self.hmm.update(measurement["class"])