# -*- coding: utf-8 -*-

import cv2
import numpy as np

class ImageHandler:
    
    def __init__(self, cv_bridge, im_width, im_height):
        
        self.bridge = cv_bridge
        self.im_width = im_width
        self.im_height = im_height

    def convert_to_DepthJet(self, depth_image):
        
        min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(depth_image)
        
        depthJet_image = cv2.convertScaleAbs(depth_image, None, 255 / (max_val-min_val), -min_val); 
        depthJet_image = cv2.applyColorMap(depthJet_image, cv2.COLORMAP_JET)
        
        return depthJet_image
    
    def get_image(self, image_msg):
        
        image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
        
        if len(image.shape) < 3:
            #it is a 1-channel image, we assume it is a depth image
            image = self.convert_to_DepthJet(image)
        
        if "rgb" in image_msg.encoding:
            #if the encoding is rgb, change it to bgr
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #resize to network input size
        (h, w) = image.shape[:2]
        
        ratio_w = float(self.im_width)/w
        ratio_h = float(self.im_height)/h
        
        im_ratio = np.min([ratio_h, ratio_w])
        
        h_new = int(h*im_ratio)
        w_new = int(w*im_ratio)
        
        image = cv2.resize(image, (w_new, h_new))
        
        #add border to make sure the image has the correct input size while keeping the original aspect ratio
        cv2.copyMakeBorder(image, int(float(540-h_new)/2), int(float(540-h_new)/2), int(float(960-w_new)/2), int(float(960-w_new)/2), cv2.BORDER_CONSTANT)
        
        return image