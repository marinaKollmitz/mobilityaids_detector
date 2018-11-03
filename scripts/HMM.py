#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 13:04:17 2018

@author: kollmitz
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

class HMM:
    def __init__(self, observation_model):
        
        self.num_classes = 6
        
        self.observ_model = observation_model

        tr = 2e-4 #2e-6 #2e-12
        hh = 1 - 4*tr

        self.transition_model = np.array([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000], 
                                          [0.0000, hh, tr, tr, tr, tr], 
                                          [0.0000, tr, hh, tr, tr, tr], 
                                          [0.0000, tr, tr, hh, tr, tr], 
                                          [0.0000, tr, tr, tr, hh, tr], 
                                          [0.0000, tr, tr, tr, tr, hh]])
        
        #uniform distributions over classes
        self.belief = 1./self.num_classes * np.ones(self.num_classes)
        #self.belief = np.zeros([num_classes,1])
        #self.belief[4,0] = 1
        
    def predict(self):
        self.belief = self.belief.dot(self.transition_model)
        
    def update(self, class_obs):
        
        observation = np.zeros(self.num_classes)
        observation[class_obs] = 1
        
        observ_prob = observation.dot(np.transpose(self.observ_model))
        
        #observation model
        self.belief = np.multiply(observ_prob, self.belief)
        
        #normalize
        self.belief = self.belief / sum(self.belief)
        
    def get_max_class(self):
        return np.argmax(self.belief)
    
    def get_max_score(self):
        return np.max(self.belief)