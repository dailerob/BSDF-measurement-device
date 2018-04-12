# -*- coding: utf-8 -*-
"""
Created on Sun Apr 08 19:15:11 2018

@author: Owner
"""

import numpy as np
import matplotlib.pyplot as pl

def get_observation_angle(theta_h, phi_h, theta_i, phi_i):
    eps = .0001
    if theta_i < eps:
        theta_o = 2*theta_h
        phi_o = phi_h
    elif theta_h < eps:
        theta_o = theta_i
        phi_o = phi_i + np.pi
    else:
        xy_term = np.sin(theta_i) * np.sin(theta_h) * np.cos(phi_i - phi_h)
        z_i = np.cos(theta_i)
        z_h = np.cos(theta_h)
        trig_term = xy_term + z_i*z_h
        theta_o = np.arccos(2.*trig_term*z_h-z_i)
        if theta_o < eps:
            phi_o = 0.
        else:
            x_i = np.sin(theta_i)*np.sin(phi_i)
            y_i = np.sin(theta_i)*np.cos(phi_i)
            x_h = np.sin(theta_h)*np.sin(phi_h)
            y_h = np.sin(theta_h)*np.cos(phi_h)
            x_o = 2.*trig_term*x_h-x_i
            ratio = x_o/np.sin(theta_o)
            if ratio > (1. - eps):
                phi_o = np.pi/2
            elif ratio < (-1. + eps):
                phi_o = -np.pi/2
            else:
                phi_o = np.arcsin((2.*trig_term*x_h-x_i)/np.sin(theta_o))
                y_o = 2.*trig_term*y_h-y_i
                if y_o < 0:
                    phi_o = np.pi - phi_o
    return theta_o, phi_o

def determine_validity(theta_o, phi_o, theta_i, phi_i):
    sensor_buffer = 10.*np.pi/180.
    sensor_limit = 80.*np.pi/180.
    if theta_o > np.pi/2 or theta_i > np.pi/2:
        return 0
    elif theta_o > sensor_limit:
        return 0
    elif abs(phi_o - phi_i) < sensor_buffer or abs(phi_o - phi_i) > (2*np.pi - sensor_buffer):
        return 0
    else:
        return 1
    
def get_locations(theta_i, phi_i, h_lim, contour_num, per_contour):
    eps = .000001
    theta_h_bins = np.linspace(0, h_lim, contour_num)
    phi_h_bins = np.linspace(-np.pi, np.pi, per_contour)
    observation_locations = np.empty((0,2))
    for i in xrange(contour_num):
        if theta_h_bins[i] < eps:
            theta_h = 0.
            phi_h = 0.
            theta_o, phi_o = get_observation_angle(theta_h, phi_h, theta_i, phi_i)
            observation_locations = np.array([[theta_o, phi_o]])
        else:
            for j in xrange(per_contour):
                theta_h = theta_h_bins[i]
                phi_h = phi_h_bins[j]
                theta_o, phi_o = get_observation_angle(theta_h, phi_h, theta_i, phi_i)
                if determine_validity(theta_o, phi_o, theta_i, phi_i):
                    observation_locations = np.append(observation_locations, np.array([[theta_o, phi_o]]), axis = 0)
    return observation_locations
    
def main():
    source_points = 1
    source_start = np.pi*30./180
    source_end = np.pi*80./180
    source_locations = np.append(np.transpose(np.array([np.linspace(source_start, source_end, source_points)])), np.zeros((source_points, 1)),axis = 1)
    locations = np.empty((0, 4))
    pl.polar(0, 1.)
    for i in xrange(source_points):
        theta_i = source_locations[i, 0]
        phi_i = source_locations[i, 1]
        observation_locations = get_locations(theta_i, phi_i, np.pi*80./180, 12, 40)
        current_locations = np.append(np.dot(np.ones((np.shape(observation_locations)[0], 1)), np.array([[theta_i, phi_i]])), observation_locations, axis = 1)
        locations = np.append(locations, current_locations, axis = 0)
        print np.shape(observation_locations)[0]
    for i in xrange(np.shape(locations)[0]):
        print locations[i, :]
        pl.polar(locations[i, 3], np.sin(locations[i,2]), 'ro')
    pl.polar(locations[0, 1], np.sin(locations[0,0]), 'bo')
        
if __name__ == "__main__":
    main()