from __future__ import division
from os import walk
from copy import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
from skimage.io import imread
from skimage import filters
from skimage.measure import regionprops, label
from pykalman import KalmanFilter

def track(path, start_frame, duration, x, y, w):
    """
        Track a mouses position in an open field test
    
    Parameters:
        path = path to AVI video
    
        start_frame = frame at which to start tracking
        
        duration = tracking duration in seconds
    
        x = x position of upper left of box floor
    
        y = y position of upper left of box floor
    
        w = width of box in pixels
    
    Returns:
        mouse's position at each frame as a numpy array with x as column one 
        and y as column two. Units are pixels
    """
    cap = cv2.VideoCapture(path)
    i = 0
    centroids = [[0., 0.]]
    while(i < start_frame + 24*duration):
        # Take each frame
        _, frame = cap.read()

        if i >= start_frame:
            # Convert BGR to grayscale
            bw = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Crop image to behavior box
            im = bw[y - int(w*0.1) : y + int(w*1.1), x - int(w*0.1) : x + int(w*1.1)]
            #im = bw[y:(y+w), x:(x+w)]
            im[:int(w*0.1), :] = 255
            im[int(w*0.1) + w:] = 255
            im[:,:int(w*0.1)] = 255
            im[:,int(w*0.1) + w:] = 255
            
            centroids.append(mouse_centroid(im, centroids[i-start_frame]))

        i += 1

    centroids = np.array(centroids)[1:,:]

    # apply kalman filter
    kalman_filter = kalman(centroids)
    filtered_state_means, _ = kalman_filter.smooth(centroids)
    return filtered_state_means

def kalman(centroids):
    """
    Generate kalman filter for "track" method
        centroids = mouse centroid positions as a time series
        
    Returns:
        A kalman filter intance
    """
    Transition_Matrix = [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]
    Observation_Matrix = [[1,0,0,0],[0,1,0,0]]

    xinit = centroids[0,0]
    yinit = centroids[0,1]
    vxinit = centroids[1,0] - centroids[0,0]
    vyinit = centroids[1,1] - centroids[0,1]
    initstate = [xinit,yinit,vxinit,vyinit]
    initcovariance = 1.0e-3*np.eye(4)
    transistionCov = 1.0e-2*np.eye(4)
    observationCov = 1.0e-1*np.eye(2)
    
    kf=KalmanFilter(transition_matrices=Transition_Matrix,
                observation_matrices=Observation_Matrix,
                initial_state_mean=initstate,
                initial_state_covariance=initcovariance,
                transition_covariance=transistionCov,
                observation_covariance=observationCov)
    
    return kf

def mouse_centroid(im, previous_centroid):
    """
    Find mouse's centroid in a single image
    
    Parameters:
        im = image of analyze (numpy array)
        
        previous_centroid = coordinates of the mouse in the previous frame
        
    Returns:
        Coordinates of the mouse's centroid
    """
    original = copy(im)
    im = im < filters.threshold_otsu(im) * 0.2
    distance = ndimage.distance_transform_edt(im)
    centers = (distance > 0.8 * distance.max())
    if len(centers) == 0:
        return previous_centroid
    labels = label(centers)
    centroids = [r.weighted_centroid for r in regionprops(labels, original)]
    if len(centroids) == 1:
        return list(centroids[0])
    elif len(centroids) > 1:
        d = lambda a, b: ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5
        dists = [d(c, previous_centroid) for c in centroids]
        d_idx = np.array(dists == np.min(dists))
        return list(np.array(centroids)[d_idx][0])

def save_tracking_output(filtered_state_means, path):
    """
    Save output of "track" method as a CSV file
    
    Parameters:
        filtered_state_means = output of "track" method
        
        path = name of output file
    """
    x_pos = filtered_state_means[:,1]
    y_pos = filtered_state_means[:,0]

    df = pd.DataFrame(data=np.array([x_pos, y_pos]).T, columns=["x", "y"])
    df.to_csv(path)

def total_distance(filtered_state_means, w):
    """
    Calculate total distance travelled by the mouse
    
    Parameters:
        filtered_state_means = output of "track" method
        
        w = width of box in pixels
    
    Returns:
        Distance travelled in meters
    """
    x = filtered_state_means[:,1]
    y = filtered_state_means[:,0]

    pixel_distance = 0
    for i in range(1, len(x)):
        d = np.abs((x[i] - x[i-1]) / (y[i] - y[i-1]))
        pixel_distance += d

    conversion = w / 30 # pixels / cm

    meter_distance = pixel_distance / conversion / 100
    return meter_distance

def when_in_center(filtered_state_means, w):
    """
    """
    x_mouse = filtered_state_means[:,1]
    y_mouse = filtered_state_means[:,0]

    left = int(0.1*w) + w/4
    right =  int(0.1*w) + w - w/4
    top = int(0.1*w) + w/4
    bottom = int(0.1*w) + w - w/4
    
    in_center = np.zeros(len(x_mouse))
    for i in range(len(x_mouse)):
        if x_mouse[i] > left:
            if x_mouse[i] < right:
                if y_mouse[i] > top:
                    if y_mouse[i] < bottom:
                        in_center[i] = 1

    return in_center

def time_in_center(in_center):
    """
    Amount of time the mouse spent in the box center (seconds)
    """
    return sum(in_center) / 24

def center_entries(in_center):
    """
    """
    entries = 0
    for i in range(1, len(in_center)):
        if in_center[i] == 1 and in_center[i-1] == 0:
            entries += 1
    return entries

def zones_explored(filtered_state_means, w):
    """
    The box is divided into 16 square regions. This function finds
    the number of regions the mouse is found in
    """
    x_zones = [int(0.1*w), int(0.1*w) + w/4, int(0.1*w) + w/2, 
               int(0.1*w) + w - w/4, int(0.1*w) + w]
    y_zones =  [int(0.1*w), int(0.1*w) + w/4, int(0.1*w) + w/2, 
                int(0.1*w) + w - w/4, int(0.1*w) + w]
    x_mouse = filtered_state_means[:,1]
    y_mouse = filtered_state_means[:,0]

    zones = []
    for x_z in range(1, len(x_zones)):
        for i in range(len(x_mouse)):
            if x_mouse[i] >= x_zones[x_z-1] and x_mouse[i] < x_zones[x_z]:
                for y_z in range(1, len(y_zones)):
                    if y_mouse[i] >= y_zones[y_z-1] and y_mouse[i] < y_zones[y_z]:
                        if [x_z, y_z] not in zones:
                            zones.append([x_z, y_z])
    return (len(zones), np.array(zones))