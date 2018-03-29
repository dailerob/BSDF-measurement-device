"""
Requests 3 images from the camera and returns coefficients for the principal
components of the camera response function
"""
import numpy as np
import numpy.linalg as la
import cv2

#k1 = np.sqrt(2.)
#k2 = 2.0
#get 3 images at exposure lengths t, t*k1, and t*k2, called i0, i1, and i2


#get the tau function between two images, where k is relative exposure e2/e1
#im1 and im2 are arrays of pixel values
#tau is a vector mapping the brightnesses of the darker image (indices of tau)
#to those of the lighter image
def get_tau(im1, im2, k):
    #histogram and cumulative distribution of the pixels in each image
    h1 = np.histogram(im1, 256, [0,255])[0]
    h2 = np.histogram(im2, 256, [0,255])[0]
    c1 = np.cumsum(h1)
    c2 = np.cumsum(h2)
    c1 = c1/c1[-1]
    c2 = c2/c2[-1]
    n = 256
    tau = np.zeros(n)
    #map the brightnesses in image 1 to those of image 2
    for i in xrange(n):
        tau[i]= np.searchsorted(c2, c1[i])
    return tau

#Get the inverse tau function, from the bright to the darker image
#White is mapped to the darkest possible color, so the function is continuous
def get_tau_inverse(tau):
    #get pre-white part of tau
    ti_end = np.searchsorted(tau, 255)
    tau_f = tau[0:ti_end+1]
    n = 256
    tau_inv = np.zeros(n)
    #create inverse tau function
    for i in xrange(n):
        tau_inv[i] = np.searchsorted(tau_f, i/255.)
    return tau_inv

#Get the value of the normalized camera response curve from the two taus
#Returns the value of the response curve at (k1^m)/(k2^n)
def get_response_datapoint(tau1, tau2_inv, m, n, k1, k2):
    data_point = 1.0
    #apply tau2 inverse n times
    for i in xrange(n):
        data_point = tau2_inv[int(np.floor(data_point*255))]
    #apply tau1 m times
    for i in xrange(m):
        data_point = tau1[int(np.floor(data_point*255))]