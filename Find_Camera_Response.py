"""
Requests 3 images from the camera and returns coefficients for the principal
components of the camera response function
"""
import numpy as np
import numpy.linalg as la
import cv2
import matplotlib.pyplot as pl
import csv

#k1 = np.sqrt(2.)
#k2 = 2.0
#get 3 images at exposure lengths t, t*k1, and t*k2, called i0, i1, and i2


#get the tau function between two images, where k is relative exposure e2/e1
#im1 and im2 are arrays of pixel values
#tau is a vector mapping the brightnesses of the darker image (indices of tau)
#to those of the lighter image
def get_tau(im1, im2):
    #histogram and cumulative distribution of the pixels in each image
    h1 = np.histogram(im1, 256, [0,255])[0]
    h2 = np.histogram(im2, 256, [0,255])[0]
    c1 = np.cumsum(h1)
    c2 = np.cumsum(h2)
    c1 = c1/float(c1[-1])
    c2 = c2/float(c2[-1])
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
    tau_f = tau[0:ti_end]
    n = 256
    tau_inv = np.zeros(n)
    #create inverse tau function
    for i in xrange(n):
        tau_inv[i] = np.searchsorted(tau_f, i)
    return tau_inv

#Get the value of the normalized camera response curve from the two taus
#Returns the value of the response curve at (k1^m)/(k2^n)
def get_response_datapoint(tau1, tau2_inv, m, n):
    data_point = 255
    #apply tau2 inverse n times
    for i in xrange(n):
        data_point = tau2_inv[int(data_point)]
    #apply tau1 m times
    for i in xrange(m):
        data_point = tau1[int(data_point)]
    return data_point/255.

#Seperate out the red, green, and blue channels of the image
def seperate_rgb(im):
    b = im[:,:,0]
    g = im[:,:,1]
    r = im[:,:,2]
    return r,g,b

#Finds all valid points for two ratios of exposure with a certain threshold
#of how many times a tau function can be applied
def find_valid_points(k1, k2, tau_threshold):
    data = np.array([[0,0,1.0]])
    for ind in np.ndindex(tau_threshold, tau_threshold):
        if ind[0]+ind[1] <= tau_threshold:
            B = (k1**ind[0])/(k2**ind[1])
            if B < 1.0:
                data = np.append(data, np.array([[ind[0], ind[1], B]]), axis = 0)
    return data

#Finds the camera response from three images given their exposure ratios
def single_camera_response(im1, im2, im3, k1, k2, tau_threshold):
    #First column is number of times tau1 is applied, second is number of 
    #times tau2 is applied, third is the normalized irradience
    data_1 = find_valid_points(k1, k2, tau_threshold)
    num_1 = data_1.shape[0]
    camera_response = np.zeros([num_1, 2])
    camera_response[:, 0] = data_1[:, 2]
    #map of brightnesses between images
    t1 = get_tau(im1, im2)
    t2 = get_tau(im2, im3)
    t2_inv = get_tau_inverse(t2)
    #populate response function with the valid points
    for i in xrange(num_1):
        camera_response[i, 1] = get_response_datapoint(t1, t2_inv, int(data_1[i, 0]), int(data_1[i, 1]))
    #add the point (0,0)
    camera_response = np.append(camera_response, np.array([[0,0]]), axis = 0)
    return camera_response

#This function adds datapoints with tau1 and tau2 switched for better coverage
#but this results in more noise
#def double_camera_response(im1, im2, im3, k1, k2, tau_threshold):
#    data_1 = find_valid_points(k1, k2, tau_threshold)
#    data_2 = find_valid_points(k2, k1, tau_threshold)
#    num_1 = data_1.shape[0]
#    num_2 = data_2.shape[0]
#    camera_response = np.zeros([num_1 + num_2, 2])
#    camera_response[:, 0] = np.append(data_1[:, 2], data_2[:, 2])
#    t1 = get_tau(im1, im2)
#    t2 = get_tau(im2, im3)
#    t2_inv = get_tau_inverse(t2)
#    t1_inv = get_tau_inverse(t1)
#    for i in xrange(num_1):
#        camera_response[i, 1] = get_response_datapoint(t1, t2_inv, int(data_1[i, 0]), int(data_1[i, 1]))
#    for i in xrange(num_2):
#        camera_response[num_1 + i, 1] = get_response_datapoint(t2, t1_inv, int(data_2[i, 0]), int(data_2[i, 1]))
#    camera_response = np.append(camera_response, np.array([[0,0]]), axis = 0)
#    return camera_response

#Fits the principle components to the scatter data collected previously
def fit_camera_model(camera_scatter, f0, h):
    #get the x datapoints that will be needed
    indices = np.round(camera_scatter[:, 0]*1023.)
    indices = indices.astype(np.int)
    #take the y data and remove the mean response f0
    scatter_data = camera_scatter[:, 1]
    deviation = scatter_data - f0[indices]
    model_scatter = h[:, indices]
    #find the least-squares fit of the principle components to the datapoints
    fit = np.dot(np.dot(la.inv(np.dot(model_scatter, np.transpose(model_scatter))), model_scatter),deviation)
    return fit    

def main():
    #The three exposure lengths of the cameras in seconds (unit doesn't matter)
    exp1 = 1./30
    exp2 = 1./20
    exp3 = 1./10
    #ratios of exposures
    k1 = exp2/exp1
    k2 = exp3/exp2
    #maximum times a tau function can be applied consecutively. A higher 
    #threshold amplifies noise
    tau_threshold = 6
    
    #load  the three images
    im1 = cv2.imread('C:\Users\Owner\Desktop\ex30.jpg', 1)
    im2 = cv2.imread('C:\Users\Owner\Desktop\ex20.jpg', 1)
    im3 = cv2.imread('C:\Users\Owner\Desktop\ex10.jpg', 1)
    
    #seperate the red green and blue channels
    r1, g1, b1 = seperate_rgb(im1)
    r2, g2, b2 = seperate_rgb(im2)
    r3, g3, b3 = seperate_rgb(im3)
    
    #get the response functions for each channel
    camera_response_red_scatter = single_camera_response(r1, r2, r3, k1, k2, tau_threshold)
    camera_response_green_scatter = single_camera_response(g1, g2, g3, k1, k2, tau_threshold)
    camera_response_blue_scatter = single_camera_response(b1, b2, b3, k1, k2, tau_threshold)
    
    #get the empirical model of camera response functions
    full_model = np.array([])
    with open('C:\Users\Owner\Desktop\emor.txt') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for row in reader:
            if len(filter(lambda x: x != '', row)) >= 4:
                full_model = np.append(full_model, filter(lambda x: x != '', row))
    
    #first row is the normalized irradiences (linearly spaced), second row is
    #the mean response, the other 25 rows are principle components
    full_model = np.resize(full_model.astype(np.float),(27, 256*4))
    x = full_model[0,:]
    f0 = full_model[1,:]
    h = full_model[2:,:]
    #use only the first three components to avoid over-fitting
    first_three = h[0:3, :]
    
    #get the coefficients for the principle components
    camera_fit_red = fit_camera_model(camera_response_red_scatter, f0, first_three)
    camera_fit_green = fit_camera_model(camera_response_green_scatter, f0, first_three)
    camera_fit_blue = fit_camera_model(camera_response_blue_scatter, f0, first_three)
        
    #plot the response functions
    pl.clf()
    f, (ax1, ax2, ax3) = pl.subplots(1, 3, sharex = True, sharey = True, figsize = (15, 5))
    ax1.scatter(camera_response_red_scatter[:, 0],camera_response_red_scatter[:, 1])
    ax2.scatter(camera_response_green_scatter[:, 0],camera_response_green_scatter[:, 1])
    ax3.scatter(camera_response_blue_scatter[:, 0],camera_response_blue_scatter[:, 1])
    ax1.plot(x, np.dot(np.transpose(camera_fit_red), first_three) + f0)
    ax2.plot(x, np.dot(np.transpose(camera_fit_green), first_three) + f0)
    ax3.plot(x, np.dot(np.transpose(camera_fit_blue), first_three) + f0)
    ax1.set_title('Red')
    ax2.set_title('Green')
    ax3.set_title('Blue')
    ax1.set_xlim(0,1)
    ax1.set_ylim(0,1)
    
    
if __name__ == "__main__":
    main()