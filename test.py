"""
Requests 3 images from the camera and returns coefficients for the principal
components of the camera response function
"""
import numpy as np
import numpy.linalg as la
import cv2
import matplotlib.pyplot as pl

#get the tau function between two images, where k is relative exposure e2/e1
#im1 and im2 are arrays of pixel values
#tau is a vector mapping the brightnesses of the darker image (indices of tau)
#to those of the lighter image. im1 is darker, i.e. shorter exposure.
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

def main():
    exp1 = 1./25
    exp2 = 1./15
    exp3 = 1./10
    k1 = exp2/exp1
    k2 = exp3/exp2
    
    im1 = cv2.imread('C:\Users\Owner\Desktop\exposure25.png', 1)
    im2 = cv2.imread('C:\Users\Owner\Desktop\exposure15.png', 1)
    im3 = cv2.imread('C:\Users\Owner\Desktop\exposure10.png', 1)
    b1 = im1[:,:,0]
    b2 = im2[:,:,0]
    b3 = im3[:,:,0]
    g1 = im1[:,:,1]
    g2 = im2[:,:,1]
    g3 = im3[:,:,1]
    r1 = im1[:,:,2]
    r2 = im2[:,:,2]
    r3 = im3[:,:,2]
    
    bot = 1.
    top = 1.
    data = np.array([[0,0,1.0]])
    tau_thresh = 10;
    bot_count = 0;
    top_count = 0;
    
    while 1:
        if bot_count + top_count == tau_thresh:
            break
        elif bot <= top*k1:
            bot = bot*k2
            bot_count += 1
            top_count = 0
            top = 1.
        else:
            top = top*k1
            top_count +=1
        data = np.append(data, np.array([[top_count, bot_count, top/bot]]), axis = 0)
        
    num = data.shape[0]
    camera_response = np.zeros([num, 4])
    camera_response[:, 0] = data[:, 2]
    t1 = get_tau(r1, r2)
    t2 = get_tau(r2, r3)
    t2_inv = get_tau_inverse(t2)
    for i in xrange(num):
        camera_response[i, 1] = get_response_datapoint(t1, t2_inv, int(data[i, 0]), int(data[i, 1]))
    
    camera_response = np.append(camera_response, np.array([[0,0,0,0]]), axis = 0)
    pl.scatter(camera_response[:, 0],camera_response[:, 1])
    pl.title('Camera response for red values')
    pl.xlim(0,1)
    pl.ylim(0,1)
        
    
    
if __name__ == "__main__":
    main()