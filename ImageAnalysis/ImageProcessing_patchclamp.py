# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 11:52:09 2021

@author: TvdrBurgt
"""

import numpy as np
from scipy import cluster
from skimage import filters, feature, transform


class PatchClampImageProcessing:
    @staticmethod
    def makeGaussian(size=(2048,2048), mu=(1024,1024), sigma=(512,512)):
        """
        This function returns a normalized Gaussian distribution in 2D with a
        user specified center position and standarddeviation.
        """
        
        x = np.arange(0, size[0], 1, float)
        y = np.arange(0, size[1], 1, float)
        
        gauss = lambda x,mu,sigma: np.exp((-(x-mu)**2)/(2*sigma**2))
        
        xs = gauss(x, mu[0], sigma[0])
        ys = gauss(y, mu[1], sigma[1])
        
        xgrid = np.tile(xs, (size[1],1))
        ygrid = np.tile(ys, (size[0],1)).transpose()
        
        window = np.multiply(xgrid,ygrid)
        
        return window/np.sum(window)
    
    @staticmethod
    def detectPipettetip(Ia, Ib, diameter, orientation, plotflag=False):
        """ 
        Tip detection algorithm 
        input parameters:
            Ia          = previous image of pipette tip
            Ib          = current image of pipette tip
            diameter    = diameter of pipette tip in pixels
            orientation = pipette orientation in degree, clockwise calculated from
                          the horizontal pointing to the right (= 0 degree)
            plotflag    = if True it will generate figures
        output parameters:
            xpos        = x position of the pipette tip
            ypos        = y position of the pipette tip
        """
        # Gaussian blur
        LB = filters.gaussian(Ia, 1)
        RB = filters.gaussian(Ib, 1)
        
        # Image subtraction
        IB = LB - RB
        
        # Canny edge detection
        BW = feature.canny(IB, sigma=3, low_threshold=0.99, high_threshold=0, use_quantiles=True)
        
        # Hough transform
        angle_range = np.linspace(0, np.pi, 500) + np.deg2rad(orientation)
        H, T, R = transform.hough_line(BW, angle_range)
        
        # Find Hough peaks
        _, Tpeaks, Rpeaks = transform.hough_line_peaks(H,T,R, num_peaks=5, threshold=0)
        
        # Cluster peaks
        idx_lowT = np.argmin(Tpeaks)
        idx_highT = np.argmax(Tpeaks)
        initial_clusters = np.array([[Tpeaks[idx_lowT],Rpeaks[idx_lowT]], [Tpeaks[idx_highT],Rpeaks[idx_highT]]])
        data = np.transpose(np.vstack([Tpeaks,Rpeaks]))
        centroids, labels = cluster.vq.kmeans2(data, k=initial_clusters, iter=10, minit='matrix')
        centroid1, centroid2 = centroids
        
        # Find intersection between X1*cos(T1)+Y1*sin(T1)=R1 and X2*cos(T2)+Y2*sin(T2)=R2
        if centroid1[0] > centroid2[0]:
            angle1,dist1 = centroid1
            angle2,dist2 = centroid2
        else:
            angle1,dist1 = centroid2
            angle2,dist2 = centroid1
        LHS = np.array([[np.cos(angle1), np.sin(angle1)], [np.cos(angle2), np.sin(angle2)]])
        RHS = np.array([dist1, dist2])
        xpos, ypos = np.linalg.solve(LHS, RHS)
        
        # Bias correction
        H = diameter/(2*np.tan((angle1-angle2)/2))
        alpha = (angle1+angle2)/2 - np.pi/2
        xpos = xpos - H*np.cos(alpha)
        ypos = ypos - H*np.sin(alpha)
        
        return xpos, ypos
    
    @staticmethod
    def comp_variance_of_Laplacian(I):
        """
        Computes a sharpness score that is minimal for sharp images.
        """
        # average images
        I_average = filters.gaussian(I, 4)
        
        # calculate laplacian
        I_laplace = filters.laplace(I_average, 3)
        
        # calculate variance
        score = np.var(I_laplace)
        
        return score
