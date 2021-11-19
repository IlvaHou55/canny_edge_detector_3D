#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import convolve


class cannyEdgeDetector3D:
    def __init__(self, img, sigma=1, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15):
        self.img = img
        self.img_final = []
        self.img_smoothed = None
        self.gradientMat = None
        self.phiMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.weak_voxel = weak_voxel
        self.strong_voxel = strong_voxel
        self.sigma = sigma
        self.lowThreshold = lowthreshold
        self.highThreshold = highthreshold
        return 
    
    def sobel_filters(self, img):
        Kx = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],[[2, 0, -2], [4, 0, -4], [2, 0, -2]],[[1, 0, -1], [2, 0, -2], [1, 0, -1]]], np.float32)
        Ky = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],[[2, 4, 2], [0, 0, 0], [-2, -4, -2]],[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], np.float32)
        Kz = np.array([[[1, 2, 1], [2, 4, 2], [1, 2, 1]],[[0, 0, 0], [0, 0, 0], [0, 0, 0]],[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]], np.float32)
    
        Ix = convolve(img, Kx)
        Iy = convolve(img, Ky)
        Iz = convolve(img, Kz)
    
        G = np.sqrt(Ix**2 + Iy**2 + Iz**2)
        G = G / G.max() * 255
    
        phi = np.arctan2(Iy,Ix) 
        theta = np.arctan2(Iy,Iz)
       
        return (G,phi,theta)
  
    def non_max_suppression(self, img, phi, theta):
        M, N, O = img.shape
        Z = np.zeros((M,N,O), dtype=np.int32)
        phi[phi < 0] += np.pi
        theta[theta < 0] += np.pi
    
        for i in range(1,M-1):
            for j in range(1,N-1):
                for k in range(1,O-1):
                    try:
                        q = 255
                        r = 255
                    
                        #theta 0
                        if (0 <= theta[i,j,k] < (0.125*np.pi)) or ((0.875*np.pi) <= theta[i,j,k] <= np.pi):
                            q = img[i, j, k+1]
                            r = img[i, j, k-1]
                    
                        #theta 1/4 pi
                        elif ((0.125*np.pi) <= theta[i,j,k] < (0.375*np.pi)):
                            #phi 0
                            if(0 <= phi[i,j,k] < (0.125*np.pi)) or ((0.875*np.pi) <= phi[i,j,k] <= np.pi):
                                q = img[i+1, j, k+1]
                                r = img[i-1, j, k-1]
                            
                            #phi 1/4 pi
                            elif((0.125*np.pi) <= phi[i,j,k] < (0.375*np.pi)):
                                q = img[i+1, j+1, k+1]
                                r = img[i-1, j-1, k-1]
                            
                            #phi 1/2 pi
                            elif((0.375*np.pi) <= phi[i,j,k] < (0.625*np.pi)):
                                q = img[i, j+1, k+1]
                                r = img[i, j-1, k-1]
                        
                            #phi 3/4 pi
                            elif((0.625*np.pi) <= phi[i,j,k] < (0.875*np.pi)):
                                q = img[i-1, j+1, k+1]
                                r = img[i+1, j-1, k-1]
                    
                        #theta 1/2 pi
                        elif ((0.375*np.pi) <= theta[i,j,k] < (0.625*np.pi)):
                            #phi 0
                            if(0 <= phi[i,j,k] < (0.125*np.pi)) or ((0.875*np.pi) <= phi[i,j,k] <= np.pi):
                                q = img[i+1, j, k]
                                r = img[i-1, j, k]
                            
                            #phi 1/4 pi
                            elif((0.125*np.pi) <= phi[i,j,k] < (0.375*np.pi)):
                                q = img[i+1, j+1, k]
                                r = img[i-1, j-1, k]
                            
                            #phi 1/2 pi
                            elif((0.375*np.pi) <= phi[i,j,k] < (0.625*np.pi)):
                                q = img[i, j+1, k]
                                r = img[i, j-1, k]
                        
                            #phi 3/4 pi
                            elif((0.625*np.pi) <= phi[i,j,k] < (0.875*np.pi)):
                                q = img[i-1, j+1, k]
                                r = img[i+1, j-1, k]
                    
                        #theta 3/4 pi
                        elif ((0.625*np.pi) <= theta[i,j,k] < (0.875*np.pi)):
                            #phi 0
                            if(0 <= phi[i,j,k] < (0.125*np.pi)) or ((0.875*np.pi) <= phi[i,j,k] <= np.pi):
                                q = img[i+1, j, k-1]
                                r = img[i-1, j, k+1]
                            
                            #phi 1/4 pi
                            elif((0.125*np.pi) <= phi[i,j,k] < (0.375*np.pi)):
                                q = img[i+1, j+1, k-1]
                                r = img[i-1, j-1, k+1]
                            
                            #phi 1/2 pi
                            elif((0.375*np.pi) <= phi[i,j,k] < (0.625*np.pi)):
                                q = img[i, j+1, k-1]
                                r = img[i, j-1, k+1]
                        
                            #phi 3/4 pi
                            elif((0.625*np.pi) <= phi[i,j,k] < (0.875*np.pi)):
                                q = img[i-1, j+1, k-1]
                                r = img[i+1, j-1, k+1]
                    
                    
                        if (img[i,j,k] >= q) and (img[i,j,k] >= r):
                            Z[i,j,k] = img[i,j,k]
                        else:
                            Z[i,j,k] = 0

                    except IndexError as e:
                        pass
    
        return Z
    
    def threshold(self, img):
    
        highThreshold = img.max() * self.highThresholdRatio;
        lowThreshold = highThreshold * self.lowThresholdRatio;
    
        M, N, O = img.shape
        res = np.zeros((M,N,O), dtype=np.int32)
    
        weak = np.int32(self.weak_voxel)
        strong = np.int32(self.strong_voxel)
    
        strong_i, strong_j, strong_k = np.where(img >= highThreshold)
        zeros_i, zeros_j, zeros_k = np.where(img < lowThreshold)
    
        weak_i, weak_j, weak_k = np.where((img <= highThreshold) & (img >= lowThreshold))
    
        res[strong_i, strong_j, strong_k] = strong
        res[weak_i, weak_j, weak_k] = weak
    
        return res
    
    def hysteresis(self, img):
        
        M, N, O = img.shape  
        weak = self.weak_voxel
        strong = self.weak_voxel
        
        for i in range(1, M-1):
            for j in range(1, N-1):
                for k in range(1, O-1):
                    if (img[i,j,k] == weak):
                        try:
                            if ((img[i,j,k-1] == strong) or (img[i,j,k+1] == strong) 
                                or (img[i,j-1,k] == strong) or (img[i,j-1,k-1] == strong) or (img[i,j-1,k+1] == strong)
                                or (img[i,j+1,k] == strong) or (img[i,j+1,k-1] == strong) or (img[i,j+1,k+1] == strong)
                                or (img[i-1,j,k] == strong) or (img[i-1,j,k-1] == strong) or (img[i-1,j,k+1] == strong)
                                or (img[i-1,j-1,k] == strong) or (img[i-1,j-1,k-1] == strong) or (img[i-1,j-1,k+1] == strong)
                                or (img[i-1,j+1,k] == strong) or (img[i-1,j+1,k-1] == strong) or (img[i-1,j+1,k+1] == strong)
                                or (img[i+1,j,k] == strong) or (img[i+1,j,k-1] == strong) or (img[i+1,j,k+1] == strong)
                                or (img[i+1,j-1,k] == strong) or (img[i+1,j-1,k-1] == strong) or (img[i+1,j-1,k+1] == strong)
                                or (img[i+1,j+1,k] == strong) or (img[i+1,j+1,k-1] == strong) or (img[i+1,j+1,k+1] == strong)):
                                img[i,j,k] = strong
                            else:
                                img[i,j,k] = 0
                        except IndexError as e:
                            pass
        return img
   
    def detect(self): 
        self.img_smoothed = gaussian_filter(img, self.sigma)
        self.gradientMat, self.phiMat, self.thetaMat = self.sobel_filters(self.img_smoothed)
        self.nonMaxImg = self.non_max_suppression(self.gradientMat, self.phiMat, self.thetaMat)
        self.thresholdImg = self.threshold(self.nonMaxImg)
        self.img_final = self.hysteresis(self.thresholdImg)

        return self.img_final

