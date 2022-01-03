import numpy as np # Import numpy for mathematical operations and functions
import sys # Required for starting and exiting application
import copy # Need the deepcopy function to copy entire arrays
from PIL import Image # Required to read jpeg images
import cv2

import sys
import os
from PyQt5.QtGui import *

from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

def conv2D(image,kernel):

    #flip the kernel so that convolution can be done
    kernel = np.flip(kernel,0)
    kernel = np.flip(kernel,1)

    #computing the sizes of the kernel
    ker_x = kernel.shape[0]
    ker_y = kernel.shape[1]
    kerx_by2 = (np.floor(ker_x/2)).astype(int)
    kery_by2 = (np.floor(ker_y/2)).astype(int)

    # print("input image shape")
    # print(image.shape)
    #vectorizing the algorithm for fast implementation
    #hence iflattened the kernel to a vector 
    kernel_vector = np.transpose(np.reshape(kernel,(kernel.size,1)))

    img_width = image.shape[0]
    img_height = image.shape[1]
    conv_image = np.zeros((img_width,img_height))

    #padding the image by a fixed size of 1 
    #willlead to smaller size output 
    #but seems to me better then having outputs of same size but black border
    pad_image = np.pad(image,kerx_by2,'constant',constant_values=0)

    #now we run our kernel over each of the pixel of the input image to calculate the output
    for i in range(0 + kerx_by2 , img_width + kerx_by2):
        for j in range(0 + kery_by2, img_height + kery_by2 ):

            #extract the patch of the input image overlapping the kernel in its curent position
            current_patch = pad_image[i-kerx_by2 : i+kerx_by2+1 , j-kery_by2 : j+kery_by2+1]
            current_patch_vector = np.reshape(current_patch,(kernel.size,1))
            # print(current_patch_vector.shape)

            #copy the calculatedvalue to the output image
            conv_image[i-kerx_by2][j-kery_by2] = np.dot(kernel_vector,current_patch_vector)

    return conv_image


def gamma_correction(image,val_gamma):
    return np.power(image,val_gamma)

def histogram_eq(image):

    image = image.astype(int)
    #first compute the frequncies of each of the intensity levels
    #made a list size 256 for storing the number of times each intensity value occured
    #index of list corresponds to the value of intensity
    intensity_freq = np.zeros((256,1))
    intensity_freq_output = np.zeros((256,1))

    #filling up the intensity freq list 
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            intensity_val = image[i][j]
            intensity_freq[intensity_val] = intensity_freq[intensity_val] + 1;

    #calculating teh cumulatove frequcies as required by the algoirthm
    cumulative_freq = np.zeros((256,1))
    sum =0
    for k in range(0,256):
        sum = sum + intensity_freq[k]
        cumulative_freq[k] = sum

    #calculating the total pixels in the image
    total_pixels = image.shape[0] * image.shape[1]

    #hence calculating the probabilities
    cumulative_freq_norm = 255*cumulative_freq/total_pixels    

    #construct the final image
    hist_eq_output_image = np.zeros((image.shape[0],image.shape[1]))

    #assigning values to the final image matrix as per the algorithm
    for p in range(0,image.shape[0]):
        for q in range(0,image.shape[1]):
            hist_eq_output_image[p][q] = cumulative_freq_norm[image[p][q]]

    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            intensity_val = hist_eq_output_image[i][j]
            intensity_val = intensity_val.astype(int)
            intensity_freq_output[intensity_val] = intensity_freq_output[intensity_val] + 1;


    return hist_eq_output_image, intensity_freq, intensity_freq_output


def neg_pixel(image):

    #function made to take care of teh issues when image has negative pixel values
    min_intensity = image.min()
    print(min_intensity)
    image = image - min_intensity
    max_intensity = image.max()

    image = image * 255/max_intensity


    return image

def gen_gaussian(kernel_size,sigma):

    #function to geenrate teh gaussian amtrix
    kernel = np.zeros((kernel_size,kernel_size))
    kernel_sizeby2 = kernel_size/2
    #sqared of sigma
    sigma2 = np.square(sigma)

    #computing the gaussian values for each index position in the kernel
    for i in range(kernel_size):
        for j in range(kernel_size):

            #x^2 + y^2
            kernel[i][j] = np.square(i - kernel_sizeby2) + np.square(j - kernel_sizeby2)
            # -(x^2 + y^2)
            kernel[i][j] = -kernel[i][j]
            #divide by sigma squarea
            kernel[i][j] = np.exp(kernel[i][j]/(2*sigma2))

    #divide by 2pisigma sqaurea        
    kernel = kernel/(2*3.14*sigma2)

    #normalize the kernel
    sum_all = np.sum(kernel)
    kernel = kernel/sum_all

    return kernel