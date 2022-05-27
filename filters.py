from scipy import fftpack
import matplotlib.pyplot as plt
from scipy import signal
from skimage import color, data, restoration
import cv2 as cv
import random
import numpy as np
import numpy.fft as fp
from PIL import Image, ImageFilter
from skimage.color import rgb2gray
import PyQt5
from PyQt5.QtWidgets import QMainWindow, QStatusBar, QApplication, QLabel, QRadioButton
from PyQt5.QtGui import QImage, QPixmap

np.seterr(divide='ignore', invalid='ignore')

def cv2qim(image):
    height, width, channel = image.shape
    bytesPerLine = 3 * width
    q_image = QImage(image.data,width, height, bytesPerLine, QImage.Format_RGB888)
    return q_image


#Degradation (adding noise)

#mean [0 or 1, 10] inctrements of 1 , var [0 or 1, 100] increments of 1
def noise_gaussian(image, mean, var):
    row, col, ch = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = np.array(gauss, dtype=np.uint8)
    noisy = cv.add(image, gauss)
    return noisy

#prob [0.00, 1.00] increments of 0.01
def noise_salt_pepper(img, prob):
    image = img
    threshold = 1 - prob

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):

            # return a value between [0, 1]
            rdn = random.random()

            if rdn < prob:
                image[i][j] = 0

            elif rdn > threshold:
                image[i][j] = 255

            else:
                image[i][j] = image[i][j]

    return image

#mean [0, 10] increments of 0.1
def noise_rayleigh(image, mean):
    row, col, ch = image.shape
    ray = np.random.rayleigh(scale=mean, size=(row, col, ch))
    ray = np.array(ray, dtype=np.uint8)
    noisy = cv.add(image, ray)
    return noisy

#mean [0 or 1, 10] inctrements of 0.1 , var [0 or 1, 100] increments of 1
def noise_gamma(image, mean, variance):
    row, col, ch = image.shape
    gamma = np.random.gamma(shape=mean, scale=variance, size=(row, col, ch))
    gamma = np.array(gamma, dtype=np.uint8)
    image = cv.add(image, gamma)
    return image

#mean [0, 10] increments of 0.1
def noise_exponential(image, mean):
    row, col, ch = image.shape
    expo = np.random.exponential(scale=mean, size=(row, col, ch))
    expo = np.array(expo, dtype=np.uint8)
    image = cv.add(image, expo)
    return image

# a < b, a [0, 255] and b [0, 255]
def noise_uniform(image, a, b):
    row, col, ch = image.shape
    uni = np.random.uniform(low=a, high=b, size=(row, col, ch))
    uni = np.array(uni, dtype=np.uint8)
    image = cv.add(image, uni)
    return image

#Filtering (Denoising)

#kSize = [3,5,7,...,99] increments +2 , equation (2n+1)
def median_filter(image, kSize=5):
    denoised = cv.medianBlur(image, kSize)
    return denoised

#d = [1,20] increments +1, sigmaColor = [5, 100] incremenrs +5, sigmaSpace = [5, 100] incremenrs +5
def bilateral_filter(image, d=15, sigmaColor=75, sigmaSpace=75):
    #d is the Diameter of each pixel neighborhood that is used during filtering.
    denoised = cv.bilateralFilter(image, d, sigmaColor, sigmaSpace)
    return denoised

#kSize = [3,5,7,...,99] increments +2 , equation (2n+1)
def arithmatic_mean_filter(image, kSize=5):
    Pimage = Image.fromarray(np.uint8(image)).convert('RGB')
    box_blur_kernel = np.reshape(np.ones(kSize * kSize), (kSize, kSize)) / (kSize * kSize)
    filtered = Pimage.filter(ImageFilter.Kernel((kSize, kSize), box_blur_kernel.flatten()))
    #Nimage_filtered = np.array(filtered_image)
    return np.array(filtered)

#radius = [0.1, 100] increments of +0.1
def gaussian_filter(image, radius=2):
    Pimage = Image.fromarray(np.uint8(image)).convert('RGB')
    filtered = Pimage.filter(ImageFilter.GaussianBlur(radius))
    return np.array(filtered)

#kSize = [3,5,7,9,11] increments +2 , equation (2n+1)
def max_filter(image, kSize=3):
    '''used to magnifying the white areas in an image, applicabale to very dim images'''
    Pimage = Image.fromarray(np.uint8(image)).convert('RGB')
    filtered = Pimage.filter(ImageFilter.MaxFilter(size=kSize))
    return np.array(filtered)

#kSize = [3,5,7,9,11] increments +2 , equation (2n+1)
def min_filter(image, kSize=3):
    '''used to magnifying the white areas in an image, applicabale to very dim images'''
    Pimage = Image.fromarray(np.uint8(image)).convert('RGB')
    filtered = Pimage.filter(ImageFilter.MinFilter(size=kSize))
    return np.array(filtered)

# n = [1,10] increments of +1, constant = [0.01, 0.2] increments of +0.01
def wiener_filter(image, n=5, constant=0.05):
    rng = np.random.default_rng()
    image1 = color.rgb2gray(image)
    psf = np.ones((n, n)) / n**2
    image1 = signal.convolve2d(image1, psf, 'same')
    image1 += constant * image1.std() * rng.standard_normal(image1.shape)
    image2, _ = restoration.unsupervised_wiener(image1, psf)
    return image2


#garbage in garbage out functions
def blur_kernel(image):
    image1 = 255.0 * cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gauss_kernel = np.outer(signal.gaussian(image1.shape[0], 3), signal.gaussian(image1.shape[1], 3))
    freq = fp.fft2(image1)
    freq_kernel = fp.fft2(fp.ifftshift(gauss_kernel)) #This is our H
    convolved = freq_kernel * freq #by convolution theorem
    im_blur = fp.ifft2(convolved).real
    im_blur = 255 * im_blur / np.max(im_blur)
    return im_blur
def inverse_filter(image):
    image = 255.0 * cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    epsilon = 0.000001
    gauss_kernel = np.outer(signal.gaussian(image.shape[0], 3), signal.gaussian(image.shape[1], 3))
    freq_kernel = np.fft.fft2(np.fft.ifftshift(gauss_kernel))
    freq = fp.fft2(image)
    freq_kernel = 1 / (epsilon + freq_kernel)
    convolved = freq*freq_kernel
    im_restored = fp.ifft2(convolved).real
    im_restored = 255 * im_restored / np.max(im_restored)
    return im_restored
def denoising_fft(image, keep_fraction=0.1):
    image1 = image.astype(float)
    im_fft = fftpack.fft2(image1)
    im_fft2 = im_fft
    r, c, channel = image1.shape
    im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
    im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
    im_new = fftpack.ifft2(im_fft2).real
    return np.uint8(im_new)
def notch_filter(image):
    im = np.mean(image, axis=2)/255
    F1 = fftpack.fft2((im.astype(float)))
    F2 = fftpack.fftshift(F1)
    F2[170:176, :220] = F2[170:176, 230:] = 0 #Elimnating the frequencies most likely responsible for noise(keep Low Freq)
    im1 = fftpack.ifft2(fftpack.ifftshift(F2)).real
    return im1