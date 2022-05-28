import scipy.fftpack as fftpack
import zlib
import cv2 as cv
import random
import numpy as np
from PIL import Image, ImageFilter

np.seterr(divide='ignore', invalid='ignore')

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
def noise_salt_pepper(image, prob):

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


class jpeg:

    def __init__(self, im, quants):
        self.image = im
        self.quants = quants
        self.output = None
        super().__init__()

    def encode_quant(self, quant):
        return (self.enc / quant).astype(int)

    def decode_quant(self, quant):
        return (self.encq * quant).astype(float)

    def encode_dct(self, bx, by):
        new_shape = (
            self.image.shape[0] // bx * bx,
            self.image.shape[1] // by * by,
            3
        )
        new = self.image[
              :new_shape[0],
              :new_shape[1]
              ].reshape((
            new_shape[0] // bx,
            bx,
            new_shape[1] // by,
            by,
            3
        ))
        return fftpack.dctn(new, axes=[1, 3], norm='ortho')

    def decode_dct(self, bx, by):
        return fftpack.idctn(self.decq, axes=[1, 3], norm='ortho'
                             ).reshape((
            self.decq.shape[0] * bx,
            self.decq.shape[2] * by,
            3
        ))

    def encode_zip(self):
        return zlib.compress(self.encq.astype(np.int8).tobytes())

    def decode_zip(self):
        return np.frombuffer(zlib.decompress(self.encz), dtype=np.int8).astype(float).reshape(self.encq.shape)

    def intiate(self, qscale, bx, by):
        quant = (
            (np.ones((bx, by)) * (qscale * qscale))
                .clip(-100, 100)  # to prevent clipping
                .reshape((1, bx, 1, by, 1))
        )
        self.enc = self.encode_dct(bx, by)
        self.encq = self.encode_quant(quant)
        self.encz = self.encode_zip()
        self.decz = self.decode_zip()
        self.decq = self.decode_quant(quant)
        self.dec = self.decode_dct(bx, by)
        img_bgr = ycbcr2rgb(self.dec)
        self.output = img_bgr.astype(np.uint8)
        return


def rgb2ycbcr(im_rgb):
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv.cvtColor(im_rgb, cv.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    im_ycbcr[:, :, 0] = (im_ycbcr[:, :, 0] * (235 - 16) + 16) / \
                        255.0  # to [16/255, 235/255]
    im_ycbcr[:, :, 1:] = (im_ycbcr[:, :, 1:] * (240 - 16) + 16) / \
                         255.0  # to [16/255, 240/255]
    return im_ycbcr

def ycbcr2rgb(im_ycbcr):
    im_ycbcr = im_ycbcr.astype(np.float32)
    im_ycbcr[:, :, 0] = (im_ycbcr[:, :, 0] * 255.0 - 16) / (235 - 16)  # to [0, 1]
    im_ycbcr[:, :, 1:] = (im_ycbcr[:, :, 1:] * 255.0 - 16) / (240 - 16)  # to [0, 1]
    im_ycrcb = im_ycbcr[:, :, (0, 2, 1)].astype(np.float32)
    im_rgb = cv.cvtColor(im_ycrcb, cv.COLOR_YCR_CB2RGB)
    return im_rgb

def runLengthEncoding(message):
    encoded_message = []
    i = 0

    while (i <= len(message) - 1):
        count = 1
        ch = message[i]
        j = i
        while (j < len(message) - 1):
            if (message[j] == message[j + 1]):
                count = count + 1
                j = j + 1
            else:
                break
        encoded_message.append(ch)
        encoded_message.append(count)
        i = j + 1
    return encoded_message

def runLengthDecoding(input):
    ans = []
    for i in (0, len(input) - 2, 2):

        for j in (1, input[i + 1]):
            ans.append(input[i])
    return ans

#Our holy grail Function
def jpeg_comp(image, quant_size=2, block_size=3):
    Ycr = rgb2ycbcr(image)
    obj = jpeg(Ycr, [5])
    quants = [quant_size]
    blocks = [(block_size, block_size)]
    for qscale in quants:
        for bx, by in blocks:
            obj.intiate(qscale, bx, by)
    return obj.output

# image = cv.imread('/Users/hazemkilzieh/PycharmProjects/NoiseRestoration/Images/sample.bmp')
# cv.imshow('jpeg',jpeg_comp(image))
# cv.waitKey(0)