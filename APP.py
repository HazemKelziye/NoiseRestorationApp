import PIL.Image
import cv2 as cv
from filters import *

image = cv.imread("nature.jpeg")
image = cv.resize(image, (800, 600), interpolation=cv.INTER_LINEAR)
cv.imshow('image', image)

noisy = noise_salt_pepper(image, 0.01)
noisy = cv.resize(noisy, (800, 600), interpolation=cv.INTER_LINEAR)
cv.imshow('noisy', noisy)
cv.imwrite("noisy.jpeg", noisy)
#filter_pil = PIL.Image.open("/Users/hazemkilzieh/PycharmProjects/DIP_Restoration_Reconstruction_Compression/nature.jpeg")
im = Image.open('/Users/hazemkilzieh/PycharmProjects/DIP_Restoration_Reconstruction_Compression/gaussian_noise_img.jpg')

box_blur_kernel = np.reshape(np.ones(3*3),(3,3)) / (3*3)
im1 = im.filter(ImageFilter.Kernel((3,3), box_blur_kernel.flatten()))
im.show()
im1.show()
cv.waitKey(0)

def arithmatic_mean_filter(image, kernel_size):
    return None
