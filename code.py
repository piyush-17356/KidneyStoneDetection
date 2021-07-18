import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import math

# s = r'images\final_normal1.jpeg'
s = r'images\final_stone1.jpg'
img = cv2.imread(s,0)

def Histeq(img):
    histogram = [0]*256
    for i in img:
        for j in i:
            histogram[int(j)]+=1
    cum = [histogram[0]]
    for i in range(1,len(histogram)):
    	cum.append( cum[-1]+histogram[i] )
    # cum = np.array(cum)
    num = (cum - np.min(cum)) * 255
    den = np.max(cum) - np.min(cum)
    cum = num/den
    new_img = np.zeros(img.shape)
    for i in range(len(new_img)):
        for j in range(len(new_img[0])):
            new_img[i][j] = cum[int(img[i][j])]
    return new_img
def get_Lap_kernel(kernel,inp_shape):
    m=inp_shape[0]
    n=inp_shape[1]
    lap_kernel = np.zeros((m+2,n+2))
    lap_kernel[0][0] = kernel[1][1]
    lap_kernel[0][1] = kernel[1][2]
    lap_kernel[1][0] = kernel[2][1]
    lap_kernel[1][1] = kernel[2][2]
    lap_kernel[m+1][0] = kernel[1][0]
    lap_kernel[0][n+1] = kernel[0][1]
    lap_kernel[m+1][n+1] = kernel[0][0]
    lap_kernel[m+1][1] = kernel[0][2]
    lap_kernel[1][n+1] = kernel[2][0]
    return lap_kernel

def lap(img):
    inp = img
    inp_shape = inp.shape
    inp_pad = np.zeros((inp_shape[0]+2,inp_shape[1]+2))
    for i in range(inp_shape[0]+2):
        for j in range(inp_shape[1]+2):
            if(i<inp_shape[0] and j<inp_shape[1]):
                inp_pad[i][j] = inp[i][j]
    inp_fft = np.fft.fft2(inp_pad)
    kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    lap_kernel = get_Lap_kernel(kernel,inp_shape)
    lap_kernel_fft = np.fft.fft2(lap_kernel)
    result_fft = np.multiply(inp_fft,lap_kernel_fft)
    result = np.fft.ifft2(result_fft)
    result =  inp_pad - result
    for i in range(inp_shape[0]+2):
        for j in range(inp_shape[1]+2):
            result[i][j] = min(255,result[i][j])
            result[i][j] = max(0,result[i][j])
    result=result[:inp_shape[0],:inp_shape[1]]
    return np.abs(result)
def do_fft(x):
    return np.fft.fftshift(np.fft.fft2(x))

def do_ifft(x):
    return np.fft.ifft2(np.fft.ifftshift(x))

def get_Gaussian_kernel(inp_shape):
    m=inp_shape[0]
    n=inp_shape[1]
    kernel = np.zeros(inp_shape)
    for i in range(m):
        for j in range(n):
            kernel[i][j] = math.sqrt(pow((i-m//2),2)+pow(j-n//2,2))
            kernel[i][j] = math.exp(-1*pow(kernel[i][j],2)/800)
    return kernel

def gaussian(img):
    inp = img
    inp_shape = inp.shape
    inp_fft = do_fft(inp)
    kernel_fft = get_Gaussian_kernel(inp_shape)
    result_fft = np.multiply(inp_fft,kernel_fft)
    result = do_ifft(result_fft)
    return np.abs(result)

def median_filter(img):
    img = np.pad(img, (1, 1), 'constant', constant_values=(0))
    for i in range(img.shape[0]-2):
        for j in range(img.shape[1]-2):
            # print(img[i:i+3,j:j+3])
            img[i][j] = np.median(img[i:i+3,j:j+3])
    return img

def median(img):
    im1 =  cv2.medianBlur(img,5)
    im2 =  cv2.medianBlur(im1,3)
    return im2

def segmentation(img):
    kernel = np.ones((5,5), np.uint8) 
    img_erosion = cv2.erode(img, kernel, iterations=1) 
    img_dilation = cv2.dilate(img, kernel, iterations=1) 
    return img_erosion, img_dilation

def filter_extraction(img):
    ret, thresh1 = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh1

def pre_processing(img):
    plt.figure("Pre-processing")
    plt.title("Pre-processing")
    img_rm_small = morphology.remove_small_objects(img, min_size=5, connectivity=2, in_place=False)
    plt.subplot(2,2,1)
    plt.title("Input Image")
    plt.imshow(img,'gray')
    plt.xticks([])
    plt.yticks([])

    img_ostu = filter_extraction(img_rm_small)
    plt.subplot(2,2,2)
    plt.title("Applying otsu")
    plt.imshow(img_ostu,'gray')
    plt.xticks([])
    plt.yticks([])
    
    im_floodfill = img_ostu.copy()
    h, w = img_ostu.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    img_filled = img_ostu | im_floodfill_inv
    plt.subplot(2,2,3)
    plt.title("Filling")
    plt.imshow(img_filled,'gray')
    plt.xticks([])
    plt.yticks([])

    final_pre_processing_img = np.zeros(img.shape)

    for i in range(len(img)):
        for j in range(len(img[0])):
            if(img_filled[i][j]==0):
                final_pre_processing_img[i][j]=0
            else:
                final_pre_processing_img[i][j] = img[i][j]
    plt.subplot(2,2,4)
    plt.title("Final preprocessing")
    plt.imshow(final_pre_processing_img,'gray')
    plt.xticks([])
    plt.yticks([])
    # print(final_pre_processing_img)
    plt.show()
    return final_pre_processing_img

def enhancement(img):
    
    plt.figure("Image Enchancement")
    plt.title("Image Enchancement")
    plt.subplot(3,1,1)
    plt.title("Pre-processing Image")
    plt.imshow(img,'gray')
    plt.xticks([])
    plt.yticks([])

    img_gaussian = gaussian(img)
    plt.subplot(3,1,2)
    plt.title("Applying gaussian filter")
    plt.imshow(img_gaussian,'gray')
    plt.xticks([])
    plt.yticks([])


    img_median=median_filter(img_gaussian)
    plt.subplot(3,1,3)
    plt.title("Applying median filter")
    plt.imshow(img_median,'gray')
    plt.xticks([])
    plt.yticks([])

    plt.show()
    return img_median

def detection(img):
    final_pre_processing_img = pre_processing(img)
    final_enhanced_img = enhancement(final_pre_processing_img)

    img_erosion , img_dilation =segmentation(final_enhanced_img)

    for i in range(len(img_erosion)):
        for j in range(len(img_erosion[0])):
            if(img_erosion[i][j]<190):
                img_erosion[i][j] = 0
    
    for i in range(len(img_dilation)):
        for j in range(len(img_dilation[0])):
            if(img_dilation[i][j]<190):
                img_dilation[i][j] = 0
    plt.figure("Segmentation")
    plt.title("Segmentation")
    plt.subplot(3,1,2)
    plt.title("Enhanced Image")
    plt.imshow(final_enhanced_img,'gray')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(3,1,3)
    plt.title("Dilation followed by segmentation")
    plt.imshow(img_dilation,'gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3,1,1)
    plt.title("Input Image")
    plt.imshow(img,'gray')
    plt.xticks([])
    plt.yticks([])
    
    plt.show()

detection(img)



