import math
import numpy as np
import cv2 
import scipy.ndimage as ndimage
import scipy.signal as signal
import scipy
from math import sqrt


def findMask(image, threshold=0.1, w=32):
    #############################
    #
    #
    # find the region of intrest ( cut fingerprint from background)
    #
    ############################
    mask = np.empty(image.shape)
    height, width = image.shape
    for y in range(0, height, w):
        for x in range(0, width, w):
            block = image[y:y+w, x:x+w]
            standardDeviation = np.std(block)
            if standardDeviation < threshold:
                mask[y:y+w, x:x+w] = 0.0
            elif block.shape != (w, w):
                mask[y:y+w, x:x+w] = 0.0
            else:
                mask[y:y+w, x:x+w] = 1.0

    return mask

def conv_gabor(img,orient_map,mask,frequency=14,gabor_kernel_shape=31):
    #
    # loop on all pixels in the image and convolve it with it's angel in the orientation map
    #
    roo,coo=img.shape
    
    #to get the padding value for immage before convolving it with kernels
    pad=(gabor_kernel_shape-1)
    
    
    padded = np.pad(img, [(int(pad/2), int(pad/2)), (int(pad/2), int(pad/2))],'constant', 
                 constant_values=0)
    
    #result image
    padded=padded.astype(np.float32)
    dst=(np.zeros(padded.shape)).astype(np.float32)
    
   
    
    for r in range(int(pad/2),int(pad/2)+roo): # start from the image that inside the padded image
        for c in range(int(pad/2),int(pad/2)+coo):
            
            
            origin_y=(r-int(pad/2))
            origin_x=(c-int(pad/2))
            if mask[origin_y,origin_x]==0:
                continue
            
            
            ang=np.rad2deg(orient_map[origin_y,origin_x])
            real_angel = 90+ang if ang <90 else ang-90
            
            # bloack around the pixe to convolve it 
            block=padded[r-int(pad/2):r+int(pad/2)+1,c-int(pad/2):c+int(pad/2)+1]
            # get Gabor kernel 
            # here is my question ->> what to get the parametres values for ( lambda and gamma and phi)
            ker= cv2.getGaborKernel( (gabor_kernel_shape,gabor_kernel_shape), 4, 
                                    np.deg2rad(real_angel),
                                    frequency,1,0, ktype=cv2.CV_32F )
            ker /=ker.sum()
            
            dst[r,c]=np.sum((block*ker))
       
            
    return dst[int(pad/2):int(pad/2)+roo,int(pad/2):int(pad/2)+coo]


def normalize_pixel(x, v0, v, m, m0):
    dev_coeff = sqrt((v0 * ((x - m)**2)) / v)
    return m0 + dev_coeff if x > m else m0 - dev_coeff

def normalize(im, m0, v0):
    #normalize the image
    m = np.mean(im)
    v = np.std(im) ** 2
    (y, x) = im.shape
    normilize_image = im.copy()
    for i in range(x):
        for j in range(y):
            normilize_image[j, i] = normalize_pixel(im[j, i], v0, v, m, m0)

    return normilize_image

def freqs_all(img,ori,mask,w=128):
    
    # get the width of the fingerprint
    rows,cols=img.shape
    rotated =img.copy()
    result=[]

    for y in range(0,rows,w):
      for x in range(0,cols,w):
        if mask[y,x]==0:
            continue
          
        block=img[y:y+w,x:x+w]
        angbr=ori[y,x]
        
        angb=np.rad2deg(angbr)
        angb = 90+angb if angb <90 else angb-90
        
        fb,kk=frequency(block,angb,w)
        kk=cv2.resize(kk,(w,w))
        rotated[y:y+w,x:x+w]=kk
        if fb >0:
          result.append(fb)

    result=np.array(result)
    rr=np.mean(result)
    return rr,rotated


def frequency(block,angel,w=128):
    rotim = scipy.ndimage.rotate(block,angel,axes=(1,0),reshape = False,order = 3,mode = 'nearest')
    
    cropsze = int(np.fix(w/np.sqrt(2)))
    offset = int(np.fix((w-cropsze)/2))
    rotim = rotim[offset:offset+cropsze][:,offset:offset+cropsze]
            
    ridge_sum = np.sum(rotim, axis = 0)

 
    peaks = signal.find_peaks_cwt(ridge_sum, np.arange(1,10))
    if len(peaks) < 2:
        return -1,rotim
    else:
        f = (peaks[-1] - peaks[0]) / (len(peaks) - 1)
    if f < 3 or f > 15:
        return -1,rotim
    else:
        return f,rotim
  
    
def orientation_map(img,mask,w=32):
    rows,cols = img.shape
   
    
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    
    orinet = np.zeros((rows,cols))
    
    for y in range(1,rows,w):
        for x in range(1,cols,w):
            endy=min(y + w, rows - 1)
            endx=min(x + w , cols - 1)
            nominator=2 * sobelx[y:endy,x:endx] * sobely[y:endy,x:endx]
            denominator = sobelx[y:endy,x:endx]**2 - sobely[y:endy,x:endx]**2

            nominator=nominator.sum()
            denominator=denominator.sum()
            
            if nominator or denominator:
                angle = (math.pi + math.atan2(nominator, denominator)) / 2
                orinet[(y-1):(y-1)+w,(x-1):(x-1)+w]=angle
            else:
                orinet[(y-1):(y-1)+w,(x-1):(x-1)+w]=0
                
    return orinet
                    