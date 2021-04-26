import cv2
import numpy as np
import math
from utils import orientation_map
from utils import freqs_all
from utils import frequency
from utils import normalize
from utils import findMask
from utils import conv_gabor
import scipy.signal as signal
import scipy.ndimage
#import matplotlib.pyplot as plt
 
            
img=cv2.imread('printe6.jpg',0)
img=cv2.resize(img,(512,512))
norm=normalize(img,100,100)



w=64
mask=findMask(norm,0.5,8)
ori=orientation_map(norm,mask,w)

print('ori finish')
freq,ro=freqs_all(norm,ori,mask,w)
freq=int(freq)
print('freqs finish')


result_without = conv_gabor(norm,orient_map=ori,mask=mask,
                             frequency=freq,gabor_kernel_shape=31)

rmask=mask>0

#you can choose any threshold value
meann = (result_without[rmask].mean())*1.1

result_without2=result_without.copy()
result_without2[result_without > meann]=255
result_without2[result_without <= meann]=0


cv2.imshow('img',img)
cv2.imshow('enhanced',result_without2)

cv2.waitKey(0)
cv2.destroyAllWindows()


