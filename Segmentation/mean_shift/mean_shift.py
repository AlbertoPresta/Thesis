import pandas as pd
import numpy as np
import cv2
import os


hs = 8 #spatial bandwidth
hr= 7   # range bandwidth
threshold_convergence_mean = 0.25
bandwidth=[hs,hr]



def rgb2luv(R,G,B):
    var_r = double(R)/255
    var_g = double(G)/255
    var_b = double(B)/255

    if(var_r>0.04045):
        var_r = ((var_r + 0.055) / 1.055)^2.4
    else:
        var_r = var_r / 12.92

    if(var_g > 0.04045):
        var_g = ((var_g + 0.055)/1.055)^2.4
    else:
        var_g = var_g / 12.92

    if(var_b > 0.04045):
        var_b = ((var_b + 0.055) / 1.055)^2.4
    else:
        var_b = var_b / 12.92

    var_r = var_r*100
    var_g = var_g*100
    var_b = var_b*100

    X = var_r * 0.4124 + var_g * 0.3576 + var_b * 0.1805
    Y = var_r * 0.2126 + var_g * 0.7152 + var_b * 0.0722
    Z = var_r * 0.0193 + var_g * 0.1192 + var_b * 0.9505

    var_U = ( 4 * X ) / ( X + ( 15 * Y ) + ( 3 * Z ) )
    var_V = ( 9 * Y ) / ( X + ( 15 * Y ) + ( 3 * Z ) )

    var_Y = Y / 100

    if(var_Y > 0.008856):
        var_Y = var_Y^(1/3)
    else:
        var_Y = (7.787 * var_Y) + (16/116)

    ref_X =  95.047
    ref_Y = 100.000
    ref_Z = 108.883

    ref_U = (4*ref_X )/(ref_X+( 15 * ref_Y ) + ( 3 * ref_Z ) )
    ref_V = (9*ref_Y )/(ref_X+( 15 * ref_Y ) + ( 3 * ref_Z ) )


    L1 = (116*var_Y)-16
    U1 = 13*L1*(var_U-ref_U )
    V1 = 13*L1*(var_V-ref_V )


    #L1(isnan(L1))=0
    #U1(isnan(U1))=0
    #V1(isnan(V1))=0


    L=L1
    U=U1
    V=V1

    return L,U,V









img=cv2.imread('Segmentation/mean_shift/42409.jpg')
[height,width,frame] = img.shape


width



x=np.zeros([5,height*width])

x[0,1]
for j in range(1,height):
    for l in range(1,width):
        x[0,l + width*(j-1)] = j
        x[1, l + width*(j - 1)] = l

for ii in x[0]:
    print(ii)
