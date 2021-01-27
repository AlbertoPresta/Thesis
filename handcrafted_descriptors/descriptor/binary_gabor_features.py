import numpy as np
import math
from math import exp
import cv2
from numpy.fft import fft2, ifft2




def crop_image(img_path,new_width = 400, new_height = 400):
    img = cv2.imread(img_path)
    print("original_shape: ",img.shape)
    img = cv2.resize(img,(1000,1000))
    #plt.imshow(img)
    #print(img.shape)
    height, width,_ = img.shape
    height = height//2
    width = width//2

    left = (width - new_width)
    top = (height - new_height)
    right = (width + new_width)
    bottom = (height + new_height)
    print(left,right,top,bottom)

    img = img[top:bottom,left:right,:]
    return img


def bitget(number, pos):
    res = np.zeros(pos.shape)
    for i,p in enumerate(pos):
        res[i] = (number >> p) & 1
    return np.array(res)[::-1]


def find_ind(lista, value):
    res = []
    for i,l in enumerate(lista):
        if(l==value):
            res.append(i)
    return res

def preprocess_images(img):
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean)/std



def gabor_filter_fixed_scale(mask, x, y, fftRows, fftCols, noOrientatations, ratio, sigma, lamb):

    gaborFirstScale = np.zeros((fftRows , fftCols, noOrientatations),dtype = "complex_")

    for oriIndex in np.arange(noOrientatations):
        theta = math.pi / noOrientatations * (oriIndex - 1)
        x_theta = x*math.cos(theta)+y*math.sin(theta)
        y_theta=-x*math.sin(theta)+y*math.cos(theta)


        gb =  np.multiply(np.exp(-.5*(np.power(x_theta,2)/sigma**2+np.power(y_theta,2)/(ratio * sigma)**2)),np.cos(np.multiply(2*math.pi/lamb,x_theta)))

        #print(gb.shape)

        total = np.sum(np.sum(np.multiply(gb,mask)))
        meanInner = total / np.sum(np.sum(mask))
        gb = gb - np.mean(meanInner)
        evenGabor = np.multiply(gb,mask,dtype = "complex_")

        gb = np.multiply(np.exp(-.5*(np.power(x_theta,2)/sigma**2+np.power(y_theta,2)/(ratio * sigma)**2)),np.sin(np.multiply(2*math.pi/lamb,x_theta)))
        #print(gb.shape)
        total = np.sum(np.sum(np.multiply(gb,mask)))
        meanInner = total / np.sum(np.sum(mask))
        gb = gb - np.mean(meanInner)
        oddGabor = np.multiply(gb,mask)

        evenGabor = np.array(evenGabor,dtype = "complex_" )
        oddGabor =  np.array(oddGabor,dtype = "complex_")

        res =  np.array(evenGabor + np.multiply(1j,oddGabor,dtype="complex_"),dtype = "complex_")

        gaborFirstScale[:,:,oriIndex] = fft2(evenGabor + np.multiply(1j,oddGabor,dtype="complex_"),s = (fftRows, fftCols))
    return gaborFirstScale






def gaborArray(rows, cols, halfLength = 17, noOrientatations = 8):
    #these two variables are used to extend the Gabor filter for the fft use.
    fftRows = rows + (halfLength * 2 + 1) - 1
    fftCols = cols + (halfLength * 2 + 1) -1
    xmax = halfLength;
    xmin = -halfLength;
    ymax = halfLength;
    ymin = -halfLength;
    x,y = np.meshgrid(np.arange(xmin,xmax + 1 ),np.arange(ymin,ymax + 1))
    ratio = 1.82
    mask = np.ones((halfLength * 2 + 1, halfLength * 2 + 1),dtype = "complex_")
    for i in range(halfLength * 2 + 1):
        for j in range(halfLength * 2 + 1):
            if ((i - halfLength)**2 + (j - halfLength)**2 > halfLength ** 2) :
                mask[i,j] = 0
    # scale of gabor filters
    gaborFirstScale = gabor_filter_fixed_scale(mask, x, y, fftRows, fftCols, noOrientatations, 1.82, 0.7, 1.3)
    gaborSecondScale = gabor_filter_fixed_scale(mask, x, y, fftRows, fftCols, noOrientatations, 1.82, 2.5, 5.2)
    gaborThirdScale = gabor_filter_fixed_scale(mask, x, y, fftRows, fftCols, noOrientatations, 1.82, 4.5, 22)
    gaborFFT = {}
    gaborFFT[1] = gaborFirstScale
    gaborFFT[2] = gaborSecondScale
    gaborFFT[3] = gaborThirdScale
    return gaborFFT




def GetMaping(bitNo):
    bins = np.uint32(2**bitNo)
    mapping = np.zeros([bins,1])
    labelMap = []
    tag = 2
    for index in range(bins):
        bitsForThisNumber = bitget(index, np.arange(bitNo))
        largestNumber = index
        for shiftIndex  in range(bitNo):
            shiftedBits = np.roll(bitsForThisNumber,shiftIndex)
            thisNum = 0
            for i in range(bitNo):
                thisNum = thisNum + (2**(i))*shiftedBits[i]
            if thisNum > largestNumber:
                largestNumber = thisNum
        if(len(labelMap) == 0):
            labelMap.append([largestNumber] + [1])
            mapping[index] = 1
        else:
            row = find_ind(np.asarray(labelMap)[:,0], largestNumber)
            if(len(row)== 0):
                labelMap.append([largestNumber] + [tag])
                mapping[index] = tag
                tag = tag + 1
            else:
                mapping[index] = labelMap[row[0]][1]
    return mapping






def BGF(img, center_x, center_y, size, halflngt = 17, all = False):
    if (all == False):
        print("SONO QUA")
        left_x = max(0,center_y - size)
        right_x = min(center_y + size,img.shape[0])
        left_y = max(0,center_x - size)
        right_y = min(center_x + size,img.shape[1])
        img = img[left_x:right_x, left_y: right_y]
        #print(left_x," ",right_x," ",left_y," ",right_y )
    img = preprocess_images(img)
    mapping = GetMaping(8)
    rows = img.shape[0]
    cols = img.shape[1]
    GA = gaborArray(rows, cols,halfLength = halflngt)
    halfLength = halflngt
    filterRows = 2*halfLength + 1
    filterCols = 2*halfLength + 1
    fftRows = rows + filterRows -1
    fftCols = cols + filterCols -1
    noOrientatations = 8
    imageFFT = fft2(img, s = (fftRows, fftCols))
    tHist = np.zeros([216, 1])
    evenGaborRes = np.zeros([rows - halfLength * 2, cols - halfLength * 2, 8])
    oddGaborRes = np.zeros([rows - halfLength * 2, cols - halfLength * 2, 8])
    for scaleIndex in range(1,4):
        evenHistAtThisScale = np.zeros([36, 1])
        oddHistAtThisScale = np.zeros([36, 1])
        gabor = GA[scaleIndex]

        for oriIndex in range(noOrientatations):
            gaborResponse = ifft2(np.multiply(imageFFT,gabor[:,:,oriIndex]))
            # valid part
            evenGaborRes[:,:,oriIndex] = np.real(gaborResponse[halfLength*2+1:rows+1, halfLength*2+1:cols+1])
            oddGaborRes[:,:,oriIndex] = np.imag(gaborResponse[halfLength*2+1:rows+1, halfLength*2+1:cols+1])

        evenBinaryRes = evenGaborRes > 0
        oddBinaryRes = oddGaborRes > 0

        evenNoArray = np.zeros([rows - halfLength * 2, cols - halfLength * 2])
        oddNoArray = np.zeros([rows - halfLength * 2, cols - halfLength * 2])

        for oriIndex in range(noOrientatations):
            evenNoArray = evenNoArray + np.multiply(evenBinaryRes[:,:, oriIndex],(2**(oriIndex)))
            oddNoArray = oddNoArray + np.multiply(oddBinaryRes[:,:, oriIndex],(2**(oriIndex)))
        validRows = evenNoArray.shape[0]
        validCols = evenNoArray.shape[1]

        for row in range(validRows):
            for col in range(validCols):
                number = evenNoArray[row, col]
                pattern = int(mapping[int(number)])

                evenHistAtThisScale[pattern-1] = evenHistAtThisScale[pattern-1] + 1


                number = oddNoArray[row, col]
                pattern = int(mapping[int(number)])
                oddHistAtThisScale[pattern-1] = oddHistAtThisScale[pattern-1] + 1

        evenHistAtThisScale = evenHistAtThisScale/ np.linalg.norm(evenHistAtThisScale)
        oddHistAtThisScale = oddHistAtThisScale / np.linalg.norm(oddHistAtThisScale)

        tHist[(scaleIndex - 1) * 72: (scaleIndex - 1) * 72 + 36] = evenHistAtThisScale
        tHist[(scaleIndex - 1) * 72 + 36: scaleIndex * 72] = oddHistAtThisScale
    return tHist




