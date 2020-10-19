import numpy as np
from scipy import signal
from matplotlib import pyplot
import matplotlib

# sift features
Nangles = 8
Nbins = 4
Nsamples = Nbins**2
alpha = 9.0
angles = np.array(range(Nangles))*2.0*np.pi/Nangles

angles
def gen_dgauss(sigma):
    '''
    generating a derivative of Gauss filter on both the X and Y
    Direction.//Generate the derivative of the Gaussian filter in the X and Y directions.
    '''
    fwid = np.int(2*np.ceil(sigma))
    G = np.array(range(-fwid,fwid+1))**2
    G = G.reshape((G.size,1)) + G
    G = np.exp(- G / 2.0 / sigma / sigma)
    G /= np.sum(G)
    GH,GW = np.gradient(G)
    GH *= 2.0/np.sum(np.abs(GH))
    GW *= 2.0/np.sum(np.abs(GW))
    return GH,GW

class DsiftExtractor:
    '''
         The class that does dense sift feature extractor.//Class extractor for intensive screening
    Sample Usage:
        extractor = DsiftExtractor(gridSpacing,patchSize,[optional params])
        feaArr,positions = extractor.process_image(Image)
    '''
    def __init__(self, gridSpacing, patchSize,
                 nrml_thres = 1.0,\
                 sigma_edge = 0.8,\
                 sift_thres = 0.2):
        '''
                 gridSpacing: the spacing for sampling dense descriptors//sampling interval of dense descriptors
                 patchSize: the size for each sift patch//size of each sift patch
                 Nrml_thres: low contrast normalization threshold//low contrast normalization threshold
                 Sigma_edge: the standard deviation for the gaussian smoothing//the standard deviation of Gaussian smoothing
            before computing the gradient
        sift_thres: sift thresholding (0.2 works well based on
                         Lowe's SIFT paper) // sift thresholding (0.2 based on Lowe's sift paper works well)
        '''
        self.gS = gridSpacing
        self.pS = patchSize
        self.nrml_thres = nrml_thres
        self.sigma = sigma_edge
        self.sift_thres = sift_thres
        # compute the weight contribution map
        sample_res = self.pS / np.double(Nbins)
        sample_p = np.array(range(self.pS))
        sample_ph, sample_pw = np.meshgrid(sample_p,sample_p)
        sample_ph.resize(sample_ph.size)
        sample_pw.resize(sample_pw.size)
        bincenter = np.array(range(1,Nbins*2,2)) / 2.0 / Nbins * self.pS - 0.5
        bincenter_h, bincenter_w = np.meshgrid(bincenter,bincenter)
        bincenter_h.resize((bincenter_h.size,1))
        bincenter_w.resize((bincenter_w.size,1))
        dist_ph = abs(sample_ph - bincenter_h)
        dist_pw = abs(sample_pw - bincenter_w)
        weights_h = dist_ph / sample_res
        weights_w = dist_pw / sample_res
        weights_h = (1-weights_h) * (weights_h <= 1)
        weights_w = (1-weights_w) * (weights_w <= 1)
        # weights is the contribution of each pixel to the corresponding bin center
        self.weights = weights_h * weights_w
        pyplot.imshow(self.weights)
        pyplot.show()

    def process_image(self, image, positionNormalize = True,\
                       verbose = True):
        '''
        processes a single image, return the locations
                 And the values ​​of detected SIFT features.//Process a single image, returning the position and value of the detected SIFT feature.
        image: a M*N image which is a numpy 2D array. If you
            pass a color image, it will automatically be converted
                         To a grayscale image.//An M*N image, which is a numpy two-dimensional array. If you pass a color image, it will automatically be converted to a grayscale image.
        positionNormalize: whether to normalize the positions
            to [0,1]. If False, the pixel-based positions of the
                         Top-right position of the patches is returned.//Identifies the position as [0,1]. If False, returns the pixel-based position in the upper right corner of the patch.

        Return values:
                 feaArr: the feature array, each row is a feature//feature array, each line is a feature
                 Positions: the positions of the features//features
        '''

        image = image.astype(np.double)
        if image.ndim == 3:
            # we do not deal with color images.
            image = np.mean(image,axis=2)
        # compute the grids
        H,W = image.shape
        gS = self.gS
        pS = self.pS
        print("grid_spacing: ",gS)
        print("patch_size: ",pS)
        remH = np.mod(H-pS, gS)
        remW = np.mod(W-pS, gS)
        print("remH: ",remH)
        print("remW: ",remW)
        offsetH = remH//2
        offsetW = remW//2
        gridH,gridW = np.meshgrid(range(offsetH,H-pS+1,gS), range(offsetW,W-pS+1,gS))
        print("gridWshape: ",gridW.shape)

        gridH = gridH.flatten()
        gridW = gridW.flatten()
        print("gridW: ",gridW)
        print("gridH ",gridH)
        if verbose:
            print('Image: w {}, h {}, gs {}, ps {}, nFea {}'.\
                    format(W,H,gS,pS,gridH.size))
        feaArr = self.calculate_sift_grid(image,gridH,gridW)
        feaArr = self.normalize_sift(feaArr)
        if positionNormalize:
            positions = np.vstack((gridH / np.double(H), gridW / np.double(W)))
        else:
            positions = np.vstack((gridH, gridW))
        return feaArr, positions

    def calculate_sift_grid(self,image,gridH,gridW):
        '''
        This function calculates the unnormalized sift features
                 It is called by process_image().//This function calculates the unnormalized sift feature.
                                                                                 //It is called by process_image().
        '''
        H,W = image.shape
        Npatches = gridH.size
        feaArr = np.zeros((Npatches,Nsamples*Nangles))
        print("feaArr: ",)
        # calculate gradient
        GH,GW = gen_dgauss(self.sigma)
        IH = signal.convolve2d(image,GH,mode='same')
        IW = signal.convolve2d(image,GW,mode='same')
        Imag = np.sqrt(IH**2+IW**2)
        Itheta = np.arctan2(IH,IW)
        Iorient = np.zeros((Nangles,H,W))
        for i in range(Nangles):
            Iorient[i] = Imag * np.maximum(np.cos(Itheta - angles[i])**alpha,0)
            #pyplot.imshow(Iorient[i])
            #pyplot.show()
        for i in range(Npatches):
            currFeature = np.zeros((Nangles,Nsamples))
            for j in range(Nangles):
                currFeature[j] = np.dot(self.weights,\
                        Iorient[j,gridH[i]:gridH[i]+self.pS, gridW[i]:gridW[i]+self.pS].flatten())
            feaArr[i] = currFeature.flatten()
        return feaArr

    def normalize_sift(self,feaArr):
        '''
        This function does sift feature normalization
        following David Lowe's definition (normalize length ->
        thresholding at 0.2 -> renormalize length)
        '''
        siftlen = np.sqrt(np.sum(feaArr**2,axis=1))
        hcontrast = (siftlen >= self.nrml_thres)
        siftlen[siftlen < self.nrml_thres] = self.nrml_thres
        # normalize with contrast thresholding
        feaArr /= siftlen.reshape((siftlen.size,1))
        # suppress large gradients
        feaArr[feaArr>self.sift_thres] = self.sift_thres
        # renormalize high-contrast ones
        feaArr[hcontrast] /= np.sqrt(np.sum(feaArr[hcontrast]**2,axis=1)).\
                reshape((feaArr[hcontrast].shape[0],1))
        return feaArr

class SingleSiftExtractor(DsiftExtractor):
    '''
    The simple wrapper class that does feature extraction, treating
    The whole image as a local image patch.//A simple wrapper class that treats the entire image as a partial image patch
    '''
    def __init__(self, patchSize,nrml_thres = 1.0,sigma_edge = 0.8,sift_thres = 0.2):
        # simply call the super class __init__ with a large gridSpace
        DsiftExtractor.__init__(self, patchSize, patchSize, nrml_thres, sigma_edge, sift_thres)

    def process_image(self, image):
        return DsiftExtractor.process_image(self, image, False, True)[0]

if __name__ == '__main__':
    # ignore this. I only use this for testing purpose...
    import cv2
    extractor = DsiftExtractor(8,16,1)
    image = cv2.imread("../prova2.jpg")
    image = np.mean(np.double(image),axis=2)
    feaArr,positions = extractor.process_image(image)
    #pyplot.hist(feaArr.flatten(),bins=100)
    #pyplot.imshow(feaArr[:256])
    #pyplot.plot(np.sum(feaArr,axis=0))
    pyplot.imshow(feaArr[np.random.permutation(feaArr.shape[0])[:256]])

    # test single sift extractor
    extractor = SingleSiftExtractor(16)
    feaArrSingle = extractor.process_image(image[:16,:16])
    pyplot.figure()
    pyplot.plot(feaArr[0],'r')
    pyplot.plot(feaArrSingle[0],'b')
    pyplot.show()


feaArrSingle

1550%8
feaArr.shape
