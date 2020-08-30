import numpy as np
import math
from scipy import ndimage
import time
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy import linalg as LAsci
from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean
from scipy import ndimage as ndi




def log_filter( sgm, fsize):
    """
    LoG filter
    :param sgm: sigma in Gaussian
    :param fsize: filter size, [h, w]
    :return: LoG filter
    """
    wins_x = int(fsize[1] / 2)
    wins_y = int(fsize[0] / 2)

    out = np.zeros(fsize, dtype=np.float32)

    for x in range(-wins_x, wins_x+1):
        for y in range(-wins_y, wins_y+1):
            out[wins_y+y, wins_x+x] = - 1. / (math.pi * sgm**4.) * (1. - (x*x+y*y)/(2.*sgm*sgm)) * math.exp(-(x*x+y*y)/(2.*sgm*sgm))

    return out-np.mean(out)

def gabor_filter(sgm, theta):
    """
    Gabor filter
    :param sgm: sigma in Gaussian
    :param theta: direction
    :return: gabor filter
    """
    phs=0
    gamma=1
    wins=int(math.floor(sgm*2))
    f=1/(sgm*2.)
    out=np.zeros((2*wins+1, 2*wins+1))

    for x in range(-wins, wins+1):
        for y in range(-wins, wins+1):
            xPrime = x * math.cos(theta) + y * math.sin(theta)
            yPrime = y * math.cos(theta) - x * math.sin(theta)
            out[wins+y, wins+x] = 1/(2*math.pi*sgm*sgm)*math.exp(-.5*((xPrime)**2+(yPrime*gamma)**2)/sgm**2)*math.cos(2*math.pi*f*xPrime+phs)
    return out-np.mean(out)


def image_filtering(img, filter_list):
    sub_img = []
    # put the originale image in the list
    sub_img.append(np.float32(img))
    for filter in filter_list:
        assert (filter[0] == 'log') | (filter[0] == 'gabor'), 'Undefined filter name. '
        if filter[0] == 'log':
            f = log_filter(filter[1], filter[2])
            tmp = ndimage.convolve(np.float32(img), f, mode='reflect')
            #tmp = np.float32(log_filter(np.float32(img),filter[1]))
            sub_img.append(tmp)
        elif filter[0] == 'gabor':
            f = gabor_filter(filter[1], filter[2])
            tmp = ndimage.convolve(np.float32(img), f, mode='reflect')
            sub_img.append(tmp)
    return np.float32(np.stack(sub_img, axis=-1))







def SHcomp(Ig, ws, BinN=11):
    """
    Compute local spectral histogram using integral histograms
    :param Ig: a n-band image
    :param ws: half window size
    :param BinN: number of bins of histograms
    :return: local spectral histogram at each pixel
    """
    h, w,bn = Ig.shape
    print("inizio")
    # quantize values at each pixel into bin ID
    for i in range(bn):
        b_max = np.max(Ig[:, :, i])
        b_min = np.min(Ig[:, :, i])
        assert b_max != b_min, "Band %d has only one value!" % i

        b_interval = (b_max - b_min) * 1. / BinN
        Ig[:, :, i] = np.floor((Ig[:, :, i] - b_min) / b_interval)

    Ig[Ig >= BinN] = BinN-1
    Ig = np.int32(Ig)
    print("ciao")
    # convert to one hot encoding
    one_hot_pix = []
    for i in range(bn):
        one_hot_pix_b = np.zeros((h*w, BinN), dtype=np.int32)
        one_hot_pix_b[np.arange(h*w), Ig[:, :, i].flatten()] = 1
        one_hot_pix.append(one_hot_pix_b.reshape((h, w, BinN)))
    print("finito fase 1")
    # compute integral histogram
    integral_hist = np.concatenate(one_hot_pix, axis=2)
    print("np.concatenate(one_hot_pix, axis=2)")
    np.cumsum(integral_hist, axis=1, out=integral_hist, dtype=np.float32)
    print("fase 1bisa")
    np.cumsum(integral_hist, axis=0, out=integral_hist, dtype=np.float32)
    print("np.cumsum(integral_hist, axis=0, out=integral_hist, dtype=np.float32)")
    # compute spectral histogram based on integral histogram
    padding_l = np.zeros((h, ws + 1, BinN * bn), dtype=np.int32)
    padding_r = np.tile(integral_hist[:, -1:, :], (1, ws, 1))

    integral_hist_pad_tmp = np.concatenate([padding_l, integral_hist, padding_r], axis=1)

    padding_t = np.zeros((ws + 1, integral_hist_pad_tmp.shape[1], BinN * bn), dtype=np.int32)
    padding_b = np.tile(integral_hist_pad_tmp[-1:, :, :], (ws, 1, 1))

    integral_hist_pad = np.concatenate([padding_t, integral_hist_pad_tmp, padding_b], axis=0)

    integral_hist_1 = integral_hist_pad[ws + 1 + ws:, ws + 1 + ws:, :]
    integral_hist_2 = integral_hist_pad[:-ws - ws - 1, :-ws - ws - 1, :]
    integral_hist_3 = integral_hist_pad[ws + 1 + ws:, :-ws - ws -1, :]
    integral_hist_4 = integral_hist_pad[:-ws - ws - 1, ws + 1 + ws:, :]

    sh_mtx = integral_hist_1 + integral_hist_2 - integral_hist_3 - integral_hist_4

    histsum = np.sum(sh_mtx, axis=-1, keepdims=True) * 1. / bn

    sh_mtx = np.float32(sh_mtx) / np.float32(histsum)
    print("end")
    return sh_mtx


def SHedgeness(sh_mtx, ws):
    h, w, _ = sh_mtx.shape
    edge_map = np.ones((h, w)) * -1
    for i in range(ws, h-ws-1):
        for j in range(ws, w-ws-1):
            edge_map[i, j] = np.sqrt(np.sum((sh_mtx[i - ws, j, :] - sh_mtx[i + ws, j, :])**2)
                                     + np.sum((sh_mtx[i, j - ws, :] - sh_mtx[i, j + ws, :])**2))
    return edge_map


def Fseg(Ig, ws, segn, omega, nonneg_constraint=True):
    """
    Factorization based segmentation
    :param Ig: a n-band image
    :param ws: window size for local special histogram
    :param segn: number of segment. if set to 0, the number will be automatically estimated
    :param omega: error threshod for estimating segment number. need to adjust for different filter bank.
    :param nonneg_constraint: whether apply negative matrix factorization
    :return: segmentation label map
    """

    N1, N2, bn = Ig.shape

    ws = int(ws / 2)
    sh_mtx = SHcomp(Ig, ws)
    sh_dim = sh_mtx.shape[2]

    Y = (sh_mtx.reshape((N1 * N2, sh_dim)))
    S = np.dot(Y.T, Y)
    d, v = LA.eig(S)

    d_sorted = np.sort(d)
    idx = np.argsort(d)
    k = np.abs(d_sorted)

    if segn == 0:  # estimate the segment number
        lse_ratio = np.cumsum(k) * 1. / (N1 * N2)
        print(lse_ratio)
        print(np.sum(k)/(N1 * N2))
        segn = np.sum(lse_ratio > omega)
        print('Estimated segment number: %d' % segn)

        if segn <= 1:
            segn = 2
            print('Warning: Segment number is set to 2. May need to reduce omega for better segment number estimation.')

    dimn = segn

    U1 = v[:, idx[-1:-dimn-1:-1]]

    Y1 = np.dot(Y, U1)  # project features onto the subspace

    edge_map = SHedgeness(Y1.reshape((N1, N2, dimn)), ws)

    edge_map_flatten = edge_map.flatten()

    Y_woedge = Y1[(edge_map_flatten >= 0) & (edge_map_flatten <= np.max(edge_map)*0.4), :]

    # find representative features using clustering
    cls_cen = np.zeros((segn, dimn), dtype=np.float32)
    L = np.sum(Y_woedge ** 2, axis=1)
    cls_cen[0, :] = Y_woedge[np.argmax(L), :]  # find the first initial center

    D = np.sum((cls_cen[0, :] - Y_woedge) ** 2, axis=1)
    cls_cen[1, :] = Y_woedge[np.argmax(D), :]

    cen_id = 1
    while cen_id < segn-1:
        cen_id += 1
        D_tmp = np.zeros((cen_id, Y_woedge.shape[0]), dtype=np.float32)
        for i in range(cen_id):
            D_tmp[i, :] = np.sum((cls_cen[i, :] - Y_woedge) ** 2, axis=1)
        D = np.min(D_tmp, axis=0)
        cls_cen[cen_id, :] = Y_woedge[np.argmax(D), :]

    D_cen2all = np.zeros((segn, Y_woedge.shape[0]), dtype=np.float32)
    cls_cen_new = np.zeros((segn, dimn), dtype=np.float32)
    is_converging = 1
    while is_converging:
        for i in range(segn):
            D_cen2all[i, :] = np.sum((cls_cen[i, :] - Y_woedge) ** 2, axis=1)

        cls_id = np.argmin(D_cen2all, axis=0)

        for i in range(segn):
            cls_cen_new[i, :] = np.mean(Y_woedge[cls_id == i, :], axis=0)

        if np.max((cls_cen_new - cls_cen)**2) < .00001:
            is_converging = 0
        else:
            cls_cen = cls_cen_new * 1.
    cls_cen_new = cls_cen_new.T

    ZZTinv = LAsci.inv(np.dot(cls_cen_new.T, cls_cen_new))
    Beta = np.dot(np.dot(ZZTinv, cls_cen_new.T), Y1.T)

    seg_label = np.argmax(Beta, axis=0)

    if nonneg_constraint:
        w0 = np.dot(U1, cls_cen_new)
        dnorm0 = 1

        h = Beta * 1.
        for i in range(100):
            tmp, _, _, _ = LA.lstsq(np.dot(w0.T, w0) + np.eye(segn) * .01, np.dot(w0.T, Y.T))
            h = np.maximum(0, tmp)
            tmp, _, _, _ = LA.lstsq(np.dot(h, h.T) + np.eye(segn) * .01, np.dot(h, Y))
            w = np.maximum(0, tmp)
            w = w.T * 1.

            d = Y.T - np.dot(w, h)
            dnorm = np.sqrt(np.mean(d * d))
            print(i, np.abs(dnorm - dnorm0), dnorm)
            if np.abs(dnorm - dnorm0) < .1:
                break

            w0 = w * 1.
            dnorm0 = dnorm * 1.

        seg_label = np.argmax(h, axis=0)

    return seg_label.reshape((N1, N2))







time0 = time.time()
    # an example of using Fseg
    # read image
#img = io.imread('M2.pgm',as_gray=True)

img = io.imread('Segmentation/images/vip5.jpg',as_gray = True)
img[:,:].shape
img= resize(img, (img.shape[0]//3, img.shape[1]//3),
anti_aliasing=True)
    # define filter bank and apply to image. for color images, convert rgb to grey scale and then apply filter bank
filter_list = [('log', .5,[3,3]), ('log', 1,[5,5]),
                   ('gabor', 1.5, 0), ('gabor', 1.5, math.pi/2), ('gabor', 1.5, math.pi/4), ('gabor', 1.5, -math.pi/4),
                   ('gabor', 2.5, 0), ('gabor', 2.5, math.pi/2), ('gabor', 2.5, math.pi/4), ('gabor', 2.5, -math.pi/4)
                   ]





Ig = image_filtering(img, filter_list=filter_list)



plt.imshow(Ig[:,:,0])

#goal è avere (818 1106,3,10)



# include original image as one band
#Ig = np.concatenate((np.float32(img.reshape((img.shape[0], img.shape[1], img.shape[2]))), filter_out), axis=2)

seg_out = Fseg(Ig, ws=25, segn=2, omega=.089, nonneg_constraint=True)

print('FSEG runs in %0.2f seconds. ' % (time.time() - time0))

    # show results
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(12, 6))
ax[0].imshow(img, cmap='gray')
ax[1].imshow(seg_out, cmap='gray')
plt.tight_layout()
plt.show()
