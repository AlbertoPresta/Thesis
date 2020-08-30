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
import os


class FB_seg:

    def __init__(self, Ig, image_name, ws=25,  segn=0,  omega=.049, nonneg_constraint=True):
        self.Ig = Ig
        self.image_name = image_name
        self.ws = ws
        self.segn = segn
        self.omega = omega
        self.nonneg_constraint = nonneg_constraint






    def SHcomp(self, BinN=11):
        """
        Compute local spectral histogram using integral histograms
        :param Ig: a n-band image
        :param ws: half window size
        :param BinN: number of bins of histograms
        :return: local spectral histogram at each pixel
        """
        h, w, bn = self.Ig.shape
        # quantize values at each pixel into bin ID
        for i in range(bn):
            b_max = np.max(self.Ig[:, :, i])
            b_min = np.min(self.Ig[:, :, i])
            assert b_max != b_min, "Band %d has only one value!" % i

            b_interval = (b_max - b_min) * 1. / BinN
            self.Ig[:, :, i] = np.floor((self.Ig[:, :, i] - b_min) / b_interval)

        self.Ig[self.Ig >= BinN] = BinN-1
        self.Ig = np.int32(self.Ig)

        # convert to one hot encoding
        one_hot_pix = []
        for i in range(bn):
            one_hot_pix_b = np.zeros((h*w, BinN), dtype=np.int32)
            one_hot_pix_b[np.arange(h*w), self.Ig[:, :, i].flatten()] = 1
            one_hot_pix.append(one_hot_pix_b.reshape((h, w, BinN)))

        # compute integral histogram
        integral_hist = np.concatenate(one_hot_pix, axis=2)

        np.cumsum(integral_hist, axis=1, out=integral_hist, dtype=np.float32)
        np.cumsum(integral_hist, axis=0, out=integral_hist, dtype=np.float32)
        # compute spectral histogram based on integral histogram
        padding_l = np.zeros((h, self.ws + 1, BinN * bn), dtype=np.int32)
        padding_r = np.tile(integral_hist[:, -1:, :], (1, self.ws, 1))

        integral_hist_pad_tmp = np.concatenate([padding_l, integral_hist, padding_r], axis=1)

        padding_t = np.zeros((self.ws + 1, integral_hist_pad_tmp.shape[1], BinN * bn), dtype=np.int32)
        padding_b = np.tile(integral_hist_pad_tmp[-1:, :, :], (self.ws, 1, 1))

        integral_hist_pad = np.concatenate([padding_t, integral_hist_pad_tmp, padding_b], axis=0)

        integral_hist_1 = integral_hist_pad[self.ws + 1 + self.ws:, self.ws + 1 + self.ws:, :]
        integral_hist_2 = integral_hist_pad[:-self.ws - self.ws - 1, :-self.ws - self.ws - 1, :]
        integral_hist_3 = integral_hist_pad[self.ws + 1 + self.ws:, :-self.ws - self.ws -1, :]
        integral_hist_4 = integral_hist_pad[:-self.ws - self.ws - 1, self.ws + 1 + self.ws:, :]

        sh_mtx = integral_hist_1 + integral_hist_2 - integral_hist_3 - integral_hist_4

        histsum = np.sum(sh_mtx, axis=-1, keepdims=True) * 1. / bn

        sh_mtx = np.float32(sh_mtx) / np.float32(histsum)

        return sh_mtx





    def SHedgeness(self, sh_mtx):
        h, w, _ = sh_mtx.shape
        edge_map = np.ones((h, w)) * -1
        for i in range(self.ws, h-self.ws-1):
            for j in range(self.ws, w-self.ws-1):
                edge_map[i, j] = np.sqrt(np.sum((sh_mtx[i - self.ws, j, :] - sh_mtx[i + self.ws, j, :])**2)
                                         + np.sum((sh_mtx[i, j - self.ws, :] - sh_mtx[i, j + self.ws, :])**2))
        return edge_map






    def Fseg(self):
        """
        Factorization based segmentation
        :param Ig: a n-band image
        :param ws: window size for local special histogram
        :param segn: number of segment. if set to 0, the number will be automatically estimated
        :param omega: error threshod for estimating segment number. need to adjust for different filter bank.
        :param nonneg_constraint: whether apply negative matrix factorization
        :return: segmentation label map
        """

        N1, N2, bn = self.Ig.shape

        self.ws = int(self.ws / 2)
        sh_mtx = self.SHcomp()
        sh_dim = sh_mtx.shape[2]

        Y = (sh_mtx.reshape((N1 * N2, sh_dim)))
        S = np.dot(Y.T, Y)
        d, v = LA.eig(S)

        d_sorted = np.sort(d)
        idx = np.argsort(d)
        k = np.abs(d_sorted)

        if self.segn == 0:  # estimate the segment number
            lse_ratio = np.cumsum(k) * 1. / (N1 * N2)
            print(np.sum(k)/(N1 * N2))
            self.segn = np.sum(lse_ratio > self.omega)
            print('Estimated segment number: %d' % self.segn)

            if self.segn <= 1:
                self.segn = 2
                print('Warning: Segment number is set to 2. May need to reduce omega for better segment number estimation.')

        dimn = self.segn

        U1 = v[:, idx[-1:-dimn-1:-1]]

        Y1 = np.dot(Y, U1)  # project features onto the subspace

        edge_map = self.SHedgeness(Y1.reshape((N1, N2, dimn)))

        edge_map_flatten = edge_map.flatten()

        Y_woedge = Y1[(edge_map_flatten >= 0) & (edge_map_flatten <= np.max(edge_map)*0.4), :]

        # find representative features using clustering
        cls_cen = np.zeros((self.segn, dimn), dtype=np.float32)
        L = np.sum(Y_woedge ** 2, axis=1)
        cls_cen[0, :] = Y_woedge[np.argmax(L), :]  # find the first initial center

        D = np.sum((cls_cen[0, :] - Y_woedge) ** 2, axis=1)
        cls_cen[1, :] = Y_woedge[np.argmax(D), :]

        cen_id = 1
        while cen_id < self.segn-1:
            cen_id += 1
            D_tmp = np.zeros((cen_id, Y_woedge.shape[0]), dtype=np.float32)
            for i in range(cen_id):
                D_tmp[i, :] = np.sum((cls_cen[i, :] - Y_woedge) ** 2, axis=1)
            D = np.min(D_tmp, axis=0)
            cls_cen[cen_id, :] = Y_woedge[np.argmax(D), :]

        D_cen2all = np.zeros((self.segn, Y_woedge.shape[0]), dtype=np.float32)
        cls_cen_new = np.zeros((self.segn, dimn), dtype=np.float32)
        is_converging = 1
        while is_converging:
            for i in range(self.segn):
                D_cen2all[i, :] = np.sum((cls_cen[i, :] - Y_woedge) ** 2, axis=1)

            cls_id = np.argmin(D_cen2all, axis=0)

            for i in range(self.segn):
                cls_cen_new[i, :] = np.mean(Y_woedge[cls_id == i, :], axis=0)

            if np.max((cls_cen_new - cls_cen)**2) < .00001:
                is_converging = 0
            else:
                cls_cen = cls_cen_new * 1.
        cls_cen_new = cls_cen_new.T

        ZZTinv = LAsci.inv(np.dot(cls_cen_new.T, cls_cen_new))
        Beta = np.dot(np.dot(ZZTinv, cls_cen_new.T), Y1.T)

        seg_label = np.argmax(Beta, axis=0)

        if self.nonneg_constraint:
            w0 = np.dot(U1, cls_cen_new)
            dnorm0 = 1

            h = Beta * 1.
            for i in range(100):
                tmp, _, _, _ = LA.lstsq(np.dot(w0.T, w0) + np.eye(self.segn) * .01, np.dot(w0.T, Y.T))
                h = np.maximum(0, tmp)
                tmp, _, _, _ = LA.lstsq(np.dot(h, h.T) + np.eye(self.segn) * .01, np.dot(h, Y))
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





    def plot_and_save_results(self, seg_out):
        print("FB_segmentation/results/_fbseg_" + os.path.basename(self.image_name))
        img = io.imread(self.image_name,as_gray = True)
        fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(12, 6))
        ax[0].imshow(img, cmap='gray')
        ax[1].imshow(seg_out, cmap='gray')
        plt.tight_layout()

        plt.savefig("FB_segmentation/results/_fbseg_" + os.path.basename(self.image_name))
        plt.show()
