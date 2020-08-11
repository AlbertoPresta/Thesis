import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import time as time
import math
import sys

def gaussian_mean(kernel,seed,bandwidth):
	weights = np.exp(-1*np.linalg.norm((kernel - seed)/bandwidth,axis=1))
	mean = np.array(np.sum(weights[:,None]*kernel,axis=0)/np.sum(weights), dtype=np.int64)
	return mean

threshold = 1.0
bandwidth = 10
Bin = 40
kertype = "flat"




img = Image.open("Mean-shift/images/vip.jpg")
img.load()
img = np.array(img)

seg_img = img

rows, cols, dim = img.shape
rows
m = 1
S = 5
meandist = np.array([[1000.0 for r in range(cols)] for c in range(rows)])
labels = np.array([[-1 for r in range(cols)] for c in range(rows)])

start = time.time()
means = []
for r in range(0,rows,Bin):
	print(r)
	for c in range(0,cols,Bin):
		seed = np.array([r,c,img[r][c][0],img[r][c][1],img[r][c][2]])
		for n in range(15):
			print(n)
			x = seed[0]
			y = seed[1]
			r1 = max(0,x - Bin)
			r2 = min(r1 + Bin*2,rows)
			c1 = max(0,y-Bin)
			c2 = min(c1 + Bin*2,cols)
			kernel = []
			for i in range(r1,r2):
				for j in range(c1,c2):
					print(j)
					dc = np.linalg.norm(img[i][j] - seed[2:])
					ds = (np.linalg.norm(np.array([i,j]) - seed[:2]))*m/S
					D = np.linalg.norm([dc,ds])
					if D < bandwidth:
						kernel.append([i,j,img[i][j][0],img[i][j][1],img[i][j][2]])
			kernel = np.array(kernel)

			mean = np.mean(kernel,axis = 0,dtype = np.int64)

			# get the shift
			dc = np.linalg.norm(seed[2:] - mean[2:])
			ds = (np.linalg.norm(seed[:2] - mean[:2]))*m/S
			dsm = np.linalg.norm([dc,ds])
			seed = mean
			if dsm <= threshold:
				break
		means.append(seed)


end = time.time()
