import numpy as np
import ROOT
from ROOT import TFile, TH2D, TCanvas, TH1F, THStack, gROOT, TF1, TH1D, TSpectrum,TGraph,TLegend
from root_numpy import array2hist, hist2array, fill_hist
from matplotlib import pylab as plt#script for calculating the standard deviation for every pixel from a number of frames
from scipy.ndimage.filters import uniform_filter1d

def window_stdev(arr, radius):
    c1 = uniform_filter1d(arr, radius*2, axis=0, mode='constant', origin=-radius)
    c2 = uniform_filter1d(arr*arr, radius*2, axis=0, mode='constant', origin=-radius)
    return ((c2 - c1*c1)**.5)[:-radius*2+1,:-radius*2+1]

all_im = np.fromfile("Pedestal/ped_run_LE.raw", dtype='uint16', sep="") #load all frames of raw image into numpy array
pixels_x=4096
pixels_y=4096
pixels = pixels_x*pixels_y #number of pixels in Lassena

stan_dev_array=np.array([]) #create array to standard deviations for each pixel
frames=len(all_im)/pixels
all_im=all_im.reshape([frames,pixels_y,pixels_x])

all_im[:,1022,:]=(all_im[:,1021,:]+all_im[:,1024,:])/2
all_im[:,1023,:]=(all_im[:,1022,:]+all_im[:,1024,:])/2
all_im[:,3389,:]=(all_im[:,3388,:]+all_im[:,3391,:])/2
all_im[:,3390,:]=(all_im[:,3389,:]+all_im[:,3391,:])/2
all_im[:,3446,:]=(all_im[:,3445,:]+all_im[:,3447,:])/2
all_im[:,:,138]=(all_im[:,:,137]+all_im[:,:,141])/2
all_im[:,:,139]=(all_im[:,:,138]+all_im[:,:,141])/2
all_im[:,:,140]=(all_im[:,:,139]+all_im[:,:,141])/2
all_im[:,:,2599]=(all_im[:,:,2598]+all_im[:,:,2602])/2
all_im[:,:,2600]=(all_im[:,:,2599]+all_im[:,:,2602])/2
all_im[:,:,2601]=(all_im[:,:,2600]+all_im[:,:,2602])/2
#"firmware lines"
all_im[:,:,4095]=all_im[:,:,4094]
all_im[:,:,3967]=(all_im[:,:,3966]+all_im[:,:,3968])/2
all_im[:,:,3839]=(all_im[:,:,3838]+all_im[:,:,3840])/2
all_im[:,:,3711]=(all_im[:,:,3710]+all_im[:,:,3712])/2
all_im[:,:,3583]=(all_im[:,:,3582]+all_im[:,:,3584])/2
all_im[:,:,3455]=(all_im[:,:,3454]+all_im[:,:,3456])/2
all_im[:,:,3327]=(all_im[:,:,3326]+all_im[:,:,3328])/2
all_im[:,:,3199]=(all_im[:,:,3198]+all_im[:,:,3200])/2
all_im[:,:,3071]=(all_im[:,:,3070]+all_im[:,:,3072])/2
all_im[:,:,2943]=(all_im[:,:,2942]+all_im[:,:,2944])/2
all_im[:,:,2815]=(all_im[:,:,2814]+all_im[:,:,2816])/2
all_im[:,:,2687]=(all_im[:,:,2686]+all_im[:,:,2688])/2
all_im[:,:,2559]=(all_im[:,:,2558]+all_im[:,:,2560])/2
all_im[:,:,2431]=(all_im[:,:,2430]+all_im[:,:,2432])/2
all_im[:,:,2303]=(all_im[:,:,2302]+all_im[:,:,2304])/2
all_im[:,:,2175]=(all_im[:,:,2174]+all_im[:,:,2176])/2
all_im[:,:,2047]=(all_im[:,:,2046]+all_im[:,:,2048])/2
all_im[:,:,1919]=(all_im[:,:,1918]+all_im[:,:,1920])/2
all_im[:,:,1791]=(all_im[:,:,1790]+all_im[:,:,1792])/2
all_im[:,:,1663]=(all_im[:,:,1662]+all_im[:,:,1664])/2
all_im[:,:,1535]=(all_im[:,:,1534]+all_im[:,:,1536])/2
all_im[:,:,1407]=(all_im[:,:,1406]+all_im[:,:,1408])/2
all_im[:,:,1279]=(all_im[:,:,1278]+all_im[:,:,1280])/2
all_im[:,:,1151]=(all_im[:,:,1150]+all_im[:,:,1152])/2
all_im[:,:,1023]=(all_im[:,:,1022]+all_im[:,:,1024])/2
all_im[:,:,895]=(all_im[:,:,894]+all_im[:,:,896])/2
all_im[:,:,767]=(all_im[:,:,766]+all_im[:,:,768])/2
all_im[:,:,639]=(all_im[:,:,638]+all_im[:,:,640])/2
all_im[:,:,511]=(all_im[:,:,510]+all_im[:,:,512])/2
all_im[:,:,383]=(all_im[:,:,382]+all_im[:,:,384])/2
all_im[:,:,255]=(all_im[:,:,254]+all_im[:,:,256])/2
all_im[:,:,127]=(all_im[:,:,126]+all_im[:,:,128])/2

print "corrected bad rows/cols"

all_im=all_im.reshape([frames,pixels])
print "reshaped"
# stan_dev_array=all_im.std(axis=0,keepdims=True)
# stan_dev_array=window_stdev(all_im, 20)
all_im=np.transpose(all_im)
stan_dev_array=all_im.std(axis=1,keepdims=True)
all_im=([1])

stan_dev_array=np.transpose(stan_dev_array)
# for fi in frames:
# 	temp=all_im[]

print "std taken"
stan_dev_array=stan_dev_array.reshape([pixels_y,pixels_x])
print "reshaped again"
# stan_dev_right_shape=np.transpose(stan_dev_array)
print stan_dev_array.mean()
stan_dev_array = np.array(stan_dev_array, dtype = np.float32)

stan_dev_array.tofile("stanDevs2.raw")
print "to file complete"
stdf=stan_dev_array
plt.hist(stdf)
plt.pause(0.01)
raw_input("press enter")
# stan_dev_right_shape=np.flip(stan_dev_right_shape,1)

# stan_dev_array.tofile("S_devs.raw", sep="")
# fileOut=TFile("standardDevs.root","RECREATE")
# hist=TH2D("hist","hist",2400,-0.5,2399.5,2800,-0.5,2799.5)
# _=array2hist(stan_dev_right_shape,hist)

# # canv=TCanvas("c1","c1",400,400)
# hist.Draw("col2z")
# hist.Write("standardDevs")
# canv.SaveAs("dark_M005.root")



