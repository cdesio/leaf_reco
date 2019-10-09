import numpy as np
import math
import cv2.ximgproc 
from matplotlib import pylab as plt
import scipy as sp
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy.signal import find_peaks_cwt
from scipy.special import gamma

ped=np.fromfile("Pedestal/AVG_ped_run_LE.raw", dtype='uint16', sep="") # little endian (<) int 2 (i2) bytes = 16 bit signed int
std=np.fromfile("stanDevs2.raw", dtype='float32', sep="") # little endian (<) int 2 (i2) bytes = 16 bit signed int
gains=np.fromfile("FlatFieldGains.raw", dtype='float64', sep="")
gains=gains.reshape([1,4096])
ped=ped.reshape([4096,4096])
print gains[0]
ped[1022,:]=(ped[1021,:]+ped[1024,:])/2
ped[1023,:]=(ped[1022,:]+ped[1024,:])/2
ped[3389,:]=(ped[3388,:]+ped[3391,:])/2
ped[3390,:]=(ped[3389,:]+ped[3391,:])/2
ped[3446,:]=(ped[3445,:]+ped[3447,:])/2
ped[:,138]=(ped[:,137]+ped[:,141])/2
ped[:,139]=(ped[:,138]+ped[:,141])/2
ped[:,140]=(ped[:,139]+ped[:,141])/2
ped[:,2599]=(ped[:,2598]+ped[:,2602])/2
ped[:,2600]=(ped[:,2599]+ped[:,2602])/2
ped[:,2601]=(ped[:,2600]+ped[:,2602])/2
#"firmware lines"
ped[:,4095]=ped[:,4094]
ped[:,3967]=(ped[:,3966]+ped[:,3968])/2
ped[:,3839]=(ped[:,3838]+ped[:,3840])/2
ped[:,3711]=(ped[:,3710]+ped[:,3712])/2
ped[:,3583]=(ped[:,3582]+ped[:,3584])/2
ped[:,3455]=(ped[:,3454]+ped[:,3456])/2
ped[:,3327]=(ped[:,3326]+ped[:,3328])/2
ped[:,3199]=(ped[:,3198]+ped[:,3200])/2
ped[:,3071]=(ped[:,3070]+ped[:,3072])/2
ped[:,2943]=(ped[:,2942]+ped[:,2944])/2
ped[:,2815]=(ped[:,2814]+ped[:,2816])/2
ped[:,2687]=(ped[:,2686]+ped[:,2688])/2
ped[:,2559]=(ped[:,2558]+ped[:,2560])/2
ped[:,2431]=(ped[:,2430]+ped[:,2432])/2
ped[:,2303]=(ped[:,2302]+ped[:,2304])/2
ped[:,2175]=(ped[:,2174]+ped[:,2176])/2
ped[:,2047]=(ped[:,2046]+ped[:,2048])/2
ped[:,1919]=(ped[:,1918]+ped[:,1920])/2
ped[:,1791]=(ped[:,1790]+ped[:,1792])/2
ped[:,1663]=(ped[:,1662]+ped[:,1664])/2
ped[:,1535]=(ped[:,1534]+ped[:,1536])/2
ped[:,1407]=(ped[:,1406]+ped[:,1408])/2
ped[:,1279]=(ped[:,1278]+ped[:,1280])/2
ped[:,1151]=(ped[:,1150]+ped[:,1152])/2
ped[:,1023]=(ped[:,1022]+ped[:,1024])/2
ped[:,895]=(ped[:,894]+ped[:,896])/2
ped[:,767]=(ped[:,766]+ped[:,768])/2
ped[:,639]=(ped[:,638]+ped[:,640])/2
ped[:,511]=(ped[:,510]+ped[:,512])/2
ped[:,383]=(ped[:,382]+ped[:,384])/2
ped[:,255]=(ped[:,254]+ped[:,256])/2
ped[:,127]=(ped[:,126]+ped[:,128])/2

gain_mean=gains.mean()
# ped=ped*gain_mean

# ped=ped/gains
# print gains
ped_mean=np.mean(ped)
ped_std=np.std(ped)
std_mean=np.mean(std)

print std.shape
std_std=np.std(std)
upper_ped=ped_mean+4*ped_std
lower_ped=ped_mean-6*ped_std
upper_std=std_mean+3*std_std
lower_std=std_mean-3*std_std
print ("mean",ped_mean,"std",ped_std,"lower",lower_ped,"upper",upper_ped)
print ("mean",std_mean,"std",std_std,"lower",lower_std,"upper",upper_std)
ped=ped.reshape([4096*4096,])
ped = np.array(ped, dtype = np.uint16)

ped.tofile("newped.raw")
over_6000_ped=np.argwhere( ped > 4950)
under_1200_ped=np.argwhere( ped <2400 )
over_20_std=np.argwhere( std > 20 )
under_5_std=np.argwhere( std < 5 )
# print over_5000
where_low_std=np.where( std < 5 )[0]
where_high_std=np.where(std > 20)[0]
where_low_std=np.where(  ped <2400 )[0]
where_high_std=np.where( ped > 4955 )[0]
badpix=np.zeros(4096*4096)

# print std
# print len(where_high_std)
std_mat=(std<5)*1
where_std=np.argwhere(std_mat>0.5)
# print len(where_std)
badpix1 =(( std <5 )*1)
badpix2 =((std > 20)*1)
badpix3 =((ped <2400 )*1)
badpix4 =(( ped > 4955 )*1)

# print badpix1
# print badpix2
# print badpix3
# print badpix4

badpix_tot = badpix1+badpix2+badpix3+badpix4

# print badpix_tot
where_badpix=np.argwhere(badpix_tot>0.5)
print len(where_badpix)
badpix_final=(badpix_tot>0.5)*1
badpix_final = np.array(badpix_final, dtype = np.uint16)
badpix_final.tofile("badpix.raw")
# print badpix_final.dtype

# print std_mat
# print std_mat>0.5
# print ( std < 5 )*1
# print (std > 30)*1
# print (ped <1200 )*1
# print ( ped > 6000 )*1
# badpix=(( std < 5 )*1+(std > 30)*1+(ped <1200 )*1+( ped > 6000 )*1)/4
# print len(np.argwhere(badpix==0))
# print np.any(np.array([ped>6000,ped<1200]))
# print np.where(np.any(np.array([ped>6000,ped<1200])))
# for jj in range(0,len(wherestd)):
# 	badpix[wherestd[jj]]=1

# for jj in range(0,len(wherestd)):
# 	badpix[wherestd[jj]]=1

# for jj in range(0,len(wherestd)):
# 	badpix[wherestd[jj]]=1

# for jj in range(0,len(wherestd)):
# 	badpix[wherestd[jj]]=1

print ("ped>6000",len(over_6000_ped))
print ("std>20",len(over_20_std))
print ("ped<1200",len(under_1200_ped))
print ("std<5",len(under_5_std))
# print (len(over_6000_ped)+len(over_20_std)+len(under_1200_ped)+len(under_5_std))
# ped=np.ravel(ped)
# print std
plt.subplot(2,1,1)
plt.hist(ped,300, range=(0,7000))
plt.subplot(2,1,2)
plt.hist(std,200,color='r', range=(0,20))
# plt.hist2d(ped,std,bins=2000,cmax=10)
plt.pause(0.01)
raw_input("enter")