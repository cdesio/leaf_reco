def gainsBadPix(origIm,frames):
  badpixmap=np.fromfile(badpixelmapFile, dtype='uint16', sep="") # little endian (<) int 2 (i2) bytes = 16 bit signed int
  gainIm=np.fromfile("gainsFile", dtype='float64', sep="")
  gainIm=gainIm.reshape([1,4096])
  pedIm=np.fromfile("Pedestal/ped_10_27_LE.raw", dtype='<i2', sep="") # little endian (<) int 2 (i2) bytes = 16 bit signed int
  pedIm=pedIm.reshape(1,pixels)
  origIm=origIm.reshape([frames,pixels])

  #pedestal subtraction
  origIm=origIm-pedIm

  print "ped subtraction"

  origIm=origIm.reshape([frames,pixels_y,pixels_x])
  gainMean=gainIm.mean()
  gainIm=np.tile([gainIm],(frames,1,1))
  origIm=origIm/gainIm*gainMean
  print "gain corrected"

  origIm[:,:,640]=(origIm[:,:,639]+origIm[:,:,641])/2.0
  origIm[:,:,511]=(origIm[:,:,510]+origIm[:,:,512])/2.0
  origIm[:,:,383]=(origIm[:,:,382]+origIm[:,:,384])/2.0
  origIm[:,:,255]=(origIm[:,:,254]+origIm[:,:,256])/2.0
  origIm[:,:,127]=(origIm[:,:,126]+origIm[:,:,128])/2.0

  print "firmware lines corrected"


  badpixmap=badpixmap.reshape([4096,4096])
  # embed in n+2 bigger array
  bigPed=np.ones([frames,4098,4098])
  bigPed=bigPed*(-9999)
  bigPed[:,1:4097,1:4097]=origIm

  bigBPM=np.ones([frames,4098,4098])
  bigBPM=bigBPM*(-9999)
  bigBPM[:,1:4097,1:4097]=badpixmap

  xCoords, yCoords,zCoords = np.where(bigBPM==1) # or np.nonzero(mask)

  numPoints = bigBPM[xCoords, yCoords, zCoords] # or num[mask]
  nBadPix=len(numPoints)/frames


  for pix in range(0,nBadPix):
    pix_x=yCoords[pix]
    pix_y=zCoords[pix]


    subPed=bigPed[:,pix_x-1:pix_x+2,pix_y-1:pix_y+2]
    subBPM=bigBPM[:,pix_x-1:pix_x+2,pix_y-1:pix_y+2]

    numGoodPix=int(((subPed[np.logical_and(subBPM==0,subPed>0)]).size)/frames)

    subPedPerFrame=subPed[np.logical_and(subBPM==0,subPed>0)].reshape([frames,numGoodPix])

    bigPed[:,pix_x,pix_y]=subPedPerFrame.mean(axis=1)
    bigBPM[:,pix_x,pix_y]=0


  print "bad pixels corrected"

  origIm=bigPed[:,1:4097,1:4097]

  return origIm
