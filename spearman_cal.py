from openeo.udf import XarrayDataCube
from typing import Dict
import numpy as np
from datetime import datetime
from scipy.stats import rankdata
import xarray

def apply_datacube(cube: XarrayDataCube, context: Dict) -> XarrayDataCube:
    dat=cube.get_array()

    # Data to be substituted from input
    C=0
    W=0
    Q=0
    D=0
    
    # Preparing input
    IP2=np.empty([D.size,4]).astype(float)
    IP2[:,1:]=np.nan
    IP2[:,0]=D
    IP2[:,1]=C
    IP2[:,2]=W
    IP2[:,3]=Q
    IP2[IP2<0]=np.nan
    D = np.array ([datetime.strptime(str(d)[0:10],'%Y-%m-%d').toordinal() for d in dat.t.values])
    mat0=dat.values[:,0,:,:].astype(float)
    mat0[mat0>10000]=np.nan
    ID=np.where(np.in1d(IP2[:,0],D))[0]
    pix=IP2[ID,1:].astype(float).transpose()
    pix=np.tile(pix,[mat0.shape[1],mat0.shape[2],1,1])
    pix=np.moveaxis(pix,[0,1],[-2,-1])
    
    # CMW calculation    
    wa=(pix[0,:,:,:]-mat0)/(pix[0,:,:,:]-pix[1,:,:,:])
    wa[wa>3]=np.nan
    wa[wa>1]=1
    wa[wa<-2]=np.nan
    wa[wa<0]=0
    coeff=np.nanmax(wa*pix[1,:,:,:]-mat0,0)+np.nanmin(mat0,0)
    CMW=pix[0,:,:,:]/(mat0-wa*pix[1,:,:,:]+coeff)
    
    # Calculation Spearman correlation and masking
    pix2=pix[2,:,:,:]
    ID=np.logical_or(np.isnan(CMW),np.isnan(pix2))
    CMW[ID]=999999
    pix2[ID]=999999
    CMW=rankdata(CMW,axis=0)
    pix2=rankdata(pix2,axis=0)
    CMW[ID]=np.nan
    pix2[ID]=np.nan
    
    S=(np.nansum((CMW-np.array([np.nanmean(CMW,0)]))*(pix2-np.array([np.nanmean(pix2,0)])),
                 axis=0))/((np.nansum((CMW-np.array([np.nanmean(CMW,0)]))**2,
                 axis=0)**0.5)*(np.nansum((pix2-np.array([np.nanmean(pix2,0)]))**2,axis=0)**0.5))
    res=S==np.nanmax(S)
    out = xarray.DataArray(data=res, dims=["x", "y"], 
                           coords={'x': (['x'],dat.x.values,dat.x.attrs ),
                                   'y': (['y'], dat.y.values,dat.y.attrs)})
    out=XarrayDataCube(array=out)
    
    return out

