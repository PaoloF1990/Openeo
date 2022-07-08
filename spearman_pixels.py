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
    V=0
    V2=0
    W=0
    D=0
    
    # Preparing input
    IP=np.empty([D.size,5]).astype(float)
    IP[:,1:]=np.nan
    IP[:,0]=D
    IP[:,1]=C
    IP[:,2]=V
    IP[:,3]=V2
    IP[:,4]=W
    IP[IP<0]=np.nan
    D = np.array ([datetime.strptime(str(d)[0:10],'%Y-%m-%d').toordinal() for d in dat.t.values])
    mat0=dat.values[:,0,:,:].astype(float)
    mat0[mat0>10000]=np.nan
    ID=np.where(np.in1d(IP[:,0],D))[0]
    res=np.zeros([4,mat0.shape[1],mat0.shape[2]])
    pix=IP[ID,1:].astype(float).transpose()
    pix=np.tile(pix,[mat0.shape[1],mat0.shape[2],1,1])
    pix=np.moveaxis(pix,[0,1],[-2,-1])

    # Calculation Spearman correlation
    for i in range(0,4):
        mat2=mat0.copy()
        pix2=pix[i,:,:,:]
        ID=np.logical_or(np.isnan(pix2),np.isnan(mat2))
        mat2[ID]=999999
        pix2[ID]=999999
        mat2=rankdata(mat2,axis=0)
        pix2=rankdata(pix2,axis=0)
        mat2[ID]=np.nan
        pix2[ID]=np.nan
        res[i,:,:]=(np.nansum((mat2-np.array([np.nanmean(mat2,0)]))*(pix2-np.array([np.nanmean(pix2,0)])),
                              axis=0))/((np.nansum((mat2-np.array([np.nanmean(mat2,0)]))**2,
                              axis=0)**0.5)*(np.nansum((pix2-np.array([np.nanmean(pix2,0)]))**2,axis=0)**0.5))

    
    # Calculation M mask
    corrT=np.nanmax(res[0:3,:,:],axis=0)
    corrW=res[3,:,:]
    Mmask=np.logical_or(np.logical_and(corrT<=np.nanpercentile(corrT,10),corrW<=0.6),
                        np.logical_and(corrW<=np.nanpercentile(corrW,10),corrT<=0.7))
    out = xarray.DataArray(data=Mmask, dims=["x", "y"], 
                           coords={'x': (['x'],dat.x.values,dat.x.attrs ),
                                   'y': (['y'], dat.y.values,dat.y.attrs)})
    out=XarrayDataCube(array=out)
        
    return out

