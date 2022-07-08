# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:14:59 2022

@author: Paolo
"""

# Import required packages
import xskillscore
import matplotlib.pyplot as plt
import openeo
from openeo.processes import is_nan, not_, sd, lte,  mean, array_create, quantiles, sum, add, multiply,gte 
import numpy as np
from netCDF4 import Dataset
import time
from datetime import datetime,timedelta
import xarray
from openeo.udf.xarraydatacube import XarrayDataCube
import os
import pandas as pd
from openeo.rest.datacube import DataCube
from scipy.stats import rankdata

def load_udf(relative_path):
    with open(relative_path, 'r+') as f:
        return f.read()

# %% Initialization

# Input variables
pc = os.getcwd()
pc=str.replace(pc,'\\','/')
n=str.find(pc,'Dropbox')
pc=pc[0:n]

t0= ["2016-10-01","2017-10-01","2018-10-01","2019-10-01","2020-10-01"] #period to analyze
s0=["2018-09-30","2019-09-30","2020-09-30","2021-09-30","2022-09-30"]
band=["B08","B04"]
maskband=["CLD","CLP"]
maskthr=50
n_pix=10
filt= (np.ones([5,5])/25).tolist()

for j in [0,2,4]:
    if j==0:
        namestat='Polag'
        coord=[11.577,44.871,11.617,44.911]
        file=pc+'Dropbox (IRPI CNR)/BACKUP/CURRENT_USERS/p.filippucci/angelica/Dati_oss_portata/Po_5sta_HQvAB2.xlsx'
        cols='B:J'
        rws=6
        sh_nm='LAG'
        
    elif j==1:
        namestat='Piac';
        coord=[9.64323-0.02, 45.11079-0.02,9.64323+0.02, 45.11079+0.02]
        file=pc+'Dropbox (IRPI CNR)/BACKUP/CURRENT_USERS/p.filippucci/angelica/Dati_oss_portata/Po_5sta_HQvAB2.xlsx'
        cols='B:J'
        rws=6
        sh_nm='PIA'
        
    elif j==2:
        namestat='Kaub';
        coord=[7.8144-0.02, 50.0345-0.02,7.8144+0.02, 50.0345+0.02]
        file=pc+'Dropbox (IRPI CNR)/BACKUP/CURRENT_USERS/p.filippucci/angelica/Dati_oss_portata/Rhein.xlsx'
        cols='B:J'
        rws=6
        sh_nm='Kaub'

    elif j==3:
        namestat='Mainz';
        coord=[8.09793-0.02, 50.01564-0.02,8.09793+0.02, 50.01564+0.02]
        file=pc+'Dropbox (IRPI CNR)/BACKUP/CURRENT_USERS/p.filippucci/angelica/Dati_oss_portata/Rhein.xlsx'
        cols='B:J'
        rws=6
        sh_nm='Mainz'

    elif j==4:
        namestat='Chester';
        coord=[-89.72484-0.02, 37.83679-0.02,-89.72484+0.02, 37.83679+0.02]
        file=pc+'Dropbox (IRPI CNR)/BACKUP/CURRENT_USERS/p.filippucci/angelica/Dati_oss_portata/Mississippi.xlsx'                       
        cols='G:K'
        rws=11
        sh_nm='Chester'

    elif j==5:        
        namestat='Memphis';
        coord=[-90.104-0.02, 35.127-0.02,-90.104+0.02, 35.127+0.02]
        file=pc+'Dropbox (IRPI CNR)/BACKUP/CURRENT_USERS/p.filippucci/angelica/Dati_oss_portata/Mississippi.xlsx'
        cols='G:K'
        rws=11
        sh_nm='Memphis'

    print(namestat)
    list1=[]
    list2=[]

    # Cycle to extract the mask every 2 years
    for i in range(len(t0)):#range(0,1):#
        print('initialization '+t0[i])

        # Openeo server connection
        connection = openeo.connect("https://openeo.cloud")
        connection.authenticate_oidc()

        # Collection input
        temp_ext=[t0[i]+' 00:00:00',s0[i]+' 00:00:00']
        spat_extent={"west": coord[0], "south": coord[1], "east": coord[2], "north": coord[3]}
        rect={
          "type":"FeatureCollection",
          "features":[
            {
              "type":"Feature",
              "geometry": {
                "type":"Polygon",
                "coordinates":[[[coord[0],coord[1]],[coord[0],coord[3]],[coord[2],coord[3]],[coord[2],coord[1]],[coord[0],coord[1]]]]
              },
              "properties":{"name":"area1"}
            }
          ]
        }
        
        # Extraction of observed data of river discharge
        dr: pd.DatetimeIndex = pd.date_range(start=temp_ext[0], end=temp_ext[1], freq="MS")
        t_intervals = [[str(d), str(dr[i+1])] for i, d in enumerate(dr[:-1])]
        try:        
            Qobs=pd.read_excel(file, sheet_name=sh_nm, usecols=cols,skiprows=rws)
            Dobs=np.array(Qobs['Date'])
            Dobs=np.array([datetime.fromordinal(int(d-366)) for d in Dobs])
            Vobs=np.array(Qobs["VELOCITA'"])*1.
            Qobs=np.array(Qobs['DISCHARGE'])*1.
        except:
            Qobs=pd.read_excel(file, sheet_name=sh_nm, usecols=cols,skiprows=rws)
            Dobs=np.array(Qobs['Date.1'])
            Dobs=np.array([datetime.fromordinal(int(d-366)) for d in Dobs])
            Vobs=np.array(Qobs['Flow Velocity (m/s)'])*1.
            Qobs=np.array(Qobs['Q(m3/s)'])*1.

        Qobs[Qobs<0]=np.nan
            
        # Extracting collection of Sentinel-2 1c NIR and NDVI
        datacube = connection.load_collection(collection_id = "SENTINEL2_L1C_SENTINELHUB", 
                                              spatial_extent = spat_extent, 
                                              temporal_extent = temp_ext, 
                                              bands = band[0],
                                              properties={"eo:cloud_cover": lambda x: lte(x, 70)}).linear_scale_range(0, 10000, 0, 1)
        datacubeNDVI = connection.load_collection(collection_id = "SENTINEL2_L1C_SENTINELHUB", 
                                              spatial_extent = spat_extent, 
                                              temporal_extent = temp_ext, 
                                              bands = band,
                                              properties={"eo:cloud_cover": lambda x: lte(x, 70)}).linear_scale_range(0, 10000, 0, 1)
        
        # Generating and apply cloud masks from Sentinel-2 2a
        mask0 = connection.load_collection(collection_id = "SENTINEL2_L2A_SENTINELHUB", 
                                           spatial_extent = spat_extent, 
                                           temporal_extent = temp_ext, 
                                           bands = maskband,
                                           properties={"eo:cloud_cover": lambda x: lte(x, 70)})
        mask0 = (mask0.band(maskband[0])>= maskthr).logical_or(mask0.band(maskband[1]) >= maskthr*2.55)
        mask0 = mask0.resample_cube_spatial(datacube)
        datacube=datacube.mask(mask0)
        datacube=datacube.mask(datacube<=0)

        # Generating masks of water and land areas
        wat = connection.load_collection(collection_id = "GLOBAL_SURFACE_WATER", 
                                         spatial_extent = spat_extent, 
                                         temporal_extent = ["1980-01-01", "2020-12-31"], 
                                         bands = ['extent']).max_time()
        wat = wat.apply(lambda val: not_(is_nan(x=val)))
        wat = wat.apply_kernel(filt).resample_cube_spatial(datacube) == 0 #.apply(lambda val: eq(x=val,y=0))
        land = wat == 0 
        
        # Generating and apply masks based on the number of valid pixels and generation of NDVI collection
        count_dc: DataCube = datacube \
            .apply(process = lambda x: add(x = multiply(x = x, y = 0), y = 1)) \
            .aggregate_temporal(
                intervals=[temp_ext],
                reducer=lambda data: sum(data),
            )

        valmask: datacube = count_dc.apply(process = lambda x: lte(x=x, y=10))        
        valmask=valmask.drop_dimension("t")
        datacube=datacube.mask(valmask)
        datacubeNDVI=datacubeNDVI.mask(mask0).mask(land).mask(valmask)
        datacubeNDVI=datacubeNDVI.ndvi(nir=band[0],red=band[1],target_band='NDVI').filter_bands(['NDVI'])
        
        # Generation of second NIR collection with kernel applied      
        datacube2=datacube.apply_kernel(filt)
        datacube2=datacube2.mask(datacube2<=0)
        
        # %% C mask calculation
        
        asd=0
        while asd==0:
            try:
                # Calculation the coefficient of variation CV over land
                connection.authenticate_oidc()
                print('C mask calculation')
                m=datacube.mean_time()
                s=datacube.apply_dimension(process='sd',dimension='t')
                cv=s/m
                cv=cv.mask(land)
                cv=cv.drop_dimension("t")
                
                # Creation C mask where CV<5° percentile
                val = cv.aggregate_spatial(rect, lambda pixels: array_create([quantiles(pixels,q=20)]))
                print('extracting fifth percentile')
                job=val.execute_batch(out_format="csv",title="percentiles")
                job.get_results().download_files()
                val=pd.read_csv('timeseries.csv')   
                val=val.values[0][1]                     
                
                Cmask=cv<float(val)
                Cmask=Cmask==0
                
                # Extraction of average C timeseries
                C0=datacube.mask(Cmask)
                print('extracting masked dataset')
                C0=C0.aggregate_spatial(rect, "mean")
                print('extracting masked timeseries')
                job=C0.execute_batch(out_format="csv",outputfile="C.csv")
                job.get_results().download_files()
                C0=pd.read_csv('C.csv')   
                C0=np.append([C0.columns.to_numpy().tolist()],C0.to_numpy().tolist(),axis=0).transpose()
                
                asd=1
            except Exception as e: 
                print(e)
                print('outtime. reload')
                

        
        # %% W mask calculation
        asd=0
        while asd==0:
            try:
                # Calculation of product average*standard deviation over water
                connection.authenticate_oidc()
                print('W mask calculation')
                prodW=s*m
                prodW=prodW.mask(wat)
                prodW=prodW.drop_dimension("t")

                # Creation W mask where prod<5° percentile
                val = prodW.aggregate_spatial(rect, lambda pixels: array_create([quantiles(pixels,q=20)]))
                print('extracting fifth percentile')
                job=val.execute_batch(out_format="csv",title="percentiles")
                job.get_results().download_files()
                val=pd.read_csv('timeseries.csv')   
                val=val.values[0][1]                     

                Wmask=prodW<float(val)
                Wmask=Wmask==0
                
                # Extraction of average W timeseries
                W0=datacube.mask(Wmask)
                print('extracting masked dataset')
                W0=W0.aggregate_spatial(rect, "mean")
                print('extracting masked timeseries')
                job=W0.execute_batch(out_format="csv",outputfile="W.csv")
                job.get_results().download_files()
                W0=pd.read_csv('W.csv')   
                W0=np.append([W0.columns.to_numpy().tolist()],W0.to_numpy().tolist(),axis=0).transpose()

                asd=1
            except Exception as e: 
                print(e)
                print('outtime. reload')
                

        
        # %% V mask calculation
        asd=0
        while asd==0:
            try:
                # Calculation of NDVI average over land
                connection.authenticate_oidc()
                print('V mask calculation')
                m=datacubeNDVI.mean_time()
        
                # Creation V mask where NDVI average>95° percentile
                val = m.aggregate_spatial(rect, lambda pixels: array_create([quantiles(pixels,q=20)]))
                print('extracting ninetyfifth percentile')
                job=val.execute_batch(out_format="csv",title="percentiles")
                job.get_results().download_files()
                val=pd.read_csv('timeseries.csv')   
                val=val.values[0][-1]      
                
                #val=val[0][-1]
                Vmask=m>float(val)
                Vmask=Vmask==0
                
                # Extraction of average V timeseries
                V0=datacube.mask(Vmask)
                print('extracting masked dataset')
                V0=V0.aggregate_spatial(rect, "mean")
                print('extracting masked timeseries')
                job=V0.execute_batch(out_format="csv",outputfile="V.csv")
                job.get_results().download_files()
                V0=pd.read_csv('V.csv')   
                V0=np.append([V0.columns.to_numpy().tolist()],V0.to_numpy().tolist(),axis=0).transpose()

                asd=1
            except Exception as e: 
                print(e)
                print('outtime. reload')


        
        # %% V2 mask calculation
        asd=0
        while asd==0:
            try:
                # Calculation of NDVI standard deviation over land
                connection.authenticate_oidc()
                print('V2 mask calculation')
                s=datacubeNDVI.apply_dimension(process='sd',dimension='t')
                s=s.drop_dimension("t")
                
                # Creation V2 mask where NDVI average>95° percentile
                val = s.aggregate_spatial(rect, lambda pixels: array_create([quantiles(pixels,q=20)]))
                print('extracting ninetyfifth percentile')
                job=val.execute_batch(out_format="csv",title="percentiles")
                job.get_results().download_files()
                val=pd.read_csv('timeseries.csv')   
                val=val.values[0][-1]                     

                #val=val[0][-1]
                V2mask=s>float(val)
                V2mask=V2mask==0
                
                # Extraction of average V2 timeseries
                V20=datacube.mask(V2mask)
                print('extracting masked dataset')
                V20=V20.aggregate_spatial(rect, "mean")
                print('extracting masked timeseries')
                job=V20.execute_batch(out_format="csv",outputfile="V2.csv")
                job.get_results().download_files()
                V20=pd.read_csv('V2.csv')   
                V20=np.append([V20.columns.to_numpy().tolist()],V20.to_numpy().tolist(),axis=0).transpose()
                
                asd=1
            except Exception as e: 
                print(e)
                print('outtime. reload')
 

 
        # %% full data download
        
        asd=0
        while asd==0:
            try:
                # download of NIR datacubes over water for local analysis of M position
                connection.authenticate_oidc()
                print('full datacube download')
                datacube=datacube.mask(wat)
                #datacube.download("data_"+str(j)+".nc", format="netcdf")
                job=datacube.execute_batch(out_format="netcdf",outputfile="data_"+str(j)+".nc")
                job.get_results().download_files()

                datacube2=datacube2.mask(wat)
                #datacube2.download("data2_"+str(j)+".nc", format="netcdf")
                job=datacube2.execute_batch(out_format="netcdf",outputfile="data2_"+str(j)+".nc")
                job.get_results().download_files()
                os.remove('openEO.nc')
                asd=1
            except Exception as e: 
                print(e)
                print('outtime. reload')
        
        #try:
        #    os.remove('prova.npz')
        #except:
        #    pass
        #np.savez('prova.npz',C0=C0,V0=V0,V20=V20,W0=W0)

        # %% Preparing input data
        
        #data=np.load('prova.npz', allow_pickle=True)
        #C0=data['C0'].tolist()
        #V0=data['V0'].tolist()
        #V20=data['V20'].tolist()
        #W0=data['W0'].tolist()
        #data.close()
        
        # Extracting dates and pixel values
        DC = list(C0[:,0])
        DC = np.array ([datetime.strptime(d[0:10],'%Y-%m-%d') for d in DC])
        C0 = C0[:,1].astype(float)
        ID = np.logical_not(np.isnan(C0))
        DC = DC[ID]
        C0 = C0[ID]
        
        DV = list(V0[:,0])
        DV = np.array ([datetime.strptime(d[0:10],'%Y-%m-%d') for d in DV])
        V0 = V0[:,1].astype(float)
        ID = np.logical_not(np.isnan(V0))
        DV = DV[ID]
        V0 = V0[ID]
        
        DV2 = list(V20[:,0])
        DV2 = np.array ([datetime.strptime(d[0:10],'%Y-%m-%d') for d in DV2])
        V20 = V20[:,1].astype(float)
        ID = np.logical_not(np.isnan(V20))
        DV2 = DV2[ID]
        V20 = V20[ID]
        
        DW = list(W0[:,0])
        DW = np.array ([datetime.strptime(d[0:10],'%Y-%m-%d') for d in DW])
        W0 = W0[:,1].astype(float)
        ID = np.logical_not(np.isnan(W0))
        DW = DW[ID]
        W0 = W0[ID]

        # Creation temporal matrix of interested pixels for uncalibrated analysis
        D=np.unique(np.concatenate([DC,DV,DV2,DW]))
        ID1=np.where(np.in1d(D, DC))[0]
        ID2=np.where(np.in1d(D, DV))[0]
        ID3=np.where(np.in1d(D, DV2))[0]
        ID4=np.where(np.in1d(D, DW))[0]
        IP=np.empty([D.size,8]).astype(float)
        IP[:,1:8]=-np.nan
        D=[d.toordinal() for d in D]
        IP[:,0]=D
        IP[ID1,1]=C0
        IP[ID2,2]=V0
        IP[ID3,3]=V20
        IP[ID4,4]=W0
        IP[np.isnan(IP)]=-9999
        IP=IP.astype(object)
        
        # Creation temporal matrix of interested pixels for calibrated analysis
        D2=np.unique(np.concatenate([DC,DW]))
        ID1=np.where(np.in1d(D2, DC))[0]
        ID2=np.where(np.in1d(D2, DW))[0]
        ID3=np.where(np.in1d(D2, Dobs))[0]
        ID3_2=np.where(np.in1d(Dobs, D2))[0]
        IP2=np.empty([D2.size,4]).astype(float)
        IP2[:,1:4]=-np.nan
        D2=[d.toordinal() for d in D2]
        IP2[:,0]=D2
        IP2[ID1,1]=C0
        IP2[ID2,2]=W0
        IP2[ID3,3]=Qobs[ID3_2]
        IP2[np.isnan(IP2)]=-9999
        IP2=IP2.astype(object)

        # %% Application of uncalibrated methodology on openeo cloud
        
        asd=0
        while asd==0:
            try:
                # Loading of UDF and input of the interested pixels timeseries
                connection.authenticate_oidc()
                print('openeo uncal')
                spearman_udf = load_udf('spearman_pixels.py')
                spearman_udf=spearman_udf.replace("D=0",'D=np.array('+str(IP[:,0].tolist())+')')
                spearman_udf=spearman_udf.replace("C=0",'C=np.array('+str(IP[:,1].tolist())+')')
                spearman_udf=spearman_udf.replace("V=0",'V=np.array('+str(IP[:,2].tolist())+')')
                spearman_udf=spearman_udf.replace("V2=0",'V2=np.array('+str(IP[:,3].tolist())+')')
                spearman_udf=spearman_udf.replace("W=0",'W=np.array('+str(IP[:,4].tolist())+')')
                
                # Creation M mask uncalibrated from UDF
                Mmask = datacube.apply_dimension(code=spearman_udf, runtime='Python')
                job=Mmask.execute_batch(out_format="netcdf",outputfile=".nc")
                job.get_results().download_files()

                Mmask=Mmask.drop_dimension("t")
                """
                Mmask.download('Mmask_uc_op.nc')
                data= Dataset("Mmask_uc_op.nc", "r")
                mat=data.variables['B08'][:].data
                data.close()
                plt.figure()
                plt.imshow(mat[:,:])
                plt.title('Mmask_uncal_op')
                plt.savefig('Mmask_uncal_op.png')
                """
                Mmask=Mmask==0
                
                # Extraction of average M timeseries
                M0=datacube.mask(Mmask)
                print('extracting masked dataset')
                M0=M0.aggregate_spatial(rect, "mean")
                print('extracting masked timeseries')
                job=M0.execute_batch(out_format="csv",outputfile="M.csv")
                job.get_results().download_files()
                M0=pd.read_csv('M.csv')   
                M0=np.append([M0.columns.to_numpy().tolist()],M0.to_numpy().tolist(),axis=0).transpose()
                

                # Extraction of resulting dates and timeseries
                DM = list(M0[:,0])
                DM = np.array ([datetime.strptime(d[0:10],'%Y-%m-%d') for d in DM])
                M0 = M0[:,1].astype(float)
                ID = np.logical_not(np.isnan(M0))
                DM_op = DM[ID]
                M0_op = M0[ID]

                asd=1
            except Exception as e: 
                print(e)
                print('outtime. reload')

        # %% Application of uncalibrated methodology locally
        
        asd=0
        while asd==0:
            try:
                # Reading datacube for M analysis
                connection.authenticate_oidc()
                print('local uncal')
                cube=xarray.open_dataset("data_"+str(j)+".nc")
                cube=XarrayDataCube(cube.to_array(dim='bands'))
                dat=cube.get_array()

                # Creation xarray dataarray of the interested pixels
                IP[IP==-9999]=np.nan
                coords = {'t': (['t'], [datetime.fromordinal(int(t)) for t in IP[:,0]])}
                dC = xarray.DataArray(data=IP[:,1], coords=coords).astype(float)
                dV = xarray.DataArray(data=IP[:,2], coords=coords).astype(float)
                dV2 = xarray.DataArray(data=IP[:,3], coords=coords).astype(float)
                dW = xarray.DataArray(data=IP[:,4], coords=coords).astype(float)
                dat.values[dat.values==b'']=0
                dat=xarray.where(dat>10000,np.nan,dat)

                # Spearman correlation calculation
                sC=xskillscore.spearman_r(dat.astype(float), dC, dim="t", weights=None, skipna=True, keep_attrs=False)
                sW=xskillscore.spearman_r(dat.astype(float), dW, dim="t", weights=None, skipna=True, keep_attrs=False)
                sV=xskillscore.spearman_r(dat.astype(float), dV, dim="t", weights=None, skipna=True, keep_attrs=False)
                sV2=xskillscore.spearman_r(dat.astype(float), dV2, dim="t", weights=None, skipna=True, keep_attrs=False)
                
                # Creation M mask uncalibrated from local data
                corrT=np.nanmax(np.stack((sC.values[1,:,:],sV.values[1,:,:],sV2.values[1,:,:])),axis=0)
                corrW=sW.values[1,:,:]
                Mmask=np.logical_or(np.logical_and(corrT<=np.nanpercentile(corrT,10),corrW<=0.6),
                                    np.logical_and(corrW<=np.nanpercentile(corrW,10),corrT<=0.7))
                """
                plt.figure()
                plt.imshow(Mmask)
                plt.title('Mmask_uncal_nc')
                plt.savefig('Mmask_uncal_nc.png')
                """
                
                # Extraction local data and calculation M average timeseries
                data= Dataset("data_"+str(j)+".nc", "r")
                mat=data.variables['B08'][:].data
                mat=mat.reshape([mat.shape[0],mat.shape[1]*mat.shape[2]]).transpose().astype(float)
                x=data.variables['x'][:].data.astype(float)
                y=data.variables['y'][:].data.astype(float)
                DM=data.variables['t'][:].data.astype(float)
                data.close()
                mat[mat>10000]=np.nan
                DM=np.array ([(datetime(1990,1,1)+timedelta(days=d)).toordinal() for d in DM])
                [xx,yy]=np.meshgrid(x,y)
                xx=xx.flatten()
                yy=yy.flatten()
                Mmask=Mmask.flatten()
                Mmask=Mmask==0                    
                mat[Mmask,:]=np.nan
                M0=np.nanmean(mat,0)
                IP=IP.astype(float)        
                IDM=np.where(np.in1d(IP[:,0], DM))[0]
                IP[IDM,5]=M0
                
                asd=1
            except Exception as e: 
                print(e)
                print('outtime. reload')

        # %% Application of calibrated methodology on openeo cloud
        
        asd=0
        while asd==0:
            try:
                # Loading of UDF and input of the interested pixels timeseries
                connection.authenticate_oidc()
                spearman_c_udf = load_udf('spearman_cal.py')
                spearman_c_udf=spearman_c_udf.replace("D=0",'D=np.array('+str(IP2[:,0].tolist())+')')
                spearman_c_udf=spearman_c_udf.replace("C=0",'C=np.array('+str(IP2[:,1].tolist())+')')
                spearman_c_udf=spearman_c_udf.replace("W=0",'W=np.array('+str(IP2[:,2].tolist())+')')
                spearman_c_udf=spearman_c_udf.replace("Q=0",'Q=np.array('+str(IP2[:,3].tolist())+')')
                      
                # Creation M mask calibrated from UDF            
                Mmask_cal = datacube.apply_dimension(code=spearman_c_udf, runtime='Python')
                Mmask_cal=Mmask_cal.drop_dimension("t")
                """
                Mmask_cal.download("Mmask_c_op.nc")
                data= Dataset("Mmask_c_op.nc", "r")
                mat=data.variables['B08'][:].data
                data.close()
                plt.figure()
                plt.imshow(mat[:,:])
                plt.title('Spearman_Mmask_cal_op')
                plt.clim(0,1)
                plt.savefig('Spearman_Mmask_cal_op.png')
                Mmask_cal=Mmask_cal==0
                """
                
                # Extraction of average M timeseries
                M0_cal=datacube.mask(Mmask_cal)
                print('extracting masked dataset')
                M0_cal=M0_cal.aggregate_spatial(rect, "mean")
                print('extracting masked timeseries')
                job=M0_cal.execute_batch(out_format="csv",outputfile="M_c.csv")
                job.get_results().download_files()
                M0_cal=pd.read_csv('M_c.csv')   
                M0_cal=np.append([M0_cal.columns.to_numpy().tolist()],M0_cal.to_numpy().tolist(),axis=0).transpose()
                
                # Extraction of resulting dates and timeseries
                DM_cal = list(M0_cal[:,0])
                DM_cal = np.array ([datetime.strptime(d[0:10],'%Y-%m-%d') for d in DM_cal])
                M0_cal = M0_cal[:,1].astype(float)
                ID = np.logical_not(np.isnan(M0_cal))
                DM_cal_op = DM_cal[ID]
                M0_cal_op = M0_cal[ID]
                
                asd=1
            except Exception as e: 
                print(e)
                print('outtime. reload')

        # %% Application of calibrated methodology locally

        asd=0
        while asd==0:
            try:
                # Reading datacube for M analysis
                connection.authenticate_oidc()
                cube=xarray.open_dataset("data_"+str(j)+".nc")
                cube=XarrayDataCube(cube.to_array(dim='bands'))
                dat=cube.get_array()

                # Creation xarray dataarray of the interested pixels
                IP2[IP2==-9999]=np.nan
                coords = {'t': (['t'], [datetime.fromordinal(int(t)) for t in IP2[:,0]])}
                dC = xarray.DataArray(data=IP2[:,1], coords=coords).astype(float)
                dW = xarray.DataArray(data=IP2[:,2], coords=coords).astype(float)
                dQ = xarray.DataArray(data=IP2[:,3], coords=coords).astype(float)
                dat.values[dat.values==b'']=np.nan
                dat=xarray.where(dat>10000,np.nan,dat)

                # CMW timeseries calculation and M mask calibration
                wa=(dC-dat.astype(float)[1,:,:,:])/(dC-dW)
                wa=xarray.where(wa>3,np.nan,wa)
                wa=xarray.where(wa>1,1,wa)
                wa=xarray.where(wa<-2,np.nan,wa)
                wa=xarray.where(wa<0,0,wa)
                coeff=np.nanmax(wa*np.moveaxis(np.tile(dW,[512,512,1]),[0,1],[-2,-1])-dat.astype(float)[1,:,:,:],0)+np.nanmin(dat.astype(float)[1,:,:,:],0)
                dCMW=dC/(dat.astype(float)[1,:,:,:]-wa*np.moveaxis(np.tile(dW,[512,512,1]),[0,1],[-2,-1])+coeff)
                sM=xskillscore.spearman_r(dCMW, dQ, dim="t", weights=None, skipna=True, keep_attrs=False)
                Mmask_cal=sM==np.nanmax(sM)
                """
                plt.figure()
                plt.imshow(sM)
                plt.title('Spearman_Mmask_cal_nc')
                plt.clim(0,1)
                plt.savefig('Spearman_Mmask_cal_nc.png')
                """
                
                # Extraction local data and calculation M average timeseries
                data= Dataset("data_"+str(j)+".nc", "r")
                mat=data.variables['B08'][:].data
                mat=mat.reshape([mat.shape[0],mat.shape[1]*mat.shape[2]]).transpose().astype(float)
                x=data.variables['x'][:].data.astype(float)
                y=data.variables['y'][:].data.astype(float)
                DM=data.variables['t'][:].data.astype(float)
                data.close()
                mat[mat>10000]=np.nan
                DM=np.array ([(datetime(1990,1,1)+timedelta(days=d)).toordinal() for d in DM])
                [xx,yy]=np.meshgrid(x,y)
                xx=xx.flatten()
                yy=yy.flatten()
                Mmask_cal=Mmask_cal.values.flatten()
                Mmask_cal=Mmask_cal==0                    
                mat[Mmask_cal,:]=np.nan
                M0=np.nanmean(mat,0)
                IDM=np.where(np.in1d(IP[:,0], DM))[0]
                IP[IDM,6]=M0

                asd=1
            except Exception as e: 
                print(e)
                print('outtime. reload')

        # %% Application of calibrated methodology on 5x5 pixels locally
        
        asd=0
        while asd==0:
            try:
                connection.authenticate_oidc()
                IP2=IP2.astype(float)
                IP2[np.isnan(IP2)]=-9999
                IP2=IP2.astype(object)

                # Loading of UDF and input of the interested pixels timeseries
                spearman_c_udf = load_udf('spearman_cal.py')
                spearman_c_udf=spearman_c_udf.replace("D=0",'D=np.array('+str(IP2[:,0].tolist())+')')
                spearman_c_udf=spearman_c_udf.replace("C=0",'C=np.array('+str(IP2[:,1].tolist())+')')
                spearman_c_udf=spearman_c_udf.replace("W=0",'W=np.array('+str(IP2[:,2].tolist())+')')
                spearman_c_udf=spearman_c_udf.replace("Q=0",'Q=np.array('+str(IP2[:,3].tolist())+')')
                      
                # Creation M mask calibrated kernel5 from UDF
                Mmask_cal_5 = datacube2.apply_dimension(code=spearman_c_udf, runtime='Python')
                Mmask_cal_5=Mmask_cal_5.drop_dimension("t")
                """
                data= Dataset("Mmask_c_5_op.nc", "r")
                mat=data.variables['B08'][:].data
                data.close()
                plt.figure()
                plt.imshow(mat[0,:,:])
                plt.title('Spearman_Mmask_cal_5x5_op')
                plt.clim(0,1)
                plt.savefig('Spearman_Mmask_cal_5x5_op.png')
                """
                Mmask_cal_5=Mmask_cal_5==0
                
                # Extraction of average M timeseries
                M0_cal_5=datacube2.mask(Mmask_cal_5)
                print('extracting masked dataset')
                M0_cal_5=M0_cal_5.aggregate_spatial(rect, "mean")
                print('extracting masked timeseries')
                job=M0_cal_5.execute_batch(out_format="csv",outputfile="M_c5.csv")
                job.get_results().download_files()
                M0_cal_5=pd.read_csv('M_c5.csv')   
                M0_cal_5=np.append([M0_cal_5.columns.to_numpy().tolist()],M0_cal_5.to_numpy().tolist(),axis=0).transpose()
                
                # Extraction of resulting dates and timeseries
                DM_cal_5 = list(M0_cal_5[:,0])
                DM_cal_5 = np.array ([datetime.strptime(d[0:10],'%Y-%m-%d') for d in DM_cal_5])
                M0_cal_5 = M0_cal_5[:,1].astype(float)
                ID = np.logical_not(np.isnan(M0_cal_5))
                DM_cal_5_op = DM_cal_5[ID]
                M0_cal_5_op = M0_cal_5[ID]

                asd=1
            except Exception as e: 
                print(e)
                print('outtime. reload')

        # %% local_cal_5x5
        
        asd=0
        while asd==0:
            try:
                # Reading datacube for M analysis
                connection.authenticate_oidc()
                cube=xarray.open_dataset("data2_"+str(j)+".nc")
                cube=XarrayDataCube(cube.to_array(dim='bands'))
                dat=cube.get_array()
    
                # Creation xarray dataarray of the interested pixels
                coords = {'t': (['t'], [datetime.fromordinal(int(t)) for t in IP2[:,0]])}
                dC = xarray.DataArray(data=IP2[:,1], coords=coords).astype(float)
                dW = xarray.DataArray(data=IP2[:,2], coords=coords).astype(float)
                dQ = xarray.DataArray(data=IP2[:,3], coords=coords).astype(float)
                dat.values[dat.values==b'']=np.nan
                dat=xarray.where(dat>10000,np.nan,dat)
 
                # CMW timeseries calculation and M mask calibration
                wa=(dC-dat.astype(float)[1,:,:,:])/(dC-dW)
                wa=xarray.where(wa>3,np.nan,wa)
                wa=xarray.where(wa>1,1,wa)
                wa=xarray.where(wa<-2,np.nan,wa)
                wa=xarray.where(wa<0,0,wa)
                coeff=np.nanmax(wa*np.moveaxis(np.tile(dW,[512,512,1]),[0,1],[-2,-1])-dat.astype(float)[1,:,:,:],0)+np.nanmin(dat.astype(float)[1,:,:,:],0)
                dCMW=dC/(dat.astype(float)[1,:,:,:]-wa*np.moveaxis(np.tile(dW,[512,512,1]),[0,1],[-2,-1])+coeff)
                sM=xskillscore.spearman_r(dCMW, dQ, dim="t", weights=None, skipna=True, keep_attrs=False)
                Mmask_cal=sM==np.nanmax(sM)
                """
                plt.figure()
                plt.imshow(sM)
                plt.title('Spearman_Mmask_cal_5x5_nc')
                plt.clim(0,1)
                plt.savefig('Spearman_Mmask_cal_5x5_nc.png')
                """
                
                # Extraction local data and calculation M average timeseries
                data= Dataset("data2_"+str(j)+".nc", "r")
                mat=data.variables['B08'][:].data
                mat=mat.reshape([mat.shape[0],mat.shape[1]*mat.shape[2]]).transpose().astype(float)
                x=data.variables['x'][:].data.astype(float)
                y=data.variables['y'][:].data.astype(float)
                DM=data.variables['t'][:].data.astype(float)
                data.close()
                mat[mat>10000]=np.nan
                DM=np.array ([(datetime(1990,1,1)+timedelta(days=d)).toordinal() for d in DM])
                [xx,yy]=np.meshgrid(x,y)
                xx=xx.flatten()
                yy=yy.flatten()
                Mmask_cal=Mmask_cal.values.flatten()
                Mmask_cal=Mmask_cal==0                    
                mat[Mmask_cal,:]=np.nan
                M0=np.nanmean(mat,0)
                IDM=np.where(np.in1d(IP[:,0], DM))[0]
                IP[IDM,7]=M0
             
                asd=1
            except Exception as e: 
                print(e)
                print('outtime. reload')
            
            os.remove('timeseries.csv')
            os.remove('V2.csv')
            os.remove('V.csv')
            os.remove('W.csv')
            os.remove('C.csv')
            os.remove("data_"+str(j)+".nc")
            os.remove("data2_"+str(j)+".nc")

        # %% Data extrapolation
        
        # Creation temporal matrix of openeo M timeseries
        D=np.unique(np.concatenate([DM_op,DM_cal_op,DM_cal_5_op]))
        ID1=np.where(np.in1d(D, DM_op))[0]
        ID2=np.where(np.in1d(D, DM_cal_op))[0]
        ID3=np.where(np.in1d(D, DM_cal_5_op))[0]
        IP3=np.empty([D.size,4]).astype(float)
        IP3[:,1:4]=-np.nan
        IP3[:,0]=D
        IP3[ID1,1]=M0_op
        IP3[ID2,2]=M0_cal_op
        IP3[ID3,3]=M0_cal_5_op
        
        # Appending results for analysis
        list1.append(IP)
        list2.append(IP3)
        
    np.savez(namestat+'_'+t0[0]+'_'+s0[-1]+'.npz',l1=list1,l2=list2)
