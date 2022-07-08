# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:42:22 2022

@author: p.filippucci
"""
import numpy as np
import os
import pandas
from datetime import datetime
from swicomp_nan import swicomp_nan
from scipy.stats import pearsonr, spearmanr

pc = os.getcwd()
pc=str.replace(pc,'\\','/')
n=str.find(pc,'Dropbox')
pc=pc[0:n]

t0="2016-10-01"
s0="2022-09-30"
R=np.zeros([6,6])
S=np.zeros([6,6])

for j in range(6):
    if j==0:
        namestat='Polag'
        file=pc+'Dropbox (IRPI CNR)/BACKUP/CURRENT_USERS/p.filippucci/angelica/Dati_oss_portata/Po_5sta_HQvAB2.xlsx'
        cols='B:J'
        rws=6
        sh_nm='LAG'
        
    elif j==1:
        namestat='Piac';
        file=pc+'Dropbox (IRPI CNR)/BACKUP/CURRENT_USERS/p.filippucci/angelica/Dati_oss_portata/Po_5sta_HQvAB2.xlsx'
        cols='B:J'
        rws=6
        sh_nm='PIA'
        
    elif j==2:
        namestat='Kaub';
        file=pc+'Dropbox (IRPI CNR)/BACKUP/CURRENT_USERS/p.filippucci/angelica/Dati_oss_portata/Rhein.xlsx'
        cols='B:J'
        rws=6
        sh_nm='Kaub'

    elif j==3:
        namestat='Mainz';
        file=pc+'Dropbox (IRPI CNR)/BACKUP/CURRENT_USERS/p.filippucci/angelica/Dati_oss_portata/Rhein.xlsx'
        cols='B:J'
        rws=6
        sh_nm='Mainz'

    elif j==4:
        namestat='Chester';
        file=pc+'Dropbox (IRPI CNR)/BACKUP/CURRENT_USERS/p.filippucci/angelica/Dati_oss_portata/Mississippi.xlsx'                       
        cols='G:K'
        rws=11
        sh_nm='Chester'

    elif j==5:        
        namestat='Memphis';
        file=pc+'Dropbox (IRPI CNR)/BACKUP/CURRENT_USERS/p.filippucci/angelica/Dati_oss_portata/Mississippi.xlsx'
        cols='G:K'
        rws=11
        sh_nm='Memphis'

    data=np.load(namestat+'_'+t0+'_'+s0+'.npz', allow_pickle=True)
    l1=data['l1'].tolist()
    l2=data['l2'].tolist()
    data.close()
    
    # Extraction and mean standard deviation scaling of locally obtained timeseries 
    IP=l1[0]
    m=np.nanmin(IP[:,1:],0)
    for i in range(len(l1)-1):
        print(i)
        IP2=l1[i+1]
        m=np.nanmin(np.stack((m,np.nanmin(IP2[:,1:],0))),0)
        ID1=np.in1d(IP[:,0], IP2[:,0])
        ID2=np.in1d(IP2[:,0], IP[:,0])
        my=np.nanmean(IP2[ID2,1:],0)
        sy=np.nanstd(IP2[ID2,1:],0)
        mx=np.nanmean(IP[ID1,1:],0)
        sx=np.nanstd(IP[ID1,1:],0)
        IP3=(IP2[:,1:]-my)/sy*sx+mx
        D=np.unique(np.concatenate((IP[:,0],IP2[:,0])))
        
        ID1=np.in1d(D,IP[:,0])
        ID2=np.in1d(D,IP2[:,0])
        IPn=np.zeros([D.size,8])
        IPn[:]=np.nan
        IPn[:,0]=D
        IPn[ID1,1:]=np.nanmean(np.stack((IP[:,1:],IPn[ID1,1:])),0)
        IPn[ID2,1:]=np.nanmean(np.stack((IP3,IPn[ID2,1:])),0)
        IP=IPn    
    
    # adjusting minimum values
    IP[:,1:]=IP[:,1:]-np.nanmin(IP[:,1:],0)+m
    
    # Extraction and mean standard deviation scaling of timeseries obtained in cloud openeo
    IP1=l2[0]
    m=np.nanmin(IP1[:,1:],0)
    for i in range(len(l2)-1):
        print(np.sum(l2[i]<0))
        print(i)
        IP2=l2[i+1]
        m=np.nanmin(np.stack((m,np.nanmin(IP2[:,1:],0))),0)
        ID1=np.in1d(IP1[:,0], IP2[:,0])
        ID2=np.in1d(IP2[:,0], IP1[:,0])
        my=np.nanmean(IP2[ID2,1:],0)
        sy=np.nanstd(IP2[ID2,1:],0)
        mx=np.nanmean(IP1[ID1,1:],0)
        sx=np.nanstd(IP1[ID1,1:],0)
        IP3=(IP2[:,1:]-my)/sy*sx+mx
        D=np.unique(np.concatenate((IP1[:,0],IP2[:,0])))
        
        ID1=np.in1d(D,IP1[:,0])
        ID2=np.in1d(D,IP2[:,0])
        IPn=np.zeros([D.size,4])
        IPn[:]=np.nan
        IPn[:,0]=D
        IPn[ID1,1:]=np.nanmean(np.stack((IP1[:,1:],IPn[ID1,1:])),0)
        IPn[ID2,1:]=np.nanmean(np.stack((IP3,IPn[ID2,1:])),0)
        IP1=IPn    
        
    # adjusting minimum values
    IP1[:,1:]=IP1[:,1:]-np.nanmin(IP1[:,1:],0)+m
    
    # masking impossible values
    IP[IP<=0]=np.nan
    IP1[IP1<=0]=np.nan
  
    # Extraction of full observed timeseries
    try:        
        Qobs=pandas.read_excel(file, sheet_name=sh_nm, usecols=cols,skiprows=rws)
        Dobs=np.array(Qobs['Date'])
        Dobs=np.array([datetime.fromordinal(int(d-366)) for d in Dobs])
        Vobs=np.array(Qobs["VELOCITA'"])*1.
        Qobs=np.array(Qobs['DISCHARGE'])*1.
    except:
        Qobs=pandas.read_excel(file, sheet_name=sh_nm, usecols=cols,skiprows=rws)
        Dobs=np.array(Qobs['Date.1'])
        Dobs=np.array([datetime.fromordinal(int(d-366)) for d in Dobs])
        Vobs=np.array(Qobs['Flow Velocity (m/s)'])*1.
        Qobs=np.array(Qobs['Q(m3/s)'])*1.

    Qobs[Qobs<0]=np.nan
    
    # Preparing data for performance calculation
    D=IP[:,0]
    C=IP[:,1]
    W=IP[:,4]
    M=np.concatenate((IP[:,5:8],IP1[:,1:4]),1)
    Q=np.zeros(D.size)
    Q[:]=np.nan
    Dobs=[d.toordinal() for d in Dobs]
    ID1=np.in1d(Dobs,D)
    ID2=np.in1d(D,Dobs)
    Q[ID2]=Qobs[ID1]
    
    # %% Performance calculation
    
    for i in range(6):
        # CMW timeseries calculation
        wa=(C-M[:,i])/(C-W)
        wa[wa>3]=np.nan
        wa[wa>1]=1
        wa[wa<-2]=np.nan
        wa[wa<0]=0
        coeff=np.nanmax(wa*W-M[:,i])+np.nanmin(M[:,i])
        CMW=C/(M[:,i]-wa*W+coeff)
        CMW_s=swicomp_nan(CMW,D,5)

        # Performance calculation
        ID=np.logical_or(np.isnan(CMW_s),np.isnan(Q))
        CMW_s2=CMW_s[np.logical_not(ID)]
        Q2=Q[np.logical_not(ID)]
        R[j,i]=pearsonr(CMW_s2,Q2)[0]
        S[j,i]=spearmanr(CMW_s2,Q2)[0]
        
print(R)
print(S)
from scipy.io import savemat

savemat('openeo_results.mat',{'R':R,'S':S})
