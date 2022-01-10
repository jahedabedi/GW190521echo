#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:20:04 2020

@author: jaabed

Bsed on: https://github.com/gwastro/gw150914_investigation/blob/master/CreateResiduals.ipynb
O1 links: https://github.com/gwastro/pycbc-inference-paper/tree/master/posteriors
O2 links: https://github.com/gwastro/o2-bbh-pe/tree/master/posteriors
"""

path = '/home/cwb/waveburst/GWOSC/catalog/GW190521-pvalue/GW190521/'

import numpy as np
import copy
import h5py
import pycbc
from scipy.stats import gaussian_kde as kde
import itertools
#from pycbc.io import FieldArray
from pycbc.conversions import primary_mass, secondary_mass, chi_eff_from_spherical
from pycbc.cosmology import redshift
from matplotlib import rcParams
rcParams['font.size'] = 16
import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)

import os
from pycbc.types import TimeSeries
from pycbc.waveform.generator import FDomainCBCGenerator, FDomainDetFrameGenerator
from pycbc import psd as pypsd
from pycbc import filter
from pycbc import frame
from pycbc import waveform

from pycbc import types

#from pycbc.inference import io
import h5py

import subprocess
from subprocess import call

import glob
import shutil

#fp = io.loadfile("inference.hdf", "r")
fp = h5py.File('inference.hdf','r')
variable_args = fp["samples"].keys()
maxLpt = np.argmax(fp['samples']['loglikelihood'])
static_args = {'approximant': 'IMRPhenomXPHM','f_lower': 10.0,'f_ref': 20.0}
tmerger = 1242442967.4
fhighpass = 15
inverse_psd_len = 4
sample_rate = 4*4096 # we'll just use the same as the LOSC data
data_files = {'H1': 'H-H1_GWOSC_16KHZ_R1-1242440920-4096.gwf',
              'L1': 'L-L1_GWOSC_16KHZ_R1-1242440920-4096.gwf',
              'V1': 'V-V1_GWOSC_16KHZ_R1-1242440920-4096.gwf'}

data = {}
for ifo, fn in data_files.items():
    data[ifo] = frame.read_frame(fn, '{}:GWOSC-16KHZ_R1_STRAIN'.format(ifo))
    
generator = FDomainDetFrameGenerator(FDomainCBCGenerator, data['H1'].start_time, detectors=data.keys(),variable_args=variable_args, delta_t=data['H1'].delta_t, delta_f=data['H1'].delta_f, **static_args)

htildes = generator.generate(**{p: fp['samples'][p][maxLpt] for p in variable_args})

for ifo, h in htildes.items():
    phase_shift=1.383908780
    htildes[ifo] = h * np.exp(1j * phase_shift)

#Subtract the maximum likelihood waveform from the data
waveforms = {}
for ifo, htilde in htildes.items():
    h = waveform.fd_to_td(htilde, delta_t=data['H1'].delta_t, left_window=(10.,15.))
    waveforms[ifo] = h

time = np.arange(data['H1'].start_time,data['H1'].end_time,data['H1'].delta_t)
indx_event = np.where((time>tmerger-20) & (time<tmerger+20))

data_sub = {}
for ifo in data:
    data_sub[ifo] = copy.copy(np.array(data[ifo]))
    data_sub[ifo][indx_event] = data_sub[ifo][indx_event]-np.array(waveforms[ifo])[indx_event]
    shift = int(np.random.uniform(-3*4096,3*4096))
    #shift = 4096
    data_sub[ifo]=np.roll(data_sub[ifo],shift)
    #import matplotlib.pyplot as plt
    #plt.plot(time[indx_event]-tmerger,waveforms[ifo][indx_event])
    #plt.savefig('test.png')
    
#for i in range(0,30):
snr2_a=0
snr2_b=0
#while snr2_b-snr2_a<25:
#for i in range(0,100):
#triger=1
#while triger<2:
for i in range(0,100):
    time_inj=3
    while abs(time_inj)<5:
        time_inj=np.random.uniform(-32,32)
    
    #indx_inj = np.where((time>tmerger-20+time_inj) & (time<tmerger+20+time_inj))
    indx_inj = np.where(time<tmerger+20+time_inj)
    indx_inj = np.where(time[indx_inj]>time[indx_inj][-len(indx_event[0])-1])
    #minmum = min(len(indx_event[0]),len(indx_inj[0]))
    data_sub_i = {}
    for ifo in data:
        data_sub_i[ifo] = copy.copy(data_sub[ifo])
        data_sub_i[ifo][indx_inj] = data_sub_i[ifo][indx_inj] + np.array(waveforms[ifo])[indx_event]
        #data_sub_i[ifo][indx_inj][0:minmum] = data_sub_i[ifo][indx_inj][0:minmum] + np.array(waveforms[ifo])[indx_event][0:minmum]
    
    fn0=glob.glob(path+'FRAMES/*')
    for fni in range(0,len(fn0)):
        os.remove(fn0[fni])
    
    fn1=glob.glob('tmp/*')
    for fni in range(0,len(fn1)):
        os.remove(fn1[fni])
    
    strain_H1_GWF = types.TimeSeries(data_sub_i['H1'], delta_t=data['H1'].delta_t, epoch=data['H1'].start_time)
    frame.write_frame(path+"FRAMES/H-H1_GWOSC_16KHZ_R1-1242440920-4096.gwf", "H1:GWOSC-16KHZ_R1_STRAIN", strain_H1_GWF)
    
    strain_L1_GWF = types.TimeSeries(data_sub_i['L1'], delta_t=data['L1'].delta_t, epoch=data['L1'].start_time)
    frame.write_frame(path+"FRAMES/L-L1_GWOSC_16KHZ_R1-1242440920-4096.gwf", "L1:GWOSC-16KHZ_R1_STRAIN", strain_L1_GWF)
    
    strain_V1_GWF = types.TimeSeries(data_sub_i['V1'], delta_t=data['V1'].delta_t, epoch=data['V1'].start_time)
    frame.write_frame(path+"FRAMES/V-V1_GWOSC_16KHZ_R1-1242440920-4096.gwf", "V1:GWOSC-16KHZ_R1_STRAIN", strain_V1_GWF)
    
    fn=glob.glob('data/*/')
    if len(fn)>0:
        shutil.rmtree(fn[0])
        
    rc = call("./cWB_a.sh")
    fname1 = glob.glob('data/*/*/eventDump.txt')
    if len(fname1)==1:
        f1 = open(fname1[0],'r')
        nums = f1.readlines()
        snr2_a = float(nums[14].split()[1])
        
        fn=glob.glob('data/*/')
        if len(fn)>0:
            shutil.rmtree(fn[0])
        
        rc = call("./cWB_b.sh")
        fname2 = glob.glob('data/*/*/eventDump.txt')
        if (len(fname2)==1) and (snr2_a>100):
            f2 = open(fname2[0],'r')
            nums = f2.readlines()
            snr2_b = float(nums[14].split()[1])
            if snr2_b>100:
                try:
                    f=open(path+'SNR2.txt','r').readlines()
                    SNR2=[float(x) for x in f]
                    SNR2.append(snr2_b-snr2_a)
                    
                    f=open(path+'SNR2a.txt','r').readlines()
                    SNR2a=[float(x) for x in f]
                    SNR2a.append(snr2_a)
                    
                    f=open(path+'SNR2b.txt','r').readlines()
                    SNR2b=[float(x) for x in f]
                    SNR2b.append(snr2_b)
                except:
                    SNR2=[snr2_b-snr2_a]
                    SNR2a=[snr2_a]
                    SNR2b=[snr2_b]
                np.savetxt(path+'SNR2.txt', np.real(SNR2), delimiter=',')
                np.savetxt(path+'SNR2a.txt', np.real(SNR2a), delimiter=',')
                np.savetxt(path+'SNR2b.txt', np.real(SNR2b), delimiter=',')
                
                print('SNR2=',snr2_b-snr2_a)
                del indx_inj, data_sub_i, strain_H1_GWF, strain_L1_GWF, strain_V1_GWF
                import gc
                gc.collect()
        else:
            print('Error in cWB_b: Only one triger is accepted')
            #triger=2
    else:
        print('Error in cWB_s: Only one triger is accepted')
        #triger=2
