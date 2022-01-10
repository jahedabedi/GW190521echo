#! /usr/bin/python2.7

## index into the strain time series for this time interval:

deltati = 5
deltatf = 5


# In[ ]:

# Standard python numerical analysis imports:
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import copy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import pyplot, patches
from numpy import transpose

import pycbc
import matplotlib.colors as colors
#spec_cmap='ocean'
from pycbc import frame

from pycbc.inference import io
import h5py
from pycbc.results import str_utils, scatter_histograms

#deltat = 10
tmerger = 1242442967.4
eventname = 'GW190521'

file_names = {'NRSur7dq4 and $\Lambda$':'NRSur7dq4_NULL_Jahed.hdf'}

cWB_cnt = 4
#cWB_cnt = 0
#cWB_cnt = 1

path1= 'gwf_files/'
path3 = 'out/'

path2 = ['NEWTRY1', 'NEWTRY2', 'original_data_echo_wave_search', 'original_data_full_wave_search',
        'original_data_full_wave_search_HIGH_THRESHOLDS', 'original_data_main_wave_search',
        'original_data_main_wave_search_HIGH_THRESHOLDS', 'subtracted_data_full_wave_search_v1']

cWB_files1 = {'H1':'H1_wf_strain', 'L1':'L1_wf_strain', 'V1':'V1_wf_strain'}

# In[ ]:
eventloads = ['GW190521']
for eventload in eventloads:
    if eventload is 'GW190521':
        try:
            # read in data from H1 and L1, if available:
            strain_H1 = frame.read_frame('GW190521_strain_16KHZ_H1.gwf', 'H1:GWOSC-16KHZ_R2_STRAIN')
            strain_L1 = frame.read_frame('GW190521_strain_16KHZ_L1.gwf', 'L1:GWOSC-16KHZ_R2_STRAIN')
            strain_V1 = frame.read_frame('GW190521_strain_16KHZ_V1.gwf', 'V1:GWOSC-16KHZ_R2_STRAIN')
            start_time = strain_H1.start_time
            length = len(strain_H1)
            fs = int(strain_H1.sample_rate)
            time = float(start_time) + np.arange(length)/float(fs)
            
            injection_H1 = frame.read_frame('GW190521_NRSurg_echo_injection_16KHZ_H1.gwf', 'H1:GWOSC-16KHZ_R2_STRAIN')
            injection_L1 = frame.read_frame('GW190521_NRSurg_echo_injection_16KHZ_L1.gwf', 'L1:GWOSC-16KHZ_R2_STRAIN')
            
            GR_L1 = frame.read_frame('GW190521_NRSurg_GR_injection_16KHZ_L1.gwf', 'L1:GWOSC-16KHZ_R2_STRAIN')

            cWB_H1 = frame.read_frame(path1+eventload+'_'+path2[cWB_cnt]+'_'+cWB_files1['H1']+'_cWB_16KHZ.gwf', 'H1:GWOSC-16KHZ_R1_STRAIN')
            cWB_L1 = frame.read_frame(path1+eventload+'_'+path2[cWB_cnt]+'_'+cWB_files1['L1']+'_cWB_16KHZ.gwf', 'L1:GWOSC-16KHZ_R1_STRAIN')
            
            #fp = io.loadfile(file_name,'r')
            fp = h5py.File(file_names['NRSur7dq4 and $\Lambda$'],'r')
            del_t_echo_1 = fp['samples/del_t_echo'][()]
            horizon_freq_1 = fp['samples/horizon_freq'][()]/(2*np.pi)
            
            #fp = h5py.File(file_names['NRSur7dq4 LVC and $\Lambda$'],'r')
            #del_t_echo_3 = fp['samples/del_t_echo'][()]
            #horizon_freq_3 = fp['samples/horizon_freq'][()]/(2*np.pi)
            
            #fp = h5py.File(file_names['SEOBNRv4PHM LVC'],'r')
            #del_t_echo_4 = fp['samples/del_t_echo'][()]
            #horizon_freq_4 = fp['samples/horizon_freq'][()]/(2*np.pi)
            
            #fp = h5py.File(file_names['IMRPhenomPv3HM LVC'],'r')
            #del_t_echo_5 = fp['samples/del_t_echo'][()]
            #horizon_freq_5 = fp['samples/horizon_freq'][()]/(2*np.pi)

            params = ['del_t_echo', 'horizon_freq']
            nparams = len(params)

        except:
            quit()



        #Shift the detectors with offset 2.7ms
        strain_H1_shift = copy.copy(np.roll(strain_H1,-28))
        injection_H1_shift = copy.copy(np.roll(injection_H1,-28))
        cWB_H1_shift = copy.copy(np.roll(cWB_H1,-28))
    
    # In[ ]:
    # -- To calculate the PSD of the data, choose an overlap and a window (common to all detectors)
    #   that minimizes "spectral leakage" https://en.wikipedia.org/wiki/Spectral_leakage
    NFFT = 4*fs
    psd_window = np.blackman(NFFT)
    # and a 50% overlap:
    NOVL = NFFT/2

    # to remove effects at the beginning and end of the data stretch, window the data
    # https://en.wikipedia.org/wiki/Window_function#Tukey_window
    try:   dwindow = signal.tukey(strain_H1_shift.size, alpha=1./8)  # Tukey window preferred, but requires recent scipy version
    except: dwindow = signal.blackman(strain_H1_shift.size)          # Blackman window OK if Tukey is not available


dt = time[1] - time[0]

# index into the strain time series for this time interval:
#indxt_whiten = np.where((time_H1_GW170817 >= tevent-deltatwi) & (time_H1_GW170817 < tevent+deltatwf))
indxt = np.where((time >= tmerger-deltati) & (time < tmerger+deltatf))
indxt_echo = np.where(time >= tmerger+0.5)

make_psds = 1
if make_psds:
    # number of sample for the fast fourier transform:
    NFFT = 4*fs
    Pxx_H1, freqs = mlab.psd(strain_H1_shift, Fs = fs, NFFT = NFFT)
    Pxx_L1, freqs = mlab.psd(strain_L1, Fs = fs, NFFT = NFFT)
    Pxx_V1, freqs = mlab.psd(strain_V1, Fs = fs, NFFT = NFFT)

    # We will use interpolations of the ASDs computed above for whitening:
    psd_H1 = interp1d(freqs, Pxx_H1)
    psd_L1 = interp1d(freqs, Pxx_L1)
    psd_V1 = interp1d(freqs, Pxx_V1)

mk_plots =1
if mk_plots:
    # plot the ASDs, with the template overlaid:
    cWB_L1_fft = np.fft.fft(cWB_L1*dwindow) / fs
    cWB_H1_fft = np.fft.fft(cWB_H1*dwindow) / fs
    injection_L1_fft = np.fft.fft(injection_L1*dwindow) / fs
    GR_L1_fft = np.fft.fft(GR_L1*dwindow) / fs
    datafreq = np.fft.fftfreq(np.array(cWB_L1).size)*fs
    d_eff = 1
    cWB_L1_f = np.absolute(cWB_L1_fft)*np.sqrt(np.abs(datafreq)) / d_eff
    cWB_H1_f = np.absolute(cWB_H1_fft)*np.sqrt(np.abs(datafreq)) / d_eff
    injection_L1_f = np.absolute(injection_L1_fft)*np.sqrt(np.abs(datafreq)) / d_eff
    GR_L1_f = np.absolute(GR_L1_fft)*np.sqrt(np.abs(datafreq)) / d_eff
    f_min = 30.
    f_max = 80. 
    plt.figure(figsize=(6,4))
    plt.semilogy(freqs, np.sqrt(Pxx_L1),'cornflowerblue',label='L1 strain')
    plt.semilogy(freqs, np.sqrt(Pxx_H1),'r',label='H1 strain', alpha=0.7)
    #plt.semilogy(freqs, np.sqrt(Pxx_V1),'darkorchid',label='V1 strain', alpha=0.7)
    plt.semilogy(datafreq, cWB_L1_f, 'k', label='cWB L1 (strain(f)xsqrt(f))', alpha = 0.8)
    #plt.semilogy(datafreq, cWB_H1_f, 'gray', label='cWB H1 strain(f)xsqrt(f)', alpha=0.8)
    plt.semilogy(datafreq, injection_L1_f, 'green', label='PyCBC max likelihood Boltzmann echo (strain(f)*sqrt(f))', alpha=0.5)
    plt.semilogy(datafreq, GR_L1_f, 'brown', label='PyCBC max likelihood GR waveform (strain(f)*sqrt(f))', alpha=1.0)
    plt.axis([f_min, f_max, 1e-24, 3e-22])
    plt.grid('on')
    plt.ylabel('Strain [1/sqrt(Hz)]')
    plt.xlabel('Frequency (Hz)')
    plt.legend(loc='lower left')
    plt.title('Advanced LIGO strain data near '+eventname)
    plt.savefig('GW190521_ASDs.pdf')



fig = plt.figure(figsize=(6,4)); ax = fig.gca()
ax.plot(time[indxt]-float(tmerger), np.array(cWB_L1[indxt]), label='cWB reconstructed waveform', linewidth=0.3)
ax.plot(time[indxt]-float(tmerger), np.array(injection_L1[indxt]), label='PyCBC maximum likelihood Boltzmann echoes waveform', linewidth=0.3, alpha=1.0)
plt.xlim(-0.5,3)
ax.legend()
ax.set_ylabel('L1 strain')
ax.set_xlabel('time (s) since '+str(tmerger))
fig.savefig('injection_L1.pdf')


inj_H1=np.array(injection_H1[indxt])
inj_H1_echo=np.array(injection_H1[indxt_echo])
print('Echo energy/Event energy = ', 100*(sum( abs((inj_H1_echo-np.roll(inj_H1_echo,1))**2)  ))/(sum( (abs(inj_H1-np.roll(inj_H1,1)))**2  )) , '%')

Energy=0
if Energy == 1:
    from pycbc import waveform
    from pycbc.inference import io
    from pycbc.waveform.generator import TDomainCBCGenerator, FDomainDetFrameGenerator
    from pycbc import filter
    from pycbc import psd as pypsd
    filter_fmin = 20.
    filter_fmax = 1024.
    fp = io.loadfile("inference.hdf", "r")
    variable_args = fp["samples"].keys()
    maxLpt = np.argmax(fp['samples']['loglikelihood'])
    static_args = {'approximant': 'IMRPhenomXPHM','f_lower': 10.0,'f_ref': 20.0}
    generator = FDomainDetFrameGenerator(TDomainCBCGenerator, strain_H1.start_time, detectors=['H1', 'L1', 'V1'], variable_args=variable_args, delta_t=strain_H1.delta_t, delta_f=strain_H1.delta_f, **static_args)
    htildes = generator.generate(**{p: fp['samples'][p][maxLpt] for p in variable_args})
    cplx_hd = 0j
    hh = 0.
    psd_seg_len = 16
    psd_seg_stride = 8
    inverse_psd_len = 4
    psd_sample_rate=4096
    psds = {}
    #for ifo, d in psd_data.items():
    #    psd = pypsd.welch(d, seg_len=psd_seg_len*sample_rate, seg_stride=psd_seg_stride*sample_rate)
    for ifo,d in {'H1':strain_H1, 'L1':strain_L1, 'V1':strain_V1}.items():
        psd = pypsd.welch(d, seg_len=psd_seg_len*psd_sample_rate, seg_stride=psd_seg_stride*psd_sample_rate)
        # truncate
        psd = pypsd.inverse_spectrum_truncation(psd, inverse_psd_len*psd_sample_rate, low_frequency_cutoff=15.)
        # interpolate to needed df
        psd = pypsd.interpolate(psd, d.delta_f)
        psds[ifo] = psd
        htilde = htildes[ifo]
        dtilde = d.to_frequencyseries()
        htilde.resize(len(dtilde))
        cplx_hd += filter.overlap_cplx(htilde, dtilde, psd=psds[ifo], low_frequency_cutoff=filter_fmin, high_frequency_cutoff=filter_fmax,normalized=False)
        hh += filter.sigmasq(htilde, psd=psds[ifo], low_frequency_cutoff=filter_fmin,high_frequency_cutoff=filter_fmax)
    print(cplx_hd)
    phase_shift = np.angle(cplx_hd)
    print("Phase shift: {0:.9f}".format(phase_shift))
    #cplx_loglr = cplx_hd - 0.5*hh
    #print(cplx_loglr)
    #print("SNR: {0:.9f}".format((2*cplx_loglr.real)**0.5))
    waveforms = {}
    for ifo, htilde in htildes.items():
        htildes[ifo] = htilde * np.exp(1j * phase_shift)
        h = waveform.fd_to_td(htildes[ifo], delta_t=strain_H1.delta_t, left_window=(10.,15.))
        waveforms[ifo] = h
        
        fig = plt.figure(); ax = fig.gca()
        ax.plot(time-tmerger, waveforms[ifo], label='IMR'+ifo)
        plt.xlim(-1,1)
        ax.legend()
        ax.set_ylabel('Most likelihood waveform')
        ax.set_xlabel('time (s) since '+str(tmerger))
    fig.savefig('IMR.png')

    event_H1=np.array(waveforms['H1'][indxt])
    inj_H1_echo=np.array(injection_H1[indxt_echo])
    print('Echo energy/Event energy = ', 100*(sum( abs((inj_H1_echo-np.roll(inj_H1_echo,1))**2)  ))/(sum( (abs(event_H1-np.roll(event_H1,1)))**2  )) , '%')



# function to whiten data
def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)

    # whitening: transform to freq domain, divide by asd, then transform back, 
    # taking care to get normalization right.
    hf = np.fft.rfft(strain)
    #white_hf = hf / (np.sqrt(interp_psd(freqs) /dt/2.))
    ##white_hf = hf / ((interp_psd(freqs) /dt/2.))
    white_hf = hf
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht


plottype = "png"

whiten_data = 1
if whiten_data:
    # now whiten the data from H1 and L1, and the template (use H1 PSD):
    strain_H1_whiten = whiten(strain_H1_shift,psd_H1,dt)
    strain_L1_whiten = whiten(strain_L1,psd_L1,dt)
    injection_H1_whiten = whiten(injection_H1_shift,psd_H1,dt)
    injection_L1_whiten = whiten(injection_L1,psd_L1,dt)
    cWB_H1_whiten = whiten(cWB_H1_shift,psd_H1,dt)
    cWB_L1_whiten = whiten(cWB_L1,psd_L1,dt)

print(len(strain_H1_whiten),len(injection_H1_whiten))

# pick a shorter FTT time interval, like 1/16 of a second:
NFFT = int(fs/1.0)
# and with a lot of overlap, to resolve short-time features:
NOVL = int(NFFT*15/16.0)
# choose a window that minimizes "spectral leakage"
# (https://en.wikipedia.org/wiki/Spectral_leakage)
window = np.blackman(NFFT)

spec_strain_H1, freqs, bins = mlab.specgram(strain_H1_whiten[indxt], NFFT=NFFT, Fs=fs, window=window, noverlap=NOVL, mode='complex')
spec_strain_L1, freqs, bins = mlab.specgram(strain_L1_whiten[indxt], NFFT=NFFT, Fs=fs, window=window, noverlap=NOVL, mode='complex')

spec_inj_H1, freqs, bins = mlab.specgram(injection_H1_whiten[indxt], NFFT=NFFT, Fs=fs, window=window, noverlap=NOVL, mode='complex')
spec_inj_L1, freqs, bins = mlab.specgram(injection_L1_whiten[indxt], NFFT=NFFT, Fs=fs, window=window, noverlap=NOVL, mode='complex')

spec_cWB_H1, freqs, bins = mlab.specgram(cWB_H1_whiten[indxt], NFFT=NFFT, Fs=fs, window=window, noverlap=NOVL, mode='complex')
spec_cWB_L1, freqs, bins = mlab.specgram(cWB_L1_whiten[indxt], NFFT=NFFT, Fs=fs, window=window, noverlap=NOVL, mode='complex')

# Following two lines depends on operating system or version of python that must either keeped or removed


fmin=0
fmax=250

Spectrum_strain=[]
Spectrum_inj=[]
Spectrum_cWB=[]

for i in range (0,len(spec_strain_H1[1,:]+1)):
        
    ind1=np.arange(1,len(spec_strain_H1[:,i][fmin*2:fmax*2])+1,1)
    interp_spec1=interp1d(ind1,(spec_strain_H1*np.conjugate(spec_strain_L1))[:,i][fmin*2:fmax*2])
    ind_interp_scale1=np.arange(1,len(spec_strain_H1[:,i][fmin*2:fmax*2]),2)
    Spec_scale1=interp_spec1(ind_interp_scale1)
    Spectrum_strain.append(list((spec_strain_H1*np.conjugate(spec_strain_L1))[:,i][fmin:fmax]))
    
    ind2=np.arange(1,len(spec_inj_H1[:,i][fmin*2:fmax*2])+1,1)
    interp_spec1=interp1d(ind1,(spec_inj_H1*np.conjugate(spec_inj_L1))[:,i][fmin*2:fmax*2])
    ind_interp_scale1=np.arange(1,len(spec_inj_H1[:,i][fmin*2:fmax*2]),2)
    Spec_scale1=interp_spec1(ind_interp_scale1)
    Spectrum_inj.append(list((spec_inj_H1*np.conjugate(spec_inj_L1))[:,i][fmin:fmax]))

    ind3=np.arange(1,len(spec_cWB_H1[:,i][fmin*2:fmax*2])+1,1)
    interp_spec3=interp1d(ind3,(spec_cWB_H1*np.conjugate(spec_cWB_L1))[:,i][fmin*2:fmax*2])
    ind_interp_scale3=np.arange(1,len(spec_cWB_H1[:,i][fmin*2:fmax*2]),2)
    Spec_scale3=interp_spec3(ind_interp_scale3)
    Spectrum_cWB.append(list((spec_cWB_H1*np.conjugate(spec_cWB_L1))[:,i][fmin:fmax]))

Spectrum_strain=copy.copy(Spectrum_strain)
Spectrum_strain=np.array(Spectrum_strain)

Spectrum_inj=copy.copy(Spectrum_inj)
Spectrum_inj=np.array(Spectrum_inj)

Spectrum_cWB=copy.copy(Spectrum_cWB)
Spectrum_cWB=np.array(Spectrum_cWB)


#plottype = "pdf"
#spec_cmap = 'jet'
spec_cmap = 'magma'
make_plots=0
if make_plots:
    # Plot the H1xL1 whitened spectrogram around the signal
    plt.figure(figsize=(10,6))
    bounds = np.linspace(2e-5, 2e-4, 100)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    #spec_H1, freqs, bins, im = plt.specgram(injection_H1[indxt_0], NFFT=NFFT, Fs=fs, window=window,
    #                                        noverlap=NOVL, norm=norm, origin='upper', cmap=spec_cmap, xextent=[-deltat,deltat])
    plt.pcolormesh(bins-deltati, freqs[fmin:fmax], -np.real(transpose(Spectrum_inj)), norm=norm, shading='auto', cmap=spec_cmap)
    plt.xlabel('time (s) since '+str(tmerger))
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()
    plt.title('Most likelihood '+eventname+' H1xL1 echo waveform')
    plt.contour(bins-deltati, freqs[fmin:fmax], -np.real(transpose(Spectrum_inj)), levels=[5e-5,5e-4], colors=['#C0C0C0','red'])
    #plt.clabel(contours)
    plt.axis([-1, 3, 0, 110])
    plt.savefig('contour.png')

    plt.figure(figsize=(10,5))
    bounds = np.linspace(1.7e-4, 4e-4, 100)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    plt.pcolormesh(bins-deltati, freqs[fmin:fmax], -np.real(transpose(Spectrum_strain)), norm=norm, shading='auto', cmap='magma')
    plt.contour(bins-deltati, freqs[fmin:fmax], -np.real(transpose(Spectrum_inj)), levels=[5e-5,5e-4], colors=['#C0C0C0','red'])
    #plt.ylim(0,250)
    #plt.axis([84-30, 84+100, 0, 50
    plt.axis([-0.5, 4, 25, 80])
    plt.xlabel('$t-t_{\\rm{merger}}$ [sec]')
    plt.ylabel('Frequency [Hz]')
    #plt.colorbar(orientation='vertical')
    #plt.legend(loc='lower right', handles=handles)
    plt.savefig('spectrogram_color_3_3.png')



width_ratios = [5, 1]
height_ratios = [1, 5]
title_lbls = {'del_t_echo':'$\Delta t_{echo}$', 'horizon_freq':'$\Omega_{H}/\pi$'}
title_lbls2 = {'del_t_echo':'$2\times\Delta t_{echo}$', 'horizon_freq':'$\Omega_{H}/\pi$'}

mins = {'del_t_echo':-2, 'horizon_freq':30}
maxs = {'del_t_echo':2, 'horizon_freq':80}

#print(np.min(Spectrum_cWB))
fig, axis_dict = scatter_histograms.create_axes_grid(
            params, #labels=lbls,
            width_ratios=width_ratios, height_ratios=height_ratios,
            no_diagonals=False)

handles1 = []
handles2 = []
# Plot 2D spectogram
ax, _, _ = axis_dict[('del_t_echo','horizon_freq')]
#bounds = np.linspace(1.7e-4, 4e-4, 100)
#bounds = np.linspace(2e-46,4e-46,100)
#norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
ax.pcolormesh(bins-deltati, freqs[fmin:fmax], -np.real(transpose(Spectrum_cWB)), shading='auto', cmap='magma')
ax.contour(bins-deltati, freqs[fmin:fmax], -np.real(transpose(Spectrum_inj)), levels=[1e-46,4e-46], colors=['white', 'red'])
handles1.append(patches.Patch(color='#cccccc'))
handles1.append(patches.Patch(color='white'))
handles1.append(patches.Patch(color='red'))
fig.legend(loc=(0.13,0.615), ncol=1, handles=handles1, labels=['Maximum likelihood Boltzmann echo waveform (PyCBC)', 'Pixel strain amplitude $= 10^{-23}$', 'Pixel strain amplitude $= 2\\times 10^{-23}$'])
ax.axis([-2, 2, 30, 80])
ax.set_xlabel('$t-t_{\\rm{merger}}$ [sec]')
ax.set_ylabel('Frequency [Hz]')

samples1echo_1 = {'del_t_echo':del_t_echo_1, 'horizon_freq':horizon_freq_1}
samples2echo_1 = {'del_t_echo':2*del_t_echo_1, 'horizon_freq':horizon_freq_1}

mask = np.where((horizon_freq_1>20) & (horizon_freq_1<100))
samples_1D = {'del_t_echo':del_t_echo_1, 'horizon_freq':horizon_freq_1[mask]}

#samples1echo_2 = {'del_t_echo':del_t_echo_2, 'horizon_freq':horizon_freq_2}
#samples2echo_2 = {'del_t_echo':2*del_t_echo_2, 'horizon_freq':horizon_freq_2}

#samples1echo_3 = {'del_t_echo':del_t_echo_3, 'horizon_freq':horizon_freq_3}
#samples2echo_3 = {'del_t_echo':2*del_t_echo_3, 'horizon_freq':horizon_freq_3}

#samples1echo_4 = {'del_t_echo':del_t_echo_4, 'horizon_freq':horizon_freq_4}
#samples2echo_4 = {'del_t_echo':2*del_t_echo_4, 'horizon_freq':horizon_freq_4}

#samples1echo_5 = {'del_t_echo':del_t_echo_5, 'horizon_freq':horizon_freq_5}
#samples2echo_5 = {'del_t_echo':2*del_t_echo_5, 'horizon_freq':horizon_freq_5}

# Plot 1D histograms
for pi, param in enumerate(params):
    ax, _, _ = axis_dict[param, param]
    rotated = nparams == 2 and pi == nparams-1
    scatter_histograms.create_marginalized_hist(ax,samples_1D[param], label=title_lbls[param], color = 'lightseagreen',
                                                  fillcolor=None,linecolor='lightseagreen',
                                                  rotated=rotated, plot_min=mins[param], plot_max=maxs[param],
                                                  percentiles=[5,95])
handles2.append(patches.Patch(color='lightseagreen'))
# Plot 1D histograms
#ax, _, _ = axis_dict['del_t_echo', 'del_t_echo']
#rotated = nparams == 1 and pi == nparams-1
#scatter_histograms.create_marginalized_hist(ax,samples2echo_1['del_t_echo'], label=title_lbls2['del_t_echo'], color = 'lightseagreen',
#                                                  fillcolor=None,linecolor='lightseagreen',
#                                                  rotated=rotated, plot_min=mins['del_t_echo'], plot_max=maxs['del_t_echo'],
#                                                  percentiles=[5,95])

# Plot 1D histograms
#for pi, param in enumerate(params):
#    ax, _, _ = axis_dict[param, param]
#    rotated = nparams == 2 and pi == nparams-1
#    scatter_histograms.create_marginalized_hist(ax,samples1echo_2[param], label=title_lbls[param], color = 'blue',
#                                                  fillcolor=None,linecolor='blue',
#                                                  rotated=rotated, plot_min=mins[param], plot_max=maxs[param],
#                                                  percentiles=[5,95])
# Plot 1D histograms
#ax, _, _ = axis_dict['del_t_echo', 'del_t_echo']
#rotated = nparams == 1 and pi == nparams-1
#scatter_histograms.create_marginalized_hist(ax,samples2echo_2['del_t_echo'], label=title_lbls2['del_t_echo'], color = 'blue',
#                                                  fillcolor=None,linecolor='blue',
#                                                  rotated=rotated, plot_min=mins['del_t_echo'], plot_max=maxs['del_t_echo'],
#                                                  percentiles=[5,95])

# Plot 1D histograms
#for pi, param in enumerate(params):
#    ax, _, _ = axis_dict[param, param]
#    rotated = nparams == 2 and pi == nparams-1
#    scatter_histograms.create_marginalized_hist(ax,samples1echo_3[param], label=title_lbls[param], color = 'lime',
#                                                  fillcolor=None,linecolor='lime',
#                                                  rotated=rotated, plot_min=mins[param], plot_max=maxs[param],
#                                                  percentiles=[5,95])
# Plot 1D histograms
#ax, _, _ = axis_dict['del_t_echo', 'del_t_echo']
#rotated = nparams == 1 and pi == nparams-1
#scatter_histograms.create_marginalized_hist(ax,samples2echo_3['del_t_echo'], label=title_lbls2['del_t_echo'], color = 'lime',
#                                                  fillcolor=None,linecolor='lime',
#                                                  rotated=rotated, plot_min=mins['del_t_echo'], plot_max=maxs['del_t_echo'],
#                                                  percentiles=[5,95])

# Plot 1D histograms
#for pi, param in enumerate(params):
#    ax, _, _ = axis_dict[param, param]
#    rotated = nparams == 2 and pi == nparams-1
#    scatter_histograms.create_marginalized_hist(ax,samples1echo_4[param], label=title_lbls[param], color = 'magenta',
#                                                  fillcolor=None,linecolor='magenta',
#                                                  rotated=rotated, plot_min=mins[param], plot_max=maxs[param],
#                                                  percentiles=[5,95])
# Plot 1D histograms
#ax, _, _ = axis_dict['del_t_echo', 'del_t_echo']
#rotated = nparams == 1 and pi == nparams-1
#scatter_histograms.create_marginalized_hist(ax,samples2echo_4['del_t_echo'], label=title_lbls2['del_t_echo'], color = 'magenta',
#                                                  fillcolor=None,linecolor='magenta',
#                                                  rotated=rotated, plot_min=mins['del_t_echo'], plot_max=maxs['del_t_echo'],
#                                                  percentiles=[5,95])

# Plot 1D histograms
#for pi, param in enumerate(params):
#    ax, _, _ = axis_dict[param, param]
#    rotated = nparams == 2 and pi == nparams-1
#    scatter_histograms.create_marginalized_hist(ax,samples1echo_5[param], label=title_lbls[param], color = 'gold',
#                                                  fillcolor=None,linecolor='gold',
#                                                  rotated=rotated, plot_min=mins[param], plot_max=maxs[param],
#                                                  percentiles=[5,95])
# Plot 1D histograms
#ax, _, _ = axis_dict['del_t_echo', 'del_t_echo']
#rotated = nparams == 1 and pi == nparams-1
#scatter_histograms.create_marginalized_hist(ax,samples2echo_5['del_t_echo'], label=title_lbls2['del_t_echo'], color = 'gold',
#                                                  fillcolor=None,linecolor='gold',
#                                                  rotated=rotated, plot_min=mins['del_t_echo'], plot_max=maxs['del_t_echo'],
#                                                  percentiles=[5,95])
'''
# Plot 2D contours
ax, _, _ = axis_dict[('del_t_echo','horizon_freq')]
scatter_histograms.create_density_plot(
                'del_t_echo', 'horizon_freq', samples1echo_1, #cmap='gray',
                plot_density=False, plot_contours=True, #cmap=density-cmap,
                percentiles=[50,90],
                contour_color='lightseagreen', #axis=[-5, 5, 25, 80],
                #mins['horizon_freq'], maxs['horizon_freq']],
                #exclude_region=exclude_region,
                ax=ax, use_kombine=False)
# Plot 2D contours
ax, _, _ = axis_dict[('del_t_echo','horizon_freq')]
scatter_histograms.create_density_plot(
                'del_t_echo', 'horizon_freq', samples2echo_1, #cmap='gray',
                plot_density=False, plot_contours=True, #cmap=density-cmap,
                percentiles=[50,90],
                contour_color='lightseagreen', #axis=[-5, 5, 25, 80],
                #mins['horizon_freq'], maxs['horizon_freq']],
                #exclude_region=exclude_region,
                ax=ax, use_kombine=False)
handles2.append(patches.Patch(color='lightseagreen'))
ax.set_xlabel('$t-t_{\\rm{merger}}$ [sec]')
ax.set_ylabel('Frequency [Hz]')
'''
#fig.legend(loc='upper right', handles=handles, labels=['Most likelihood echo', 'Expected 90% region'])

# Plot 2D contours
#ax, _, _ = axis_dict[('del_t_echo','horizon_freq')]
#scatter_histograms.create_density_plot(
#                'del_t_echo', 'horizon_freq', samples1echo_2, #cmap='gray',
#                plot_density=False, plot_contours=True, #cmap=density-cmap,
#                percentiles=[90],
#                contour_color='blue', #axis=[-5, 5, 25, 80],
#                #mins['horizon_freq'], maxs['horizon_freq']],
#                #exclude_region=exclude_region,
#                ax=ax, use_kombine=False)
# Plot 2D contours
#ax, _, _ = axis_dict[('del_t_echo','horizon_freq')]
#scatter_histograms.create_density_plot(
#                'del_t_echo', 'horizon_freq', samples2echo_2, #cmap='gray',
#                plot_density=False, plot_contours=True, #cmap=density-cmap,
#                percentiles=[90],
#                contour_color='blue', #axis=[-5, 5, 25, 80],
#                #mins['horizon_freq'], maxs['horizon_freq']],
#                #exclude_region=exclude_region,
#                ax=ax, use_kombine=False)
#handles.append(patches.Patch(color='blue'))
#ax.set_xlabel('$t-t_{\\rm{merger}}$ [sec]')
#ax.set_ylabel('Frequency [Hz]')
##fig.legend(loc='upper right', handles=handles, labels=['Most likelihood echo', 'Expected 90% region'])

# Plot 2D contours
#ax, _, _ = axis_dict[('del_t_echo','horizon_freq')]
#scatter_histograms.create_density_plot(
#                'del_t_echo', 'horizon_freq', samples1echo_3, #cmap='gray',
#                plot_density=False, plot_contours=True, #cmap=density-cmap,
#                percentiles=[50,90],
#                contour_color='lime', #axis=[-5, 5, 25, 80],
#                #mins['horizon_freq'], maxs['horizon_freq']],
#                #exclude_region=exclude_region,
#                ax=ax, use_kombine=False)
# Plot 2D contours
#ax, _, _ = axis_dict[('del_t_echo','horizon_freq')]
#scatter_histograms.create_density_plot(
#                'del_t_echo', 'horizon_freq', samples2echo_3, #cmap='gray',
#                plot_density=False, plot_contours=True, #cmap=density-cmap,
#                percentiles=[50,90],
#                contour_color='lime', #axis=[-5, 5, 25, 80],
#                #mins['horizon_freq'], maxs['horizon_freq']],
#                #exclude_region=exclude_region,
#                ax=ax, use_kombine=False)
#handles2.append(patches.Patch(color='lime'))
#ax.set_xlabel('$t-t_{\\rm{merger}}$ [sec]')
#ax.set_ylabel('Frequency [Hz]')
#fig.legend(loc='upper right', handles=handles, labels=['Most likelihood echo', 'Expected 90% region'])

# Plot 2D contours
#ax, _, _ = axis_dict[('del_t_echo','horizon_freq')]
#scatter_histograms.create_density_plot(
#                'del_t_echo', 'horizon_freq', samples1echo_4, #cmap='gray',
#                plot_density=False, plot_contours=True, #cmap=density-cmap,
#                percentiles=[90],
#                contour_color='magenta', #axis=[-5, 5, 25, 80],
#                #mins['horizon_freq'], maxs['horizon_freq']],
#                #exclude_region=exclude_region,
#                ax=ax, use_kombine=False)
# Plot 2D contours
#ax, _, _ = axis_dict[('del_t_echo','horizon_freq')]
#scatter_histograms.create_density_plot(
#                'del_t_echo', 'horizon_freq', samples2echo_4, #cmap='gray',
#                plot_density=False, plot_contours=True, #cmap=density-cmap,
#                percentiles=[90],
#                contour_color='magenta', #axis=[-5, 5, 25, 80],
#                #mins['horizon_freq'], maxs['horizon_freq']],
#                #exclude_region=exclude_region,
#                ax=ax, use_kombine=False)
#handles.append(patches.Patch(color='magenta'))
#ax.set_xlabel('$t-t_{\\rm{merger}}$ [sec]')
#ax.set_ylabel('Frequency [Hz]')
##fig.legend(loc='upper right', handles=handles, labels=['Most likelihood echo', 'Expected 90% region'])

# Plot 2D contours
#ax, _, _ = axis_dict[('del_t_echo','horizon_freq')]
#scatter_histograms.create_density_plot(
#                'del_t_echo', 'horizon_freq', samples1echo_5, #cmap='gray',
#                plot_density=False, plot_contours=True, #cmap=density-cmap,
#                percentiles=[90],
#                contour_color='yellow', #axis=[-5, 5, 25, 80],
#                #mins['horizon_freq'], maxs['horizon_freq']],
#                #exclude_region=exclude_region,
#                ax=ax, use_kombine=False)
## Plot 2D contours
#ax, _, _ = axis_dict[('del_t_echo','horizon_freq')]
#scatter_histograms.create_density_plot(
#                'del_t_echo', 'horizon_freq', samples2echo_5, #cmap='gray',
#                plot_density=False, plot_contours=True, #cmap=density-cmap,
#                percentiles=[90],
#                contour_color='yellow', #axis=[-5, 5, 25, 80],
#                #mins['horizon_freq'], maxs['horizon_freq']],
#                #exclude_region=exclude_region,
#                ax=ax, use_kombine=False)
#handles.append(patches.Patch(color='gold'))
#ax.set_xlabel('$t-t_{\\rm{merger}}$ [sec]')
#ax.set_ylabel('Frequency [Hz]')
fig.legend(loc=(0.13,0.8), ncol=2, handles=handles2, labels=['Predicted Planckian echo'])
#fig.legend(loc=(0.13,0.65), ncol=2, handles=handles2, labels=['Expected from Planckian echo (NRSur7dq4)'])
fig.savefig('cWB_PyCBC_NRSur.pdf')


