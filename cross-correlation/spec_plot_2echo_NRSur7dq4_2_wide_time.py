#! /usr/bin/python2.7

## index into the strain time series for this time interval:

deltati = 26
deltatf = 26


# In[ ]:

# Standard python numerical analysis imports:
import numpy as np
from scipy import signal
from pycbc.waveform.generator import TDomainBoltzmannEchoesGenerator3, TDomainBoltzmannEchoesGenerator3_0R, FDomainCBCGenerator, FDomainDetFrameGenerator
from pycbc import psd as pypsd
from pycbc import filter
from pycbc import frame
from pycbc import waveform
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



fig = plt.figure(); ax = fig.gca()
ax.plot(time[indxt]-float(tmerger), np.array(injection_H1[indxt]), label='ringdown+echoes')
ax.plot(time[indxt_echo]-float(tmerger), np.array(injection_H1[indxt_echo]), label='echoes')
plt.xlim(-1,3)
ax.legend()
ax.set_ylabel('Most likelihood Boltzmann waveform for H1')
ax.set_xlabel('time (s) since '+str(tmerger))
fig.savefig('injection_H1.png')


EchoSNR=0
if EchoSNR == 1:
    from pycbc import waveform
    from pycbc.inference import io
    from pycbc.waveform.generator import TDomainCBCGenerator, FDomainDetFrameGenerator
    from pycbc import filter
    from pycbc import psd as pypsd
    filter_fmin = 20.
    filter_fmax = 1024.
    fp = io.loadfile("NRSur7dq4_2_Jahed.hdf", "r")
    variable_args = fp["samples"].keys()
    maxLpt = np.argmax(fp['samples']['loglikelihood'])
    #static_args = {'approximant': 'NRSur7dq4','f_lower': 20.0,'f_ref': 20.0}
    static_args = {'approximant': 'TDBechoes3', 'imr_approximant': 'NRSur7dq4', 'n_echoes': 2, 'alpha': 1, 'include_main_event': False, 'f_lower': 20.0,'f_ref': 20.0}
    generator = FDomainDetFrameGenerator(TDomainBoltzmannEchoesGenerator3, strain_H1.start_time, detectors=['H1', 'L1', 'V1'], variable_args=variable_args, delta_t=4*strain_H1.delta_t, delta_f=strain_H1.delta_f, **static_args)
    htildes = generator.generate(**{p: fp['samples'][p][maxLpt] for p in variable_args})
    print('st:',htildes['H1'].start_time, len(htildes['H1']))
    cplx_hd = 0j
    hh = 0.
    psd_seg_len = 16
    psd_seg_stride = 8
    inverse_psd_len = 4
    psd_sample_rate=4096
    psds = {}
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
    phase_shift = -0.157771551
    cplx_loglr = cplx_hd - 0.5*hh
    print(cplx_loglr)
    print("SNR: {0:.9f}".format((2*cplx_loglr.real)**0.5))
    for ifo, h in htildes.items():
        htildes[ifo] = h * np.exp(1j * phase_shift)

    cplx_hd = 0j
    hh = 0.
    plt.figure()
    for ifo,d in {'H1':strain_H1, 'L1':strain_L1, 'V1':strain_V1}.items():
        htilde = htildes[ifo]
        thtilde = htilde.to_timeseries()
        print('argmax:',np.argmax(thtilde))
        plt.plot(thtilde[8540000:8580000])
        dtilde = d.to_frequencyseries()
        htilde.resize(len(dtilde))
        cplx_hd += filter.overlap_cplx(htilde, dtilde, psd=psds[ifo], low_frequency_cutoff=filter_fmin,
                                       high_frequency_cutoff=filter_fmax,
                                       normalized=False)
        hh += filter.sigmasq(htilde, psd=psds[ifo], low_frequency_cutoff=filter_fmin,
                             high_frequency_cutoff=filter_fmax)
    plt.savefig('mytest.png')
    check_phase_shift = np.angle(cplx_hd)
    print("Phase shift: {0:.9f}".format(check_phase_shift))
    cplx_loglr = cplx_hd - 0.5*hh
    print("SNR: {0:.9f}".format((2*cplx_loglr.real)**0.5))


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
    white_hf = hf / (np.sqrt(interp_psd(freqs) /dt/2.))
    #white_hf = hf / ((interp_psd(freqs) /dt/2.))
    #white_hf = hf
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


# Following two lines depends on operating system or version of python that must either keeped or removed


fmin=0
fmax=250

Spectrum_strain=[]
Spectrum_inj=[]

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


Spectrum_strain=copy.copy(Spectrum_strain)
Spectrum_strain=np.array(Spectrum_strain)

Spectrum_inj=copy.copy(Spectrum_inj)
Spectrum_inj=np.array(Spectrum_inj)




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


#samples1echo1 = {'del_t_echo':del_t_echo_1, 'horizon_freq':horizon_freq_1}
make_plots = 1
if make_plots:
    import matplotlib.colors as colors
    plt.figure(figsize=(15,3))
    bounds = np.linspace(3.9e-4, 4e-4, 100)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    plt.pcolormesh(bins-deltati, freqs[fmin:fmax], -np.real(transpose(Spectrum_strain)), norm=norm, shading='auto', cmap='magma')
    #plt.contour(bins-deltati, freqs[fmin:fmax], -np.real(transpose(Spectrum_inj)), levels=[5e-5,5e-4], colors=['white','red'], linewidths=0.5)
    #scatter_histograms.create_density_plot('del_t_echo', 'horizon_freq', samples1echo1, plot_density=False, plot_contours=True, percentiles=[90], contour_color='lightseagreen', use_kombine=False)
    #plt.ylim(0,250)
    #plt.axis([84-30, 84+100, 0, 50
    plt.axis([-25, 25, 25, 80])
    plt.xlabel('$t-t_{\\rm{merger}}$ [sec]')
    plt.ylabel('Frequency [Hz]')
    #plt.colorbar(orientation='vertical')
    #plt.legend(loc='lower right', handles=handles)
    plt.savefig('spectrogram_color_wide_time9.png')



width_ratios = [20, 1]
height_ratios = [20, 20]
title_lbls = {'del_t_echo':'$\Delta t_{echo}$', 'horizon_freq':'$\Omega_{H}/\pi$'}
title_lbls2 = {'del_t_echo':'$2\times\Delta t_{echo}$', 'horizon_freq':'$\Omega_{H}/\pi$'}


mins = {'del_t_echo':-25, 'horizon_freq':25}
maxs = {'del_t_echo':25, 'horizon_freq':80}


fig, axis_dict = scatter_histograms.create_axes_grid(
            params, #labels=lbls,
            width_ratios=width_ratios, height_ratios=height_ratios,
            no_diagonals=False)

handles1 = []
handles2 = []
# Plot 2D spectogram
ax, _, _ = axis_dict[('del_t_echo','horizon_freq')]
bounds = np.linspace(1.7e-4, 4e-4, 100)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
ax.pcolormesh(bins-deltati, freqs[fmin:fmax], -np.real(transpose(Spectrum_strain)), norm=norm, shading='auto', cmap='magma')
#ax.contour(bins-deltati, freqs[fmin:fmax], -np.real(transpose(Spectrum_inj)), levels=[5e-5,5e-4], colors=['white', 'red'])
handles1.append(patches.Patch(color='#cccccc'))
handles1.append(patches.Patch(color='white'))
handles1.append(patches.Patch(color='red'))
fig.legend(loc='upper right', ncol=1, handles=handles1, labels=['Most likelihood echo waveform', 'pixel SNR$\geq$0.5', 'pixel SNR$\geq$4.9'])
ax.axis([-25, 25, 25, 80])

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
# Plot 1D histograms
ax, _, _ = axis_dict['del_t_echo', 'del_t_echo']
rotated = nparams == 1 and pi == nparams-1
scatter_histograms.create_marginalized_hist(ax,samples2echo_1['del_t_echo'], label=title_lbls2['del_t_echo'], color = 'lightseagreen',
                                                  fillcolor=None,linecolor='lightseagreen',
                                                  rotated=rotated, plot_min=mins['del_t_echo'], plot_max=maxs['del_t_echo'],
                                                  percentiles=[5,95])

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
                percentiles=[90],
                contour_color='lightseagreen', #axis=[-5, 5, 25, 80],
                #mins['horizon_freq'], maxs['horizon_freq']],
                #exclude_region=exclude_region,
                ax=ax, use_kombine=False)
# Plot 2D contours
ax, _, _ = axis_dict[('del_t_echo','horizon_freq')]
scatter_histograms.create_density_plot(
                'del_t_echo', 'horizon_freq', samples2echo_1, #cmap='gray',
                plot_density=False, plot_contours=True, #cmap=density-cmap,
                percentiles=[90],
                contour_color='lightseagreen', #axis=[-5, 5, 25, 80],
                #mins['horizon_freq'], maxs['horizon_freq']],
                #exclude_region=exclude_region,
                ax=ax, use_kombine=False)
handles2.append(patches.Patch(color='lightseagreen'))
ax.set_xlabel('$t-t_{\\rm{merger}}$ [sec]')
ax.set_ylabel('Frequency [Hz]')
#fig.legend(loc='upper right', handles=handles, labels=['Most likelihood echo', 'Expected 90% region'])
'''
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
fig.legend(loc='upper right', ncol=2, handles=handles2, labels=['NRSur7dq4'])

fig.savefig('Xpec_PyCBC_GR_NRSur_wider.png')

