import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

vswr_freqs, vswr = np.loadtxt("./lpda_simulation_results/results_vswr.csv", delimiter = ",", skiprows = 1, unpack = True)
gain_freqs, gain_theta, gain_phi, gain_phi, gain_theta = np.loadtxt("./lpda_simulation_results/results_gain_at_boresight.csv", delimiter = ",", skiprows = 1, unpack = True)
efield_t, efield_theta, efield_phi, ephi, etheta = np.loadtxt("./lpda_simulation_results/results_farfield_e_field_at_boresight.csv", delimiter = ",", skiprows = 1, unpack = True)

abs_s11 = (1.0 - vswr) / (1.0 + vswr)

gain_theta_realized = gain_theta * (1.0 - np.square(abs_s11))
gain_phi_realized = gain_phi * (1.0 - np.square(abs_s11))

gain_theta_realized[np.isnan(gain_theta_realized)] = 1e-10
gain_theta_realized[np.isinf(gain_theta_realized)] = 1e-10
gain_phi_realized[np.isnan(gain_phi_realized)] = 1e-10
gain_phi_realized[np.isinf(gain_phi_realized)] = 1e-10

# convert to effective height
c0 = 0.3 
n = 1.0
ZL = 50.0 
Z0 = 120.0 * np.pi

abs_H_rl_theta = c0 / (gain_freqs * np.sqrt(4.0 * n * np.pi)) * np.sqrt(ZL / Z0) * np.sqrt(np.abs(gain_theta_realized))
abs_H_rl_phi = c0 / (gain_freqs * np.sqrt(4.0 * n * np.pi)) * np.sqrt(ZL / Z0) * np.sqrt(np.abs(gain_phi_realized))

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

ephi = butter_bandpass_filter(ephi, 0.05, 1.0, 1.0 / (efield_t[1] - efield_t[0]), order=2)
etheta = butter_bandpass_filter(etheta, 0.05, 1.0, 1.0 / (efield_t[1] - efield_t[0]), order=2)

##################
#      Plots     #
##################

plt.figure()
plt.title("VSWR of Simulated LPDA, In Air")
plt.plot(vswr_freqs * 1e3, vswr)
plt.xlabel("Freqs. [MHz]")
plt.ylabel("VSWR")
plt.xlim(0.0, 1e3)
plt.ylim(0.0, 10.0)
plt.grid()
plt.savefig("./plots/results_vswr.png", dpi = 300)

plt.figure()
plt.title("Realized Forward Gain of Simulated LPDA, Theta Direction, In Air")
plt.plot(gain_freqs * 1e3, 10.0 * np.log10(gain_theta_realized))
plt.xlabel("Freqs. [MHz]")
plt.ylabel("Realized Gain [dBi]")
plt.xlim(0.0, 1e3)
plt.ylim(0.0, 10.0)
plt.grid()
plt.savefig("./plots/results_realized_gain_theta.png", dpi = 300)

plt.figure()
plt.title("Realized Forward Gain of Simulated LPDA, Phi Direction, In Air")
plt.plot(gain_freqs * 1e3, 10.0 * np.log10(gain_phi_realized))
plt.xlabel("Freqs. [MHz]")
plt.ylabel("Realized Gain [dBi]")
plt.xlim(0.0, 1e3)
plt.ylim(-40.0, -15.0)
plt.grid()
plt.savefig("./plots/results_realized_gain_phi.png", dpi = 300)

plt.figure()
plt.title("Effective Height at Boresight, In Air")
plt.plot(gain_freqs * 1e3, abs_H_rl_theta, label = "Theta")
plt.plot(gain_freqs * 1e3, abs_H_rl_phi, label = "Phi")
plt.xlabel("Freqs. [MHz]")
plt.ylabel("Effective Height [m]")
plt.xlim(0.0, 1e3)
plt.ylim(0.0, 0.5)
plt.minorticks_on()
plt.grid(which = 'major', alpha = 0.5)
plt.grid(which = 'minor', alpha = 0.25)
plt.savefig("./plots/results_effective_height.png", dpi = 300)

plt.figure()
plt.title("Far Field Electric Field at Boresight, \n Bandpassed from 50 - 1000 MHz, In Air")
plt.plot(efield_t, etheta, label = "Ttheta")
plt.plot(efield_t, ephi, label = "Phi")
plt.xlabel("Time [ns]")
plt.ylabel("E-Field [V / m]")
plt.xlim(0.0, 40.0)
plt.ylim(-0.15, 0.15)
plt.grid()
plt.legend()
plt.savefig("./plots/results_efield.png", dpi = 300)

plt.show()
