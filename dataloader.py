import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

PARAMS_GAMMA = {'rise_time': 1.5, 'fast_time': 4.0, 'time_slow': 30.0, 'B': 0.1}
PARAMS_NEUTRON  = {'rise_time': 1.5, 'fast_time': 4.0, 'time_slow': 35.0, 'B': 0.3}

def scintillation_model(rise_time, time_slow, fast_time, amp, abs_time, pulse_start, B):
    return amp * (np.exp( -(abs_time - pulse_start) / fast_time) - np.exp( -(abs_time - pulse_start) / rise_time) + B * np.exp( -(abs_time - pulse_start) / time_slow))

def generate_impulse(rise_time, time_slow, fast_time, amp, pulse_start, B, length=200):

    #Время
    t = np.arange(length)

    #Импульс
    pulse = np.zeros(length)
    mask = (t >= pulse_start)
    update_time = t[mask]
    pulse[mask] = scintillation_model(rise_time, time_slow, fast_time, amp, update_time, pulse_start, B)
    
    return t, pulse / np.max(pulse) * amp
def generate_data(number_of_samples = 20000, amp_range=(7, 25), length=200, noise_std = 1):
    
    #Данные
    X = np.zeros((number_of_samples, length))
    y = np.zeros(number_of_samples)

    #Амплитуды
    low, high = amp_range
    gamma_samples = np.random.power(a=0.83, size=number_of_samples//2)
    gamma_amplitudes = low + (high - low) * gamma_samples
    neutron_samples = np.random.power(a=0.83, size=number_of_samples//2)
    neutron_amplitudes = low + (high - low) * neutron_samples
    for i in range(number_of_samples // 2):
        
        #Gamma 1
        _, gamma_pulse = generate_impulse(
            rise_time=PARAMS_GAMMA['rise_time'],
            time_slow=PARAMS_GAMMA['time_slow'],
            fast_time=PARAMS_GAMMA['fast_time'],
            amp=gamma_amplitudes[i],
            pulse_start=20,
            B=PARAMS_GAMMA['B'],
        )
        X[i] = gamma_pulse + np.random.normal(loc=0, scale=noise_std, size=length)
        y[i] = 1

        #Neutron 0
        _, neutron_pulse = generate_impulse(
            rise_time=PARAMS_NEUTRON['rise_time'],
            time_slow=PARAMS_NEUTRON['time_slow'],
            fast_time=PARAMS_NEUTRON['fast_time'],
            amp=neutron_amplitudes[i],
            pulse_start=20,
            B=PARAMS_NEUTRON['B'],
        )
        X[i + number_of_samples // 2] = neutron_pulse + np.random.normal(loc=0, scale=noise_std, size=length)
        y[i + number_of_samples // 2] = 0
    
    return X, y, np.hstack([gamma_amplitudes, neutron_amplitudes])
    
if __name__ == '__main__':
    X, y, amplitudes = generate_data(number_of_samples = 20000, amp_range=(2, 15), length=200, noise_std=0)
    mask_gamma = (y == 1)
    mask_neutron = (y == 0)
    for i, x in enumerate(X[mask_gamma]):
        plt.plot(x)
        if i == 2:
            break
    for i, x in enumerate(X[mask_neutron]):
        plt.plot(x)
        if i == 2:
            break
    
    plt.show()
    plt.hist(amplitudes, bins=30)
    plt.show()
