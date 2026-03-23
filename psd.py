from dataloader import generate_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

def calculate_psd(X, start_time, tail_time):

    #Полная энергия и хвост
    whole_energy = simpson(X[:, start_time:], axis=1)
    tail_energy = simpson(X[:, tail_time:], axis=1)

    return whole_energy, tail_energy / whole_energy


def plot_psd_scatter(X, start_time, tail_time, y):

    gamma_mask = (y == 1)
    neutron_mask  = (y == 0)
    whole_energy, psd = calculate_psd(X, start_time, tail_time)
    factor = np.max(whole_energy[neutron_mask]) / np.max(whole_energy[gamma_mask])

    #gamma & neutron plot 
    plt.figure(figsize=(12, 8))
    plt.scatter(whole_energy[gamma_mask] * factor, psd[gamma_mask], alpha=0.1, c='green', label='Gamma')
    plt.scatter(whole_energy[neutron_mask], psd[neutron_mask], alpha=0.1, c='blue', label='Neutron')
    plt.ylim(0, 1)
    plt.title("Ограничение метода интегрального разделения (PSD)", fontsize=14)
    plt.xlabel("Энергия (отн. ед.)", fontsize=12)
    plt.ylabel("Параметр PSD ($Q_{tail}/Q_{total}$)", fontsize=12)
    plt.show()

if __name__ == "__main__":
    X, y, amplitudes = generate_data(number_of_samples = 20000, amp_range=(2, 15), length=200, noise_std = 1)
    plot_psd_scatter(X=X, start_time=20, tail_time=30, y=y)
