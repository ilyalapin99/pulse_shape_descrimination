from src.psd.psd import calculate_psd
from src.data.dataloader import generate_data
import numpy as np
import matplotlib.pyplot as plt

def evaluate_psd(X, y, start_time=20, tail_time=30):
    whole_energy, psd_value = calculate_psd(X=X, start_time=start_time, tail_time=tail_time)

    thresholds = np.linspace(np.min(psd_value), np.max(psd_value), 1000)
    best_treshold = 0
    min_errors = len(y)


    for t in thresholds:

        preds = (psd_value < t).astype(int)

        #Количество ошибок
        errors = np.sum(preds != y)

        if errors < min_errors:
            min_errors = errors
            best_treshold = t
    
    print(f"Количество ошибок PSD {min_errors}")
    gamma_mask = (y == 1)
    neutron_mask  = (y == 0)
    factor = np.max(whole_energy[neutron_mask]) / np.max(whole_energy[gamma_mask])
    preds = (psd_value < best_treshold).astype(int)
    errors = np.sum(preds != y)
    print(f"Fuck{errors}")
    #gamma & neutron plot 
    plt.figure(figsize=(12, 8))
    plt.scatter(whole_energy[gamma_mask] * factor, psd_value[gamma_mask], alpha=0.3, c='green', label='Gamma')
    plt.scatter(whole_energy[neutron_mask], psd_value[neutron_mask], alpha=0.3, c='blue', label='Neutron')
    plt.axhline(y=best_treshold, color='red', linestyle='--', label='Порог PSD')
    plt.ylim(0, 1)
    plt.title("Интегральное разделение (PSD)", fontsize=22)
    plt.xlabel("Энергия (отн. ед.)", fontsize=18)
    plt.ylabel("Параметр PSD", fontsize=18)
    plt.xlim(0.0, 350)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    plt.show()

    
    return min_errors, best_treshold





if __name__ == '__main__':
    X, y, amplitudes = generate_data(number_of_samples = 4000, amp_range=(2, 15), length=200, noise_std = 1)
    min_errors, best_treshold = evaluate_psd(X, y)
    print(min_errors)
