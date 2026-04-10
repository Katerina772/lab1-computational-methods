# Модель: Моделювання росту бактеріальної популяції (експоненціальна, логістична, Ґомперца)
# Автор: Паламарчук Катерина, група АІ-233 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 2. Вхідні параметри моделей

t = np.linspace(0, 24, 25)  # години (0–24)
params = {
    "exponential": {"N0": 1e3, "r": 0.45},
    "logistic": {"K": 1.2e6, "r": 0.6, "N0": 1e3},
    "gompertz": {"A": 1.2e6, "mu": 1.0, "lam": 2.0}
}

# 3. Опис моделей

def exp_model(t, N0, r):
    return N0 * np.exp(r * t)

def logistic_model(t, K, r, N0):
    A = (K - N0) / N0
    return K / (1 + A * np.exp(-r * t))

def gompertz_model(t, A, mu, lam):
    return A * np.exp(-np.exp(mu * (lam - t)))

# 4. Генерація "експериментальних" даних

N_exp = exp_model(t, **params["exponential"])
N_log = logistic_model(t, **params["logistic"])
N_gom = gompertz_model(t, **params["gompertz"])

# додаємо шум, щоб імітувати реальні дані
rng = np.random.default_rng(42)
noise_level = 0.05
N_exp_noisy = N_exp * (1 + noise_level * rng.normal(size=t.size))
N_log_noisy = N_log * (1 + noise_level * rng.normal(size=t.size))
N_gom_noisy = N_gom * (1 + noise_level * rng.normal(size=t.size))


# 5. Оцінка параметрів (апроксимація)
logN = np.log(np.maximum(N_exp_noisy, 1e-8))
A = np.vstack([np.ones_like(t), t]).T
coeffs, residuals, rank, s = np.linalg.lstsq(A, logN, rcond=None)
lnN0_fit, r_fit = coeffs[0], coeffs[1]
N0_fit = np.exp(lnN0_fit)

fit_results = {"exponential": {"N0_fit": N0_fit, "r_fit": r_fit}}

# пробуємо зробити апроксимацію для інших моделей (якщо є scipy)
try:
    from scipy.optimize import curve_fit

    # логістична модель
    p0_log = [params["logistic"]["K"], params["logistic"]["r"], params["logistic"]["N0"]]
    popt_log, pcov_log = curve_fit(logistic_model, t, N_log_noisy, p0=p0_log, maxfev=20000)
    fit_results["logistic"] = {"K": float(popt_log[0]), "r": float(popt_log[1]), "N0": float(popt_log[2])}

    # модель Гомперца
    p0_gom = [params["gompertz"]["A"], params["gompertz"]["mu"], params["gompertz"]["lam"]]
    popt_gom, pcov_gom = curve_fit(gompertz_model, t, N_gom_noisy, p0=p0_gom, maxfev=20000)
    fit_results["gompertz"] = {"A": float(popt_gom[0]), "mu": float(popt_gom[1]), "lam": float(popt_gom[2])}
except Exception as e:
    fit_results["logistic"] = "Fit failed: " + str(e)
    fit_results["gompertz"] = "Fit failed: " + str(e)


# 6. Формування таблиці результатів
df = pd.DataFrame({
    "t_hours": t,
    "exp_clean": N_exp,
    "exp_noisy": N_exp_noisy,
    "log_clean": N_log,
    "log_noisy": N_log_noisy,
    "gom_clean": N_gom,
    "gom_noisy": N_gom_noisy
})
print("\n=== Перші рядки даних ===")
print(df.head())

# 7. Побудова графіків

plt.figure(figsize=(8, 5))
plt.plot(t, N_exp_noisy, 'o', label='Експоненціальна (дані)')
plt.plot(t, exp_model(t, N0_fit, r_fit), '--', label='Експоненціальна (апроксимація)')
plt.xlabel("Час (години)")
plt.ylabel("Кількість бактерій")
plt.title("Експоненціальна модель росту")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(t, N_log_noisy, 'o', label='Логістична (дані)')
if isinstance(fit_results["logistic"], dict):
    plt.plot(t, logistic_model(t, **fit_results["logistic"]), '--', label='Логістична (апроксимація)')
plt.xlabel("Час (години)")
plt.ylabel("Кількість бактерій")
plt.title("Логістична модель росту")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(t, N_gom_noisy, 'o', label='Ґомперц (дані)')
if isinstance(fit_results["gompertz"], dict):
    plt.plot(t, gompertz_model(t, **fit_results["gompertz"]), '--', label='Ґомперц (апроксимація)')
plt.xlabel("Час (години)")
plt.ylabel("Кількість бактерій")
plt.title("Модель Ґомперца")
plt.legend()
plt.grid(True)
plt.show()

# 8. Вивід результатів

print("\n=== Результати апроксимації ===")
for model, result in fit_results.items():
    print(f"\nМодель: {model}")
    print(result)

