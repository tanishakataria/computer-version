
import numpy as np
import matplotlib.pyplot as plt
import pywt

np.random.seed(42)
n = 1000
data = np.random.normal(0, 1, n)

idx = np.random.choice(n, 25, replace=False)
data[idx] += np.random.choice([-2, 2], 25)

coeffs = pywt.wavedec(data, 'db4', level=4)

sigma = np.median(np.abs(coeffs[-1])) / 0.6745
thr = sigma * np.sqrt(2 * np.log(n))

new_coeffs = [coeffs[0]]
for c in coeffs[1:]:
    new_coeffs.append(pywt.threshold(c, thr, mode='soft'))

denoised = pywt.waverec(new_coeffs, 'db4')[:n]
residual = data - denoised

limit = 2 * np.std(residual)
anom = np.where(np.abs(residual) > limit)[0]
print("Anomalies detected:", len(anom))

plt.figure(figsize=(10, 7))
plt.subplot(2, 1, 1)
plt.plot(data, label="Sensor Data")
plt.plot(denoised, '--', label="Denoised Signal")
plt.title("Sensor Data and Denoised Signal")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(residual, 'r', label="Residuals")
plt.scatter(anom, residual[anom], c='g', label="Anomalies", zorder=3)
plt.title("Residuals and Detected Anomalies")
plt.legend()
plt.tight_layout()
plt.show()
