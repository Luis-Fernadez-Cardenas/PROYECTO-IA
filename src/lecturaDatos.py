import numpy as np
import wfdb
import matplotlib.pyplot as plt
import pandas as pd

# # Ruta al archivo (sin extensión)
# ruta = r"D:\PROYECTO-IA\00001_hr"
#
# # Leer la señal
# record = wfdb.rdrecord(ruta)
#
# # Mostrar información
# wfdb.plot_wfdb(record=record, title='ECG - PTB-XL')
# plt.show()

# Ruta SIN extensión (.dat y .hea deben estar en la misma carpeta)
ruta = r"D:\PROYECTO-IA\data\raw\00001_hr"   # cambia según tu archivo

# Leer el registro (wfdb detecta .hea y .dat automáticamente)
record = wfdb.rdrecord(ruta)

# ====== DATOS EN CRUDO ======
# Señal ECG en forma de matriz (muestras x derivaciones)
senial = record.p_signal

# Nombres de derivaciones
derivaciones = record.sig_name

# Frecuencia de muestreo (cuántas muestras por segundo)
fs = record.fs

# Mostrar información básica
df = pd.DataFrame(record.p_signal, columns=record.sig_name)


data = record.p_signal

sig_names = record.sig_name

# Mostrar primeras filas
# Crear vector de tiempo
t = [i / fs for i in range(len(data))]

# Graficar varias derivaciones
lead = 1  # por ejemplo la derivación II
plt.figure(figsize=(12, 4))
plt.plot(t, data[:, lead], color='red', linewidth=1)
plt.title(f'Derivación {sig_names[lead]}')
plt.xlabel('Tiempo (s)')
plt.ylabel('mV')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
#print(df.head(100))

# print("Frecuencia de muestreo:", fs, "Hz")
# print("Derivaciones:", derivaciones)
# print("Forma de los datos:", senial.shape)
# print("\nPrimeras 10 muestras (valores crudos):")
# print(senial[:10])