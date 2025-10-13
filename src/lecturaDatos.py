import numpy as np
import wfdb
import matplotlib.pyplot as plt
import pandas as pd


# Ruta SIN extensión (.dat y .hea deben estar en la misma carpeta)
ruta = r"D:\PROYECTO-IA\data\raw\00001_hr"


# Graficar varias derivaciones
lead = 1  # por ejemplo la derivación II
plt.figure(figsize=(12, 4))

#print(df.head(100))

# print("Frecuencia de muestreo:", fs, "Hz")
# print("Derivaciones:", derivaciones)
# print("Forma de los datos:", senial.shape)
# print("\nPrimeras 10 muestras (valores crudos):")
# print(senial[:10])

def cargar_datos(ruta):
    record = wfdb.rdrecord(ruta)
    return record

def visualizar_señal(record, derivacion_idx=0):

    senial = record.p_signal
    derivaciones = record.sig_name
    fs = record.fs
    # Crear vector de tiempo
    t = [i / fs for i in range(len(senial))]
    plt.figure(figsize=(15, 5))
    plt.plot(t, senial[:, derivacion_idx], color='red', linewidth=1)
    plt.title(f'Derivación: {derivaciones[derivacion_idx]}')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud (mV)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
