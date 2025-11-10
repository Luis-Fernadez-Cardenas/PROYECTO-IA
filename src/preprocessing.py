import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs = 500, order=4):
    """
    Aplica un filtro Butterworth paso banda a cada derivación de la señal.
    - signal: array 2D (timesteps x derivaciones)
    - lowcut, highcut: frecuencias de corte en Hz
    - fs: frecuencia de muestreo
    """
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    # filtra a lo largo del eje temporal (fila) para cada columna

    return filtfilt(b, a, signal, axis=0)

def normalizar_signal(signal):
    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0)
    std[std == 0] = 1.0 # Evitar división por cero
    return (signal - mean) / std

    return
def preprocesar_registro(ruta):
    """
    Carga un registro con wfdb, aplica filtro paso banda y normalización.
    Devuelve un array 2D (timesteps x 12 derivaciones).
    """
    record = wfdb.rdrecord(ruta)
    raw_signal = record.p_signal.astype(np.float32)
    filtered = bandpass_filter(raw_signal, fs=record.fs)
    normalized = normalizar_signal(filtered)
    return normalized, record.fs,record.sig_name
## Segmentacion de la señal
def segmentar_señal(singal, segment_length=5000, step = 2500 ):
    """
    Divide la señal en segmentos solapados.
    - segment_length: número de muestras por segmento (p. ej. 5 s * 500 Hz = 2500)
    - step: número de muestras que avanzas entre segmentos
    Devuelve una lista de segmentos.
    """
    segments = []
    for start in range(0, singal.shape[0] - segment_length +1, step):
       end = start + segment_length
       segments.append(singal[start:end,:])

    return np.array(segments)