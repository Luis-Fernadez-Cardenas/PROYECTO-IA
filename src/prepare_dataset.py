import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from preprocessing import preprocesar_registro

def crear_dataset_procesado(df_meta,split_name,base_raw = 'data/raw',out_dir ='data/processed'):
    """
    Preprocesa las señales listadas en df_meta y guarda cada una como .npy.
    También guarda un archivo CSV con las etiquetas para cada registro.
    - df_meta: DataFrame con columnas 'ecg_id', 'filename_hr' y 'labels'.
    - split_name: nombre del subconjunto ('train', 'val' o 'test') para prefijos.
    - base_raw: carpeta donde están los archivos .dat/.hea.
    - out_dir: carpeta donde se guardarán las señales procesadas.
    """
    # Carpeta de salida para este split
    split_dir = os.path.join(out_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    registros_info = []

    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc=f'Procesando {split_name}'):
        ruta = os.path.join(base_raw, row['filename_hr'])
        signal_norm, _, _ = preprocesar_registro(ruta)
        # Nombre del archivo basado en ecg_id
        filename = f"{int(row['ecg_id'])}.npy"
        np.save(os.path.join(split_dir, filename), signal_norm.astype(np.float32))
        # Guarda las etiquetas (lista) como cadena para CSV
        registros_info.append({'file': filename, 'labels': row['labels']})

    # Guarda un CSV con el mapeo archivo -> etiquetas
    df_info = pd.DataFrame(registros_info)
    df_info.to_csv(os.path.join(split_dir, 'labels.csv'), index=False)
    print(f"Guardado {len(df_meta)} registros procesados en {split_dir}")