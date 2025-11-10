

import pandas as pd
import ast
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
##Funcion para cargos los datos
def cargar_metadatos(meta_path=None,
                     scp_path=None):

    """
  Carga y prepara los metadatos de PTB‑XL.
  - meta_path: ruta relativa a ptbxl_database.csv
  - scp_path: ruta relativa a scp_statements.csv
    Devuelve dos DataFrames: df_meta (metadatos) y df_scp (mapeo de códigos SCP).
    """
    # Usar rutas por defecto basadas en la raíz del proyecto
    if meta_path is None:
        meta_path = PROJECT_ROOT / 'data' / 'raw' / 'ptbxl_database.csv'
    if scp_path is None:
        scp_path = PROJECT_ROOT / 'data' / 'raw' / 'scp_statements.csv'
    
    # Convertir a string para pandas
    meta_path = str(meta_path)
    scp_path = str(scp_path)
    
    df_meta = pd.read_csv(meta_path)
    df_scp = pd.read_csv(scp_path, index_col=0)
    return df_meta, df_scp

def analizar_metadatos(df_meta):
    """
    Muestra información general sobre el DataFrame de metadatos.
    """
    print("Numero de registros:", len(df_meta))
    print("\nColumnas disponibles:\n", df_meta.columns.tolist())
    print("\nResumen de tipos de datos:\n", df_meta.dtypes)
    print("\nValores unicos de 'sex':", df_meta['sex'].unique())
    print("\nRango de edades :", (df_meta['age'].min(), df_meta['age'].max()))
    print("\nPrimeras filas del DataFrame:\n", df_meta.head())

def parcear_scp_codes(scp_codes_str):
    try:
        return ast.literal_eval(scp_codes_str)
    except (ValueError, SyntaxError):
        return {}

def preparar_etiquetas(df_meta,df_scp,nivel='diagnostic_class', clases_objetivo = None):
    # Convierte scp_codes a diccionario
    df_meta['scp_dict'] = df_meta['scp_codes'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else {}
    )
    # Extrae una lista de códigos SCP por registro
    df_meta['scp_code_list'] = df_meta['scp_dict'].apply(lambda d: list(d.keys()))

    # Mapeo según el nivel elegido
    if nivel == 'diagnostic_class':
        mapper = df_scp['diagnostic_class']
    elif nivel == 'diagnostic_subclass':
        mapper = df_scp['diagnostic_subclass']
    elif nivel == 'category':
        mapper = df_scp['category']
    else:
        raise ValueError("nivel no válido. Usa 'diagnostic_class', 'diagnostic_subclass' o 'category'")

    # Asigna etiquetas mapeadas y elimina Nones
    df_meta['labels'] = df_meta['scp_code_list'].apply(
        lambda codes: list({mapper.get(code) for code in codes if mapper.get(code) is not None})
    )

    # Si se definen clases objetivo, filtra las etiquetas por ese conjunto
    if clases_objetivo is not None:
        df_meta['labels'] = df_meta['labels'].apply(
            lambda labels: [lab for lab in labels if lab in clases_objetivo]
        )

    return df_meta
def resumen_clases(df_meta):
    from collections import Counter
    all_labels = [label for labels in df_meta['labels'] for label in labels]
    counter = Counter(all_labels)
    print("\nDistribución de etiquetas (top 10):\n", counter)

    for label, count in counter.most_common(10):
        print(f"{label}: {count}")
def filtrar_por_fold(df_meta, folds_entrenamiento=range(1,9), fold_val=9, fold_test=10):
    """
    Separa df_meta en conjuntos de entrenamiento, validación y prueba
    respetando las recomendaciones del dataset:contentReference[oaicite:1]{index=1}.
    """
    df_train = df_meta[df_meta['strat_fold'].isin(folds_entrenamiento)]
    df_val = df_meta[df_meta['strat_fold'] == fold_val]
    df_test = df_meta[df_meta['strat_fold'] == fold_test]
    print(f"\nTamaños de los conjuntos:\n Entrenamiento: {len(df_train)}\n Validación: {len(df_val)}\n Prueba: {len(df_test)}")
    return df_train, df_val, df_test