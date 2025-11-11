import numpy as np
import pandas as pd
import os
import tensorflow as tf

class ECGDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, batch_size=32, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        """
        data_dir: ruta a la carpeta del split (e.g. data/processed/train)
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Lee el CSV con file y labels
        self.df_labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
        self.indexes = np.arange(len(self.df_labels))

        # Construye un diccionario para codificar etiquetas a índices
        clases = sorted({lab for labs in self.df_labels['labels']
                                 for lab in eval(labs)})
        self.class2idx = {c: i for i, c in enumerate(clases)}
        self.n_classes = len(self.class2idx)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df_labels) / self.batch_size))

    def __getitem__(self, index):
        batch_idxs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_files = self.df_labels.iloc[batch_idxs]['file'].values
        batch_labels = self.df_labels.iloc[batch_idxs]['labels'].values

        X, y = [], []
        for f, labs_str in zip(batch_files, batch_labels):
            signal = np.load(os.path.join(self.data_dir, f))
            X.append(signal)  # shape (5000, 12)
            # Etiquetas: conviértelas a vector one‑hot (multiclase o multilabel)
            labs = eval(labs_str)
            vec = np.zeros(self.n_classes, dtype=np.float32)
            for lab in labs:
                vec[self.class2idx[lab]] = 1.0
            y.append(vec)

        return np.array(X), np.array(y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
