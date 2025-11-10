
def entrenar_modelo(modelo, datos):

    print("Entrenado el modelo con los Datos...")

from dataloaders import ECGDataGenerator
from model_cnn import crear_modelo  # supongamos que defines la arquitectura aquí

train_gen = ECGDataGenerator(data_dir='data/processed/train', batch_size=32)
val_gen = ECGDataGenerator(data_dir='data/processed/val', batch_size=32, shuffle=False)

model = crear_modelo(input_shape=(5000, 12), n_classes=train_gen.n_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_gen,
          validation_data=val_gen,
          epochs=20,
          callbacks=[...])
