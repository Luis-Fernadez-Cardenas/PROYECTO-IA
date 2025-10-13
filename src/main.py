# This is a sample Python script.

# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from lecturaDatos import cargar_datos
from preprocessing import filtrar_señal
from model_cnn import crear_modelo
from train import entrenar_modelo

if __name__ == "__main__":
    datos = cargar_datos("data/raw/00001_hr")
    senal_filtrada = filtrar_señal(datos)
    modelo = crear_modelo()
    entrenar_modelo(modelo, senal_filtrada)
