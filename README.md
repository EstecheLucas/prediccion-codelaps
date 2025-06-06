# Predicción de Consumo de Combustible con TensorFlow.js

Este proyecto es una aplicación web simple que utiliza TensorFlow.js para predecir el consumo de combustible (MPG) de un auto a partir de la potencia del motor (horsepower). El modelo es una red neuronal que se entrena con datos reales y luego permite hacer predicciones con valores nuevos.

---

## Requisitos

- Navegador moderno con soporte para JavaScript (Chrome, Firefox, Edge, Safari).
- Conexión a internet para cargar TensorFlow.js desde CDN.

---

## Archivos

- `index.html`: Interfaz básica con campos para entrenar el modelo y predecir consumo.
- `script.js`: Código JavaScript con el modelo, entrenamiento y predicción..

---

## Cómo ejecutar el proyecto

1. Descargar o clonar este repositorio en tu computadora.

2. Abrir el archivo `index.html` en tu navegador preferido.

3. En la página web, hacer clic en el botón **"Entrenar Modelo"** para iniciar el entrenamiento del modelo. Esto puede tardar unos segundos.

4. Una vez finalizado el entrenamiento, ingresar un valor numérico para `horsepower` (potencia) en el campo correspondiente.

5. Hacer clic en el botón **"Predecir"** para obtener el consumo estimado en millas por galón (MPG).

---

## Notas

- El modelo utiliza datos simplificados para aprender la relación entre potencia y consumo.
- La normalización de datos mejora la precisión del modelo.
- El entrenamiento muestra la evolución de la pérdida (error) en pantalla.
- La predicción se realiza en tiempo real sin necesidad de backend.

---

## Recursos

- [TensorFlow.js](https://www.tensorflow.org/js)
- Dataset Auto MPG simplificado (en el código).

---




