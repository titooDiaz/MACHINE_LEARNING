<p align="center">
  <img width="150px" src="https://i.ibb.co/bXvzjXm/LOGO-h1.png" />
</p>

# TENSORFLOW
A medida que avances en el Curso intensivo de aprendizaje automático, podrás poner en práctica los conceptos del aprendizaje automático mediante la codificación de modelos en tf.keras. Usarás Colab como entorno de programación. Colab es la versión de Google de Notebook de Jupyter. Al igual que el notebook de Jupyter, Colab proporciona un entorno interactivo de programación de Python que combina texto, código, gráficos y resultados del programa.

## NumPy y Pandas
El uso de tf.keras requiere, al menos, un poco de comprensión de las siguientes dos bibliotecas de código abierto de Python:

NumPy, que simplifica la representación de arrays y la realización de operaciones de álgebra lineal. [Numpy](https://numpy.org/)
<br>

pandas, que proporciona una manera fácil de representar conjuntos de datos en la memoria. [Pandas](https://pandas.pydata.org/)

Si no estás familiarizado con NumPy o Pandas, comienza con los siguientes dos ejercicios de Colab:

Ejercicio de Colab de NumPy UltraQuick Tutorial, que brinda toda la información de NumPy que necesitas para este curso. (NUMPY TUTORIAL)
Ejercicio de Colab sobre el instructivo rápido de Pandas, que brinda toda la información que necesitas sobre este curso. (PANDAS TUTORIAL)

## Generalizacion
Guillermo de Ockham, un fraile y filósofo del siglo XIV, amaba la simplicidad. Creía que los científicos deberían preferir lo más simple fórmulas o teorías sobre otras más complejas. Para poner la navaja de Ockham en la máquina Términos de aprendizaje:
`Cuanto menos complejo sea un modelo de ML, más probable es que se obtenga un buen resultado empírico no se debe solo a las peculiaridades de la muestra.`

<br>
Un modelo de aprendizaje automático tiene como objetivo hacer buenas predicciones sobre datos nunca antes vistos. Pero si está construyendo un modelo a partir de su conjunto de datos, ¿cómo obtendría los datos nunca antes vistos? Pozo Una forma es dividir el conjunto de datos en dos subconjuntos:
<br>
Conjunto de entrenamiento: un subconjunto para entrenar un modelo.
Conjunto de pruebas: un subconjunto para probar el modelo.
<br>
Un buen rendimiento en el equipo de prueba es un indicador útil de un buen rendimiento sobre los nuevos datos en general, asumiendo que:
<br>
El equipo de prueba es lo suficientemente grande.
No se hace trampa utilizando el mismo conjunto de pruebas una y otra vez.

El set de prueba es el que menos datos debe contener, ya que el entrenamiento es la prioridad

Conjunto de entrenamiento: Un subconjunto para entrenar un modelo.
Conjunto de prueba: Un subconjunto para probar el modelo entrenado.
Puedes imaginarte dividir el único conjunto de datos de la siguiente manera:
<img src="https://developers.google.com/static/machine-learning/crash-course/images/PartitionTwoSets.svg?hl=es-419">

Asegúrate de que tu conjunto de prueba cumpla con las siguientes dos condiciones:

Sea lo suficientemente grande como para generar resultados significativos desde el punto de vista estadístico.
Que sea representativo del conjunto de datos en su conjunto. En otras palabras, no elijas un conjunto de prueba con características diferentes a las del conjunto de entrenamiento.

Nunca uses datos de prueba para el entrenamiento. Si ves resultados sorprendentemente buenos en tus métricas de evaluación, puede ser una señal de que estás entrenando accidentalmente en el conjunto de prueba. Por ejemplo, una precisión alta puede indicar que los datos de prueba se filtraron en el conjunto de entrenamiento.
<br>
Para saber que tantos datos necesita nuestro modelo  para trabajar correctamente, podemos crear conjuntos de entrenamiento y prueba
<br>
Cuando el conjunto de entrenamiento es mas grande, la capacidad del modelo de predecir crece significativamente...
A mayor conjunto de prueba, mayor sera la confianza de los resultados
<br>
Es bueno tener un conjunto de datos MUY GRANDE
aproximadamente el 15%

### Ejercicio de Google
[Practica!](https://developers.google.com/machine-learning/crash-course/training-and-test-sets/playground-exercise?hl=es-419)
Con este ejercicio podremos comprender un poco mas sobre los datos de entrenamiento...

# Flujo de trabajo
<img src="https://developers.google.com/static/machine-learning/crash-course/images/WorkflowWithTestSet.svg?hl=es-419">

### Figura 1: ¿Un flujo de trabajo posible?

En la figura, "Ajustar el modelo" significa ajustar cualquier elemento del modelo que puedas imaginar, desde cambiar la tasa de aprendizaje hasta agregar o quitar atributos, o diseñar un modelo completamente nuevo desde cero. Al final de este flujo de trabajo, debes elegir el modelo que mejor se desempeñe con respecto al conjunto de prueba.

Dividir el conjunto de datos en dos conjuntos es una buena idea, pero no una panacea. Para reducir en gran medida las posibilidades de sobreajuste, puedes particionar el conjunto de datos en los tres subconjuntos que se muestran en la siguiente figura:

<img src="https://developers.google.com/static/machine-learning/crash-course/images/PartitionThreeSets.svg?hl=es-419">

### Figura 2: División de un único conjunto de datos en tres subconjuntos

Usa el conjunto de validación para evaluar los resultados del conjunto de entrenamiento. A continuación, usa el conjunto de prueba para verificar la evaluación después de que el modelo haya "pasado" el conjunto de validación. En la siguiente figura, se muestra este nuevo flujo de trabajo:
<img src="https://developers.google.com/static/machine-learning/crash-course/images/WorkflowWithValidationSet.svg?hl=es-419">

Figura 3: Un flujo de trabajo mejorado.

### En este flujo de trabajo mejorado, sucede lo siguiente:

Elige el modelo que mejor se desempeñe con el conjunto de validación.
Vuelve a verificar el modelo con el conjunto de prueba.
Este flujo de trabajo es más eficaz porque crea menos exposiciones al conjunto de prueba.

#### Ejercicio: (ir al archivo *VALIDACION Y SETS DE DATOS* y completarlo)