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


# Ingeniería de atributos
Cómo asignar valores numéricos
Los datos de número entero y de punto flotante no necesitan una codificación especial porque se pueden multiplicar por un peso numérico. Como se sugiere en la Figura 2, convertir el valor entero sin procesar 6 en el valor de atributo 6.0 es sencillo:
<img src="https://developers.google.com/static/machine-learning/crash-course/images/FloatingPointFeatures.svg?hl=es-419">

<br>
Asignación de valores categóricos
Los atributos categóricos tienen un conjunto discreto de valores posibles. Por ejemplo, podría haber una función llamada street_name con opciones que incluyan las siguientes:


{'Charleston Road', 'North Shoreline Boulevard', 'Shorebird Way', 'Rengstorff Avenue'}
Dado que los modelos no pueden multiplicar strings por los pesos aprendidos, usamos la ingeniería de atributos para convertir strings en valores numéricos.

Para ello, definimos una asignación de los valores de los atributos, a los que nos referiremos como vocabulario de valores posibles, a números enteros. Dado que no todas las calles del mundo aparecerán en nuestro conjunto de datos, podemos agrupar todas las demás calles en una categoría general llamada "otras", que se conoce como un bucket OOV (fuera del vocabulario).

Mediante este enfoque, podemos asignar nombres de calles a números de la siguiente manera:

asignar Charleston Road a 0
asignar North Shoreline Boulevard a 1
asignar Shorebird Way a 2
asignar Rengstorff Avenue a 3
asignar todo lo demás (OOV) a 4
Sin embargo, si incorporamos estos números índice directamente en nuestro modelo, impondrá algunas restricciones que podrían ser problemáticas:

Aprenderemos un peso único que se aplique a todas las calles. Por ejemplo, si aprendemos un peso de 6 para street_name, lo multiplicaremos por 0 para Charleston Road, por 1 para North Shoreline Boulevard, por 2 para Shorebird Way y así sucesivamente. Considera un modelo que prediga el precio de las casas usando street_name como atributo. Es poco probable que haya un ajuste lineal del precio basado en el nombre de la calle. Además, esto supondrá que ordenaste las calles según el precio promedio de las casas. Nuestro modelo necesita la flexibilidad de aprender los diferentes pesos para cada calle, que se agregarán al precio estimado con los otros atributos.

No estamos contemplando los casos en los que street_name puede tener varios valores. Por ejemplo, muchas casas se encuentran en la esquina de dos calles y no hay forma de codificar esa información en el valor street_name si este contiene un solo índice.

Para quitar ambas restricciones, podemos crear un vector binario para cada atributo categórico de nuestro modelo que represente valores de la siguiente manera:

En el caso de los valores que se aplican al ejemplo, establece los elementos correspondientes al vector en 1.
Establecer todos los demás elementos en 0.
La longitud de este vector es igual a la cantidad de elementos en el vocabulario. Esta representación se denomina codificación one-hot cuando un único valor es 1 y codificación multi-hot cuando varios valores son 1.

La figura 3 ilustra una codificación one-hot de una calle determinada: Shorebird Way. El elemento del vector binario de Shorebird Way tiene un valor de 1, mientras que los elementos de todas las demás calles tienen un valor de 0.

<img src="https://developers.google.com/static/machine-learning/crash-course/images/OneHotEncoding.svg?hl=es-419">

Este enfoque crea de manera efectiva una variable booleana para cada valor de atributo (p.ej., nombre de la calle). En este caso, si una casa se encuentra en Shorebird Way, el valor binario es solo 1 para Shorebird Way. Por lo tanto, el modelo utiliza solo el peso para la calle Shorebird Way.

Del mismo modo, si una casa se encuentra en la esquina de dos calles, entonces dos valores binarios se establecen en 1, y el modelo usa ambos pesos respectivos.

# Representación: Limpieza de datos 
Los manzanos producen una mezcla de frutas excelentes y gusanos. Sin embargo, las manzanas que se muestran en los supermercados refinados son frutas 100% perfectas. Entre el huerto y el supermercado, alguien pasa mucho tiempo quitando las manzanas en mal estado o lanzando un poco de cera sobre las que se pueden recuperar. Como ingeniero de AA, dedicarás una gran cantidad de tu tiempo a desechar ejemplos malos y limpiar los que se pueden recuperar. Incluso unas pocas "manzanas en mal estado" pueden arruinar un gran conjunto de datos.

Ajusta valores de atributos
Escalamiento significa convertir los valores de atributos de punto flotante de su rango natural (por ejemplo, 100 a 900) al rango estándar (por ejemplo, 0 a 1 o -1 a +1). Si un conjunto de atributos consiste en una sola función, el escalamiento proporciona poco o ningún beneficio práctico. Sin embargo, si un conjunto de atributos consta de varios atributos, el escalamiento de atributos proporciona los siguientes beneficios:

Ayuda a que el descenso de gradientes converja más rápidamente.
Ayuda a evitar la "trampa de NaN", en la que un número del modelo se convierte en un NaN (p.ej., cuando un valor excede el límite de precisión de punto flotante durante el entrenamiento) y, debido a operaciones matemáticas, el resto de los números del modelo finalmente se convierte en NaN.
Permite que el modelo aprenda las ponderaciones correspondientes para cada atributo. Sin el ajuste de atributos, el modelo les prestará demasiada atención a los atributos que tienen un rango más amplio.
No es necesario que asignes el mismo ajuste de escala a cada atributo de punto flotante. No sucederá nada terrible si el Atributo A se escala de -1 a +1, mientras que el Atributo B se ajusta de -3 a +3. Sin embargo, tu modelo reaccionará mal si el Atributo B se escala de 5,000 a 100,000.

### Manejo de valores atípicos extremos
El siguiente gráfico representa un atributo llamado roomsPerPerson del conjunto de datos Viviendas de California. El valor de roomsPerPerson se dividió dividiendo la cantidad total de habitaciones en un área por la población en esa área. El gráfico muestra que la gran mayoría de áreas en California tiene una o dos habitaciones por persona. Pero veamos el eje X.

<img src="https://developers.google.com/static/machine-learning/crash-course/images/ScalingNoticingOutliers.svg?hl=es-419">

entonces:
<img src="https://developers.google.com/static/machine-learning/crash-course/images/ScalingClipping.svg?hl=es-419">

El recorte del valor del atributo en 4.0 no significa que ignoremos todos los valores superiores a 4.0. En cambio, significa que todos los valores que eran superiores a 4.0 ahora se convierten en 4.0. Esto explica la elevación extraña en 4.0. A pesar de esa elevación, el conjunto de atributos ajustado ahora es más útil que los datos originales.

<br>
En el conjunto de datos, latitude es un valor de punto flotante. Sin embargo, no tiene sentido representar latitude como un atributo de punto flotante en nuestro modelo. Eso se debe a que no existe una relación lineal entre la latitud y los valores de las viviendas. Por ejemplo, las casas en la latitud 35 no son 
 más costosas (o menos costosas) que las casas en la latitud 34. Sin embargo, las latitudes individuales probablemente son un muy buen predictor de los valores de casas.

Para que la latitud sea un predictor útil, debemos dividir las latitudes en discretizaciones, como se sugiere en la siguiente figura:

<img src="https://developers.google.com/static/machine-learning/crash-course/images/ScalingBinningPart2.svg?hl=es-419">

<br>
En lugar de tener un atributo de punto flotante, ahora tenemos 11 atributos booleanos distintos (LatitudeBin1, LatitudeBin2, ..., LatitudeBin11). Tener 11 atributos independientes es algo poco elegante, por lo que hay que unirlos en un solo vector de 11 elementos. Esto nos permitirá representar la latitud 37.4 de la siguiente manera:

`[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]`


Gracias a la discretización, nuestro modelo ahora puede aprender pesos completamente diferentes para cada latitud.


# COMBINACIONES DE ATRIBUTOS
## Codificación de no linealidad

<img src='https://developers.google.com/machine-learning/crash-course/images/LinearProblem1.png?hl=es-419'>
¿Es un problema lineal?

¿Puedes dibujar una línea que separe los árboles enfermos de los sanos? Claro. Este es un problema lineal. La línea no será perfecta. Uno o dos árboles enfermos pueden estar del lado "sano", pero la línea será un buen predictor.

Ahora, observe la siguiente figura:

<img src="https://developers.google.com/machine-learning/crash-course/images/LinearProblem2.png?hl=es-419">
¿Es un problema lineal?

¿Puedes trazar una sola línea recta que separe los árboles enfermos de los sanos? No, no puedes. Este es un problema no lineal. Cualquier línea que dibujes será un predictor deficiente del estado de los árboles.

<img src="https://developers.google.com/machine-learning/crash-course/images/LinearProblemNot.png?hl=es-419">

Para resolver el problema no lineal que se muestra en la Figura 2, crea una combinación de atributos. Una combinación de atributos es un atributo sintético que codifica la no linealidad en el espacio de los atributos al multiplicar dos o más atributos de entrada. (El término combinación proviene de productos cruzados). Creemos una combinación de atributos llamada x3 mediante la combinacion `x1` y `x2`
`x3 = x1*x2`


Tratamos esta combinación de atributos x3 como cualquier otro atributo. La fórmula lineal pasa a ser la siguiente:

`y=b+w1*x1+w2*x2+w3*x3`

Un algoritmo lineal puede aprender un peso para `w3`
del mismo modo que para  `w1` y `w2` 
En otras palabras, aunque `w3` codifica información no lineal, no necesitas cambiar la forma en la que se entrena el modelo lineal para determinar el valor de `w3`

Tipos de combinaciones de atributos
Es posible crear muchos tipos de combinaciones de atributos diferentes. Por ejemplo:

[A X B]: Una combinación de atributos formada al multiplicar los valores de dos atributos.
[A x B x C x D x E]: Una combinación de atributos formada al multiplicar los valores de cinco atributos.
[A x A]: Una combinación de atributos formada al elevar al cuadrado un solo atributo.
Gracias al descenso de gradientes estocástico, los modelos lineales se pueden entrenar de manera eficiente. En consecuencia, la complementación de los modelos lineales ajustados con combinaciones de atributos ha sido tradicionalmente una forma eficiente de entrenar conjuntos de datos de escala masiva.


# Vectores de una sola combinacion
Hasta ahora, nos hemos enfocado en la combinación de dos atributos de punto flotante individuales. En la práctica, los modelos de aprendizaje automático rara vez abarcan atributos continuos. Sin embargo, los modelos de aprendizaje automático suelen cruzar vectores de atributos one-hot. Piensa en combinaciones de atributos de vectores de un solo 1 como conjunciones lógicas. Por ejemplo, supongamos que tenemos dos atributos: país e idioma. Una codificación one-hot de cada una genera vectores con atributos binarios que pueden interpretarse como country=USA, country=France o language=English, language=Spanish. Luego, si realizas una combinación de atributos de estas codificaciones de un solo 1, obtienes atributos binarios que pueden interpretarse como conjunciones lógicas, como las siguientes:


  country:usa AND language:spanish
Como otro ejemplo, supongamos que discretizas latitud y longitud, lo que produce vectores de atributos de un solo 1 con cinco elementos. Por ejemplo, una latitud y longitud determinadas se pueden representar de la siguiente manera:


  binned_latitude = [0, 0, 0, 1, 0]
  binned_longitude = [0, 1, 0, 0, 0]
Supongamos que creas una combinación de atributos de estos dos vectores de atributos:


  binned_latitude X binned_longitude
Esta combinación de atributos es un vector de un solo 1 con 25 elementos (24 ceros y 1 uno). El único 1 en la combinación identifica una conjunción en particular de latitud y longitud. El modelo puede aprender asociaciones particulares sobre esa conjunción.

Supongamos que discretizamos latitud y longitud de manera mucho más amplia, de la siguiente manera:


binned_latitude(lat) = [
  0  < lat <= 10
  10 < lat <= 20
  20 < lat <= 30
]

binned_longitude(lon) = [
  0  < lon <= 15
  15 < lon <= 30
]
La creación de una combinación de atributos de esos discretizaciones groseras genera un atributo sintético con los siguientes significados:


binned_latitude_X_longitude(lat, lon) = [
  0  < lat <= 10 AND 0  < lon <= 15
  0  < lat <= 10 AND 15 < lon <= 30
  10 < lat <= 20 AND 0  < lon <= 15
  10 < lat <= 20 AND 15 < lon <= 30
  20 < lat <= 30 AND 0  < lon <= 15
  20 < lat <= 30 AND 15 < lon <= 30
]
Ahora supongamos que nuestro modelo necesita predecir qué tan satisfechos estarán los dueños de perros con los perros en función de dos atributos:

Tipo de comportamiento (ladrido, llanto, acurrucación, etc.)
Hora del día
Si compilamos una combinación de atributos a partir de estos dos atributos:


  [behavior type X time of day]
obtendremos una capacidad de predicción mucho mayor que cualquiera de las funciones. Por ejemplo, si un perro llora (de felicidad) a las 5:00 p.m. cuando el dueño regresa del trabajo, probablemente será un excelente predictor positivo de la satisfacción del propietario. Llorar (tal vez con tristeza) a las 3:00 a.m. cuando el propietario estaba durmiendo profundamente probablemente sea un fuerte predictor negativo de la satisfacción del propietario.

Los alumnos lineales se ajustan bien a los datos masivos. Usar combinaciones de atributos en conjuntos de datos masivos es una estrategia eficiente para aprender modelos muy complejos. Las redes neuronales proporcionan otra estrategia
[Practica!](https://developers.google.com/machine-learning/crash-course/feature-crosses/playground-exercises?hl=es-419)

# Combinaciones de atributos: Ejercicio de programación

Haremos un ejuercico muy util, ve al archivo: `rEPRESENTATION_WITH_A_FEATURE_CROSS.ipynb`
[link](https://developers.google.com/machine-learning/crash-course/feature-crosses/programming-exercise?hl=es-419)

# Combinaciones de atributos: Comprueba tu comprensión
[Comprueba tu comprensión](https://developers.google.com/machine-learning/crash-course/feature-crosses/check-your-understanding?hl=es-419)

