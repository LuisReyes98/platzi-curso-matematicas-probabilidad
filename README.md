# Clases del Curso de Matemáticas para Data Science: Probabilidad

[Slides del curso](https://static.platzi.com/media/public/uploads/slides-probabilidad-data-science_3af3bfa5-a2f6-4437-83aa-59caf778c2c7.pdf)

## ¿Qué es la probabilidad?

Axiomas de la probabilidad

- Suceso elemental
  el resultado de lanzar un dado es 4

- Suceso
  El resultado de lanzar un dado es par

$
P = \frac{N° sucesos exitosos}{N° sucesos totales}
$

### RESUMEN: ¿Qué es probabilidad?

La probabilidad es una creencia que tenemos sobre la ocurrencia de eventos elementales.

¿En qué casos usamos la probabilidad?

Intuitivamente, hacemos estimaciones de la probabilidad de que algo ocurra o no, al desconocimiento
que tenemos sobre la información relevante de un evento lo llamamos incertidumbre.

El azar en estos términos no existe, representa la ausencia del conocimiento de todas
las variables que componen un sistema.

En otras palabras, la probabilidad es un lenguaje que nos permite cuantificar la incertidumbre

#### AXIOMAS:

Es un conjunto de sentencias que no son derivables de algo más fundamental. Las damos por verdad
y no requieren demostración.

“A veces se compara a los axiomas con semillas, porque de ellas surge toda la teoría”
Axiomas
AXIOMAS DE PROBABILIDAD :

La probabilidad está dada por el número de casos de éxito sobre la cantidad total(teórica) de casos.

P = #-Casos de éxito/ # Casos-totales.

Suceso elemental: Es una única ocurrencia, “Solo tienes una cara de la moneda como resultado”

Sucesos: Son las posibilidades que tenemos en el sistema. Está compuesto de sucesos elementales,
por ejemplo, “El resultado de lanzar un dado es par”, hay tres sucesos (2,4,6) que componen este enunciado.

De la interpretación del axioma anterior divergen dos escuelas de pensamiento. Frecuentista y Bayesiana

Ejemplo: “Solo tengo dos posibles resultados al lanzar una moneda, 50% de probabilidad para cada cara
, (1/2 y 1/2), si lanzo la moneda n veces, la moneda no cae la mitad de las veces en una cara, y luego la otra”

Esta equiprobabilidad de ocurrencia en un espacio muestral ocurre bajo el supuesto de que
la proporción de exitos/totales tiende a un valor p. En otras palabras, solo lanzando la moneda
infinitas veces podemos advertir que el valor de la probabilidad es cercano a (1/2 o 50%).
Escuela frecuentista

“Toda variable aleatoria viene descrita por el espaci muestral que contiene todos los posibles sucesos
de ese problema aleatorio.”

La probabilidad que se asigna como un valor a cada posible suceso tiene varias propiedades por cumplirse

#### PROPIEDADES AXIOMAS:

0 <= P <= 1
Certeza: P = 1
Imposibilidad P = 0
Disyunción P(AuB) = P(A) +P(B)

![probabilidad-1](./images/probabilidad-1.webp)
![probabilidad-2](./images/probabilidad-2.webp)
![probabilidad-4](./images/probabilidad-4.webp)
![probabilidad-3](./images/probabilidad-3.webp)

## Probabilidad en machine learning

¿Cuáles son las fuentes de incertidumbre?

- Datos: Debido a que nuestros instrumentos de medición tienen un margen de error, se presentan datos imperfectos e incompletos, por lo tanto hay incertidumbre en los datos.
- Atributos del modelo: Son variables que representan un subconjunto reducido de toda la realidad del problema, estas variables provienen de los datos y por lo tanto presentan cierto grado de incertidumbre.
- Arquitectura del modelo: Un modelo en mates es una representación simplificada de la realidad y al ser así, por construcción, induce otra capa de incertidumbre, ya que al ser una representación simplificada se considera mucho menos información.

Y claro, todo esta incertidumbre se puede cuantificar con probabilidad:

Ejemplo, un clasificador de documento de texto:

![Modelo clasificacion](./images/modelo_clasificacion.png)

Entonces, el modelo asignara cierta probabilidad a cada documento y así de determinara la clasificación de los documentos.

Pero, ¿cómo funciona por dentro nuestro modelo de clasificación?

![Modelo clasificacion 2](./images/modelo_clasificacion2.png)

So, ¿En dónde se aplica la probabilidad?

Bueno, en realidad no todos los modelos probabilístico, a la hora de diseñarlo nosotros elegimos sui queremos que sea un modelo probabilístico o no.

Por ejemplo si escogemos el modelo de Naive Vayes, luego de que escogemos el diseño ahora definimos el entrenamiento y este es básicamente que el modelo aprenda el concepto de distribución de probabilidad y es una manera que yo uso para saber que probabilidades le asigno una de las posibles ocurrencias de mis datos, de ahí sirgue el esquema MLE que es el estimador de máxima verosimilitud y luego de esto esta la calibración se configuran los hiper-parámetros, esto se entiende mas en redes neuronales artificiales en donde el numero de neuronas de una capa tiene 10 neuronas y cada una tiene sus propios pesos que conectan a las neuronas, entonces esos pesos los podemos ir calibrando para que el modelo sea cada vez mas pequeño. Sin embargo, hay parámetros están fuera del modelo y no se pueden calibrar y a esos parámetros les llamamos hiper-parámetros, porque están fuera de todo ese esquema de optimización. Al final se hace la optimización de los hiper parámetros. Y al final tenemos la interpretación, para interpretar hay veces que se tiene que saber el funcionamiento del modelo y aplicar conceptos de estadística para poder interpretarlo.

## Tipos de probabilidad

- Conjunta (joint)
- Marginal
- Condicional

Regla del producto

$$
P(A,B) = P(A | B) * P(B)
$$

![conjunta1](./images/conjunta1.webp)

Esta probabilidad de un suceso, en realidad corresponde a 2 sucesos separados y cuando yo calculo la probabilidad de 2 o mas sucesos, a eso es a lo que se le llama probabilidad conjunta.

![conjunta2](./images/conjunta2.webp)

La condición de “dado B en par”, lo que hizo fue restringir y paso de tener 36 posibilidades a ahora tener 18 ocurrencias distintas. Y como la probabilidad de A depende de B, entonces la llamamos probabilidad condicional.
Ya ahora para calcular la probabilidad, nos damos cuenta que el numero de eventos exitosos no cambio, si no que lo que cambio fue el numero de eventos posibles.

![conjunta3](./images/conjunta3.webp)

Ahora, nos vamos a reformular le pregunta. ¿cuál es la probabilidad de que el dado B caiga en par?
Aquí no se impone ninguna condición y yo puedo usar todo el espacio muestral completo. De la misma manera tengo 36 opciones y de esas 36 opciones, ¿cuantas corresponder al dado b en par? Son 18.
Ahora si multiplico 9/18 * 10/36 = 1/4 y da exactamente la probabilidad conjunta.
Por lo tanto, la probabilidad conjunta de que suceda A y B es igual a la probabilidad condicional de que suceda A dado que ya sé que sucedió B, por la probabilidad de que suceda B. Esto la resultado a la regla del producto..

![conjunta4](./images/conjunta4.webp)

Ahora, ¿qué es la probabilidad marginal?

Es cuando se obtiene una probabilidad sencilla a partir de una probabilidad conjunta. Es decir cuando se tiene las probabilidades conjuntas de 2 sucesos y se quiere saber solo la probabilidad de que suceda el primer suceso independiente de lo que pasa con el otro, así eso se define como la suma de todas la probabilidades conjuntas sobre los demás estados que no está considerando A.

La probabilidad marginal es la misma probabilidad de la regla del producto, pero la relación fundamental siempre será la regla del producto.

## Ejemplos de cálculo de probabilidad

### Correlaciones de eventos

- A = { el resultado de lanzar un dado es 4}

- B = {el resultado de lanzar un dado es par}

- C = {el resultado de lanzar un dado es impar}

$A = \frac{1}{6}$

$P(A|B) = \frac{1}{3}$

en base a esto se puede decir que A y B estan positivamente correlacionados

$P(A|C) = \frac{0}{3} = 0$

A y C estan negativamente correlacionados
Por lo cual son eventos exclutentes y dependientes

![correlaciones](./images/correlaciones_resumen.webp)

### Reto

Si el jugador 1 escoge {1,2,3,4} de las 8 opciones y el jugador 2 escoge {2,3,6,7} la probabilidad de que el jugador 2 gane dado que gano el jugador 1 es: $P(1|2) = \frac{1}{2} = 0.5$
