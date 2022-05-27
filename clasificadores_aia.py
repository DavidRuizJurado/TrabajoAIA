
# ===================================================================
# Ampliación de Inteligencia Artificial, 2021-22
# PARTE I del trabajo práctico: Implementación de clasificadores 
# Dpto. de CC. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================


# --------------------------------------------------------------------------
# Autor(a) del trabajo:
#
# APELLIDOS: Ruiz Jurado
# NOMBRE: David
#
# Segundo(a) componente (si se trata de un grupo):
#
# APELLIDOS: Navarro Moreno
# NOMBRE: Francisco Javier
# ----------------------------------------------------------------------------


# *****************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen, por lo
# que debe realizarse de manera individual. La discusión y el intercambio de
# información de carácter general con los compañeros se permite, pero NO AL
# NIVEL DE CÓDIGO. Igualmente el remitir código de terceros, OBTENIDO A TRAVÉS
# DE LA RED o cualquier otro medio, se considerará plagio. Si tienen
# dificultades para realizar el ejercicio, consulten con el profesor. En caso
# de detectarse plagio, supondrá una calificación de cero en la asignatura,
# para todos los alumnos involucrados. Sin perjuicio de las medidas
# disciplinarias que se pudieran tomar. 
# *****************************************************************************


# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES, MÉTODOS
# Y ATRIBUTOS QUE SE PIDEN. EN PARTICULAR: NO HACERLO EN UN NOTEBOOK.

# NOTAS: 
# * En este trabajo NO SE PERMITE usar Scikit Learn (excepto las funciones que
#   se usan en carga_datos.py). 
# * SE RECOMIENDA y SE VALORA especialmente usar numpy. Las implementaciones 
#   saldrán mucho más cortas y eficientes.  

# *****************************************
# CONJUNTOS DE DATOS A USAR EN ESTE TRABAJO
# *****************************************

# Para aplicar las implementaciones que se piden en este trabajo, vamos a usar
# los siguientes conjuntos de datos. Para cargar todos los conjuntos de datos,
# basta con descomprimir el archivo datos-trabajo-aia.tgz y ejecutar el
# archivo carga_datos.py (algunos de estos conjuntos de datos se cargan usando
# utilidades de Scikit Learn, por lo que para que la carga se haga sin
# problemas, deberá estar instalado el módulo sklearn). Todos los datos se
# cargan en arrays de numpy.

# * Datos sobre concesión de prestamos en una entidad bancaria. En el propio
#   archivo datos/credito.py se describe con más detalle. Se carga en las
#   variables X_credito, y_credito.   

# * Conjunto de datos de la planta del iris. Se carga en las variables X_iris,
#   y_iris.  

# * Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#   17 votaciones realizadas durante 1984. Se trata de clasificar el partido al
#   que pertenece un congresita (republicano o demócrata) en función de lo
#   votado durante ese año. Se carga en las variables X_votos, y_votos. 

# * Datos de la Universidad de Wisconsin sobre posible imágenes de cáncer de
#   mama, en función de una serie de características calculadas a partir de la
#   imagen del tumor. Se carga en las variables X_cancer, y_cancer.
  
# * Críticas de cine en IMDB, clasificadas como positivas o negativas. El
#   conjunto de datos que usaremos es sólo una parte de los textos. Los textos
#   se han vectorizado usando CountVectorizer de Scikit Learn, con la opción
#   binary=True. Como vocabulario, se han usado las 609 palabras que ocurren
#   más frecuentemente en las distintas críticas. La vectorización binaria
#   convierte cada texto en un vector de 0s y 1s en la que cada componente indica
#   si el correspondiente término del vocabulario ocurre (1) o no ocurre (0)
#   en el texto (ver detalles en el archivo carga_datos.py). Los datos se
#   cargan finalmente en las variables X_train_imdb, X_test_imdb, y_train_imdb,
#   y_test_imdb.    

# * Un conjunto de imágenes (en formato texto), con una gran cantidad de
#   dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#   base de datos MNIST. En digitdata.zip están todos los datos en formato
#   comprimido. Para preparar estos datos habrá que escribir funciones que los
#   extraigan de los ficheros de texto (más adelante se dan más detalles). 



# ==================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA 
# ==================================================

# Definir una función 

#           particion_entr_prueba(X,y,test=0.20)

# que recibiendo un conjunto de datos X, y sus correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test. La división ha de ser ALEATORIA y
# ESTRATIFICADA respecto del valor de clasificación. Por supuesto, en el orden 
# en el que los datos y los valores de clasificación respectivos aparecen en
# cada partición debe ser consistente con el orden original en X e y.   

# ------------------------------------------------------------------------------
# Ejemplos:
# =========

# En votos:

# >>> Xe_votos,Xp_votos,ye_votos,yp_votos
#         =particion_entr_prueba(X_votos,y_votos,test=1/3)

# Como se observa, se han separado 2/3 para entrenamiento y 1/3 para prueba:
# >>> y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0]
#    (435, 290, 145)

# Las proporciones entre las clases son (aprox) las mismas en los dos conjuntos de
# datos, y la misma que en el total: 267/168=178/112=89/56

# >>> np.unique(y_votos,return_counts=True)
#  (array([0, 1]), array([168, 267]))
# >>> np.unique(ye_votos,return_counts=True)
#  (array([0, 1]), array([112, 178]))
# >>> np.unique(yp_votos,return_counts=True)
#  (array([0, 1]), array([56, 89]))

# La división en trozos es aleatoria y, por supuesto, en el orden en el que
# aparecen los datos en Xe_votos,ye_votos y en Xp_votos,yp_votos, se preserva
# la correspondencia original que hay en X_votos,y_votos.


# Otro ejemplo con más de dos clases:

# >>> Xe_credito,Xp_credito,ye_credito,yp_credito
#          =particion_entr_prueba(X_credito,y_credito,test=0.4)

# >>> np.unique(y_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([202, 228, 220]))

# >>> np.unique(ye_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([121, 137, 132]))

# >>> np.unique(yp_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([81, 91, 88]))
# ------------------------------------------------------------------










# ========================================================
# EJERCICIO 2: MODELOS LINEALES PARA CLASIFICACIÓN BINARIA
# ========================================================


# En este ejercicio se propone la implementación de dos clasificadores lineales:
# perceptrón y regresión logística (mini-batch).

# RECOMENDACIÓN: Siempre que se pueda, tratar de evitar bucles for para recorrer 
# los datos, usando en su lugar funciones de numpy. 


# ----------------------------------
# 2.1) Implementación del perceptrón
# ----------------------------------


# Implementar una clase: 

# class Perceptron():

#    def __init__(self,normalizacion=False,
#                 rate=0.1,rate_decay=False,n_epochs=100,
#                 pesos_iniciales=None):

#         .....
        
#     def entrena(self,X,y):

#         .....        

    
#     def clasifica(self,ejemplos):
                        
#          ......

        

# Explicamos a continuación cada uno de estos elementos:


# * El constructor tiene los siguientes argumentos de entrada:


#   + El parámetro normalizacion, que puede ser True o False (False por
#     defecto). Indica si los datos se tienen que normalizar, tanto para el
#     entrenamiento como para la clasificación de nuevas instancias. La
#     normalización es la estándar: a cada característica se le resta la media
#     de los valores de esa característica en el conjunto de entrenamiento, y
#     se divide por la desviación típica de dichos valores.

#  + rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#    durante todo el aprendizaje. Si rate_decay es True, rate es la
#    tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#  + rate_decay, indica si la tasa de aprendizaje debe disminuir en
#    cada epoch. En concreto, si rate_decay es True, la tasa de
#    aprendizaje que se usa en el n-ésimo epoch se debe de calcular
#    con la siguiente fórmula: 
#       rate_n= (rate_0)*(1/(1+n)) 
#    donde n es el número de epoch, y rate_0 es la cantidad introducida
#    en el parámetro rate anterior. Su valor por defecto es False. 

#  + n_epochs: número de veces que se itera sobre todo el conjunto de
#    entrenamiento. Por defecto 100.

#  + pesos_iniciales: si no es None, es un array con los pesos iniciales. Este
#    parámetro puede ser útil para empezar con unos pesos que se habían obtenido
#    y almacenado como consecuencia de un entrenamiento anterior. Si es None (por defecto), 
#    los pesos iniciales se crean aleatoriamente con números decimales entre -1 y 1.


# * El método entrena tiene como parámetros de entrada dos arrays numpy X e y, con
#   los datos del conjunto de entrenamiento y su clasificación esperada,
#   respectivamente. Las dos clases del problema son las que aparecen en el array y, 
#   y se deben almacenar en un atributo self.clases en una lista. La clase que
#   se considera positiva es la que aparece en segundo lugar en esa lista. 

# * Método clasifica: recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY de clases que el modelo predice para esos ejemplos.   

# Si se llama a los métodos de clasificación antes de entrenar el modelo, se
# debe devolver (con raise) una excepción:

class ClasificadorNoEntrenado(Exception): pass


# ----------------------------------------------------------------------------
# Ejemplos:




# Sobre los datos del cáncer de mama:
# ------------------------------------

# >>> Xe_cancer,Xp_cancer,ye_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer)

# >>> perc_cancer=Perceptron(rate=0.1,rate_decay=True,normalizacion=True)

# >>> perc_cancer.entrena(Xe_cancer,ye_cancer)

# >>> perc_cancer.clases
# array([0, 1])     # 0 maligno, 1 benigno (la clase positiva)

# >>> perc_cancer.clasifica(Xp_cancer[14:17])
# array([1, 0, 1])   # Los ejemplos 14 y 16 se clasifican como benignos, el 15 maligno

# >>> yp_cancer[14:17]
# array([1, 0, 1])   # La predicción anterior coincide con los valores esperado para esos ejemplos












# ----------------------------------------------
# 2.2) Implementación del cálculo de rendimiento
# ----------------------------------------------

# Definir una función "rendimiento(clasificador,X,y)" que devuelve la
# proporción de ejemplos bien clasificados (accuracy) que obtiene un
# clasificador sobre un conjunto de ejemplos X con clasificación esperada y. 
# Supondremos que el clasificador tiene un método clasifica como el del
# ejercicio anterior. 



# Ejemplos (sobre los datos del cáncer de mama (entrenamiento y prueba):
# ----------------------------------------------------------------------

# >>> rendimiento(perc_cancer,Xe_cancer,ye_cancer)
#  0.9846491228070176

# >>> rendimiento(perc_cancer,Xp_cancer,yp_cancer)
#  0.9734513274336283

# ------------------------------------------------------------------------------








# ----------------------------------------------------------------
# 2.3) Implementación de regresión logística (versión mini-batch)
# ---------------------------------------------------------------

# En esta sección se pide implementar un clasificador BINARIO basado en
# regresión logística, con algoritmo de entrenamiento de descenso por el
# gradiente mini-batch (para minimizar la entropía cruzada).

# En concreto se pide implementar una clase: 

# class RegresionLogisticaMiniBatch():

#    def __init__(self,normalizacion=False,
#                 rate=0.1,rate_decay=False,n_epochs=100,
#                 pesos_iniciales=None,batch_tam=64):

#         .....
        
#     def entrena(self,X,y,salida_epoch=False):

#         .....        

#     def clasifica_prob(self,ejemplos):

#         ......
    
#     def clasifica(self,ejemplo):
                        
#          ......

        

# La descripción de los argumentos del constructor y los métodos es la misma que en 
# el caso del perceptrón, incluyendo ahora además:
    
# * en el constructor, un parámetro batch_tam indicando el tamaño de minibatch
    
# * un método clasifica_prob, que recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY con las probabilidades que el modelo 
#   asigna a cada ejemplo de pertenecer a la clase 1.       
    
# * en el método entrena, un parámetro booleano salida_epoch (False por defecto).
#   Si es True, al inicio y durante el entrenamiento, cada epoch se imprime 
#   el valor de la entropía cruzada del modelo respecto del conjunto de entrenamiento, 
#   y su rendimiento. Esta opción puede ser útil para comprobar si el entrenamiento 
#   efectivamente está haciendo descender la entropía cruzada del modelo 
#   (recordemos que el objetivo del entrenamiento es encontrar los pesos que 
#    minimizan la entropía cruzada), y está haciendo subir el rendimiento.      

# RECOMENDACIONES: 

# * Definir la función sigmoide usando la función expit de scipy.special, 
#   para evitar "warnings" por "overflow":

# from scipy.special import expit    
#
# def sigmoide(x):
#    return expit(x)

# * Usar np.where para definir la entropía cruzada. 

# -------------------------------------------------------------

# Ejemplo, usando los datos del cáncer de mama (los resultados pueden variar):


# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True)

# >>> lr_cancer.entrena(Xe_cancer,ye_cancer)

# >>> lr_cancer.clasifica(Xp_cancer[21:24])
# array([0, 1, 1])   # Predicción para los ejemplos 20,21 y 22 

# >>> yp_cancer[21:24]
# array([0, 1, 1])   # La predicción anterior coincide con los valores esperado para esos ejemplos

# >>> lr_cancer.clasifica_prob(Xp_cancer[21:24])
# array([1.94266271e-12, 1.00000000e+00, 1.00000000e+00])

# >>> rendimiento(lr_cancer,Xe_cancer,ye_cancer)
# 0.9912280701754386

# >>> rendimiento(lr_cancer,Xp_cancer,yp_cancer)
# 0.9557522123893806




# Ejemplo con salida_epoch:

# >>> lr_cancer2=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,
#                                normalizacion=True,n_epochs=10)

# >>> lr_cancer2.entrena(Xe_cancer,ye_cancer,salida_epoch=True)

#  Inicialmente, entropía cruzada: 1237.1485717354321, rendimiento: 0.15570175438596492.
#  Epoch 1, entropía cruzada: 56.22827774461466, rendimiento: 0.9692982456140351.
#  Epoch 2, entropía cruzada: 45.910706259544234, rendimiento: 0.9714912280701754.
#  Epoch 3, entropía cruzada: 42.60926295720725, rendimiento: 0.9736842105263158.
#  Epoch 4, entropía cruzada: 40.8449125104497, rendimiento: 0.9736842105263158.
#  Epoch 5, entropía cruzada: 39.387894976559025, rendimiento: 0.9736842105263158.
#  Epoch 6, entropía cruzada: 38.36631713731114, rendimiento: 0.9736842105263158.
#  Epoch 7, entropía cruzada: 37.56556735926138, rendimiento: 0.9736842105263158.
#  Epoch 8, entropía cruzada: 36.91856165719209, rendimiento: 0.9736842105263158.
#  Epoch 9, entropía cruzada: 36.40106899866587, rendimiento: 0.9736842105263158.
#  Epoch 10, entropía cruzada: 35.958374630203544, rendimiento: 0.9758771929824561.



# -----------------------------------------------------------------












# =================================================
# EJERCICIO 3: IMPLEMENTACIÓN DE VALIDACIÓN CRUZADA
# =================================================

# Este ejercicio ES OPCIONAL, se puede saltar sin afectar al resto del trabajo,
# y no es necesario realizarlo para conseguir la nota máxima. 
          
# Supone 0.2 puntos adicionales (a los 2 puntos del trabajo). 


# Puede servir para el ajuste de parámetros en los ejercicios posteriores, 
# pero si no se realiza, se podrían ajustar siguiendo el método "holdout" 
# implementado en el ejercicio 1


# Definir una función: 

#  rendimiento_validacion_cruzada(clase_clasificador,params,X,y,n=5)

# que devuelve el rendimiento medio de un clasificador, mediante la técnica de
# validación cruzada con n particiones. Los arrays X e y son los datos y la
# clasificación esperada, respectivamente. El argumento clase_clasificador es
# el nombre de la clase que implementa el clasificador. El argumento params es
# un diccionario cuyas claves son nombres de parámetros del constructor del
# clasificador y los valores asociados a esas claves son los valores de esos
# parámetros para llamar al constructor.

# INDICACIÓN: para usar params al llamar al constructor del clasificador, usar
# clase_clasificador(**params)  

# ------------------------------------------------------------------------------
# Ejemplo:
# --------
# Lo que sigue es un ejemplo de cómo podríamos usar esta función para
# ajustar el valor de algún parámetro. En este caso aplicamos validación
# cruzada, con n=5, en el conjunto de datos del cancer, para estimar cómo de
# bueno es el valor batch_tam=16 con rate_decay en regresión logística mini_batch.
# Usando la función que se pide sería (nótese que debido a la aleatoriedad, 
# no tiene por qué coincidir el resultado):

# >>> rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                  {"batch_tam":16,"rate_decay":True},
#                                  Xe_cancer,ye_cancer,n=5)
# 0.9121095227289917



# El resultado es la media de rendimientos obtenidos entrenando cada vez con
# todas las particiones menos una, y probando el rendimiento con la parte que
# se ha dejado fuera. Las particiones deben ser aleatorias y estratificadas. 
 
# Si decidimos que es es un buen rendimiento (comparando con lo obtenido para
# otros valores de esos parámetros), finalmente entrenaríamos con el conjunto de
# entrenamiento completo:

# >>> lr16=RegresionLogisticaMiniBatch(batch_tam=16,rate_decay=True)
# >>> lr16.entrena(Xe_cancer,ye_cancer)

# Y daríamos como estimación final el rendimiento en el conjunto de prueba, que
# hasta ahora no hemos usado:
# >>> rendimiento(lr16,Xp_cancer,yp_cancer)
# 0.9203539823008849

#------------------------------------------------------------------------------














# ===================================================
# EJERCICIO 4: APLICANDO LOS CLASIFICADORES BINARIOS
# ===================================================



# Usando los dos modelos implementados en el ejercicio 3, obtener clasificadores 
# con el mejor rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Cáncer de mama 
# - Críticas de películas en IMDB

# Ajustar los parámetros para mejorar el rendimiento (no es necesario ser muy 
# exhaustivo, tan solo probar algunas combinaciones). Si se ha hecho el ejercicio 3, 
# usar validación cruzada para el ajuste (si no, usar el "holdout" del ejercicio 1). 

# Mostrar el proceso realizado en cada caso, y los rendimientosfinales obtenidos
# sobre un conjunto de prueba.     

# Mostrar también, para cada conjunto de datos, un ejemplo con salida_epoch, 
# en el que se vea cómo desciende la entropía cruzada y aumenta el 
# rendimiento durante un entrenamiento.     


# =====================================
# EJERCICIO 5: CLASIFICACIÓN MULTICLASE
# =====================================

# Se pide implementar un algoritmo de regresión logística para problemas de
# clasificación en los que hay más de dos clases, usando  la técnica One vs Rest. 


#  Para ello, implementar una clase  RL_OvR con la siguiente estructura, y que 
#  implemente un clasificador OvR (one versus rest) usando como base el
#  clasificador binario RegresionLogisticaMiniBatch


# class RL_OvR():

#     def __init__(self,rate=0.1,rate_decay=False,normalización=False
#                   batch_tam=64,n_epochs=100):

#        ......

#     def entrena(self,X,y):

#        .......

#     def clasifica(self,ejemplos):

#        ......
            



#  Los parámetros de los métodos significan lo mismo que en el apartado
#  anterior. 

#  Un ejemplo de sesión, con el problema del iris:


# --------------------------------------------------------------------
# >>> Xe_iris,Xp_iris,ye_iris,yp_iris=particion_entr_prueba(X_iris,y_iris)

# >>> rl_iris=RL_OvR([0,1,2],rate=0.001,batch_tam=20,n_epochs=50)

# >>> rl_iris.entrena(Xe_iris,ye_iris)

# >>> rendimiento(rl_iris,Xe_iris,ye_iris)
# 0.9732142857142857

# >>> rendimiento(rl_iris,Xp_iris,yp_iris)
# >>> 0.9736842105263158
# --------------------------------------------------------------------







# ==============================================
# EJERCICIO 6: APLICACION A PROBLEMAS MULTICLASE
# ==============================================


# ---------------------------------------------------------
# 6.1) Conjunto de datos de la concesión de crédito
# ---------------------------------------------------------

# Aplicar la implementación del apartado anterior, para obtener un
# clasificador que aconseje la concesión, estudio o no concesión de un préstamo,
# basado en los datos X_credito, y_credito. 
# Ajustar adecuadamente los parámetros (nuevamente, no es necesario ser demasiado exhaustivo)

# NOTA IMPORTANTE: En este caso concreto, los datos han de ser transformados, 
# ya que los atributos de este conjunto de datos no son numéricos. Para ello, usar la llamada 
# "codificación one-hot". Es parte de este ejercicio documentarse para entender
# en qué consiste esta codificación, e implementar una función que transforme los
# datos en X_credito. 








# ---------------------------------------------------------
# 6.2) Clasificación de imágenes de dígitos escritos a mano
# ---------------------------------------------------------


#  Aplicar la implementación o implementaciones del apartado anterior, para obtener un
#  clasificador que prediga el dígito que se ha escrito a mano y que se
#  dispone en forma de imagen pixelada, a partir de los datos que están en el
#  archivo digidata.zip que se suministra.  Cada imagen viene dada por 28x28
#  píxeles, y cada pixel vendrá representado por un caracter "espacio en
#  blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#  (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#  (es decir, no distinguiremos entre el borde y el interior). En cada
#  conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#  clasificaciones de cada imagen (es decir, el número que representan) vienen
#  en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#  funciones python que lean esos ficheros y obtengan los datos en el mismo
#  formato numpy en el que los necesita el clasificador. 

#  Los datos están ya separados en entrenamiento, validación y prueba. En este
#  caso concreto, NO USAR VALIDACIÓN CRUZADA para ajustar, ya que podría
#  tardar bastante (basta con ajustar comparando el rendimiento en
#  validación). Si el tiempo de cómputo en el entrenamiento no permite
#  terminar en un tiempo razonable, usar menos ejemplos de cada conjunto.

# Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
# rate_decay para tratar de obtener un rendimiento aceptable (por encima del
# 75% de aciertos sobre test). 

