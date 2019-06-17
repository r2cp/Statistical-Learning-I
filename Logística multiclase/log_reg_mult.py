

# Tomar en cuenta el rango y dimensiones de los tensores
# Los datos de entrada x crudos son una matriz, pero el modelo lineal necesita un vector, hacer reshape a la forma adecuada
# cada pixel de la imagen pasa a ser una feature
x = tf.placeholder(tf.float32, [?, ?]) 
# para cada imagen(o el vector que la representa)  necesitamos como salida una dist. de probabilidad lo cual se traduce a un vector de tamaño "n" con n = al numero de posibles categorias
y_ = tf.placeholder(tf.float32, [?, ?]) 

# definir los parametros entrenables incluyendo el "bias" o w0,debe ser un tensor del tamaño adecuado: por cada feature tenemos "n" salidas con n = al numero de posibles categorias
# tip: W es una matriz
W = tf.Variable() #
b = tf.Variable() # 

y = tf.nn.<remplazar por la función correcta>(tf.matmul(?, ?) + ?)

# el costo o loss es la entropia cruzada, agregar  a tensorboard con un summary.scalar
cross_entropy = tf.reduce_mean(<ver en las presentaciones como calcular el costo >, reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(?).minimize(?) 

# el accuracy mide el porcentaje de observaciones que el modelo clasifica correctamente(comparando contra los datos reales ), agregarlo a tensorboard con summary.scalar
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# nos permite agrupar todos los summaries de nuestro grafo para facilitar calcular todos juntos.
summaries = tf.summary.merge_all()


# la ejecucion(sesion) hacerla con mini-batch gradient descent con un batch size = 32
# cada cierto numero de iteraciones "n" imprimir: el numero de iteracion, el loss de la iteracion, el accuracy de la iteracion y "enviarlo" a tensorboard
