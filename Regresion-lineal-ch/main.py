import pandas as pd
import matplotlib.pyplot as plt
#lectura de datos

names = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]

# I read the dataset and store it in a dataframe from pandas
df = pd.read_csv('winequality-red (1).csv',names=names , sep=';')

# I select alcohol and ph as my independent variables
X = df['alcohol']
# The wine quality will be the dependent variable, the one to be predicted
Y = df['quality']
# valores iniciales
m = 0
b = 0
#learning rate
learning_rate = 0.0001
#la cantidad de epocas a ejecutar
epocas = 100
#longitud del dataset
n = len(X) 
# Calcular el mse.
# se saca la diferencia entre cada dato real y cada prediccion
# se suman estas diferencias, se eleva al cuadrado y se divide
# entre n 
def compute_error (y, m, x, b, n):
  acum = 0
  for i in range (0,n):
    diff = y[i] - (m*x[i] +b)
    acum += diff**2
  acum = acum / n
  return acum 
# algoritmo de gradient descent
# se ejecuta la cantidad de epocas definidas
for i in range(epocas): 
    #se obtiene la hipotesis
    Y_pred = m*X + b  
    error = compute_error (Y, m, X, b, n)
  # sacamos las derivadas usando la sumatoria de las diferencias entre la prediccion y el valor real (errores)
    Deriv_m = (-2/n) * sum(X * (Y - Y_pred)) 
    Deriv_b = (-2/n) * sum(Y - Y_pred)  
  #obtengo el nuevo valor para m y b
    m = m - learning_rate * Deriv_m  
    b = b - learning_rate * Deriv_b  
  #calculo el error usando mse
  
  #imprimo los valores
    print("iteration: ",i ,"new values: ", m , b, "MSE:", error)
print("Final equation:")
print ("y =", m ,"x +", b)


# graficamos los resultados 
plt.scatter(X, Y, color= "blue")
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='pink')
plt.show()
