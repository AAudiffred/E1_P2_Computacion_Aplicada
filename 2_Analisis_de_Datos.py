import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.stats import norm, linregress

data = pd.read_excel('Estadistica_de_Tallas.xlsx')

height = data['Estatura cm']
height_standarized = height.apply(lambda x: x if x > 5 else x*100)
shoe_size = data['Talla calzado']
sex = data['Sexo']

# Mean and standard deviation ############################################################
mean_height = height_standarized.mean()
mean_shoesize = shoe_size.mean()
std_dev_height = np.std(height_standarized, ddof=1)
std_dev_shoesize = np.std(shoe_size, ddof=1)

# Gaussian function ######################################################################
x_height = np.linspace(mean_height - 3 * std_dev_height, mean_height + 3 * std_dev_height, 100)
y_height = np.exp(-0.5 * ((x_height - mean_height) / std_dev_height)**2) / (std_dev_height * np.sqrt(2 * np.pi))

x_shoe_size = np.linspace(mean_shoesize - 3 * std_dev_shoesize, mean_shoesize + 3 * std_dev_shoesize, 100)
y_shoe_size = np.exp(-0.5 * ((x_shoe_size - mean_shoesize) / std_dev_shoesize)**2) / (std_dev_shoesize * np.sqrt(2 * np.pi))

# Probability of random person ###########################################################
lower_bound_height = mean_height - std_dev_height
upper_bound_height = mean_height + std_dev_height

lower_bound_shoesize = mean_shoesize - std_dev_shoesize
upper_bound_shoesize = mean_shoesize + std_dev_shoesize

prob_first_std_height = norm.cdf(upper_bound_height, mean_height, std_dev_height) - norm.cdf(lower_bound_height, mean_height, std_dev_height)
prob_first_std_shoesize = norm.cdf(upper_bound_shoesize, mean_shoesize, std_dev_shoesize) - norm.cdf(lower_bound_shoesize, mean_shoesize, std_dev_shoesize)

# Linear regression entire sample #########################################################
sq_sum_height = sum((xi - mean_height)**2 for xi in height_standarized)
sq_sum_shoesize = sum((yi - mean_shoesize)**2 for yi in shoe_size)

sum_height_shoesize = sum((xi - mean_height)*(yi - mean_shoesize) for xi, yi in zip(height_standarized,shoe_size))

b = sum_height_shoesize / sq_sum_height #slope
a = mean_shoesize - b * mean_height #intersection
r2 = (sum_height_shoesize / (math.sqrt(sq_sum_height)*math.sqrt(sq_sum_shoesize)))**2 #R

# Linear regression for male and female ####################################################
data_male = data[data['Sexo'] == 'Masculino']
data_female = data[data['Sexo'] == 'Femenino']

height_male = data_male['Estatura cm']
height_male_stand = height_male.apply(lambda x: x if x > 5 else x*100)
shoe_size_male = data_male['Talla calzado']

height_female = data_female['Estatura cm']
height_female_stand = height_female.apply(lambda x: x if x > 5 else x*100)
shoe_size_female = data_female['Talla calzado']

x_male = height_male_stand
y_male = shoe_size_male
coef_male = np.polyfit(x_male, y_male, 1)
p_male = np.poly1d(coef_male)
corr_male = np.corrcoef(x_male, y_male)[0, 1]

x_female = height_female_stand
y_female = shoe_size_female
coef_female = np.polyfit(x_female, y_female, 1)
p_female = np.poly1d(coef_female)
corr_female = np.corrcoef(x_female, y_female)[0, 1]


# Prints y figures #########################################################################
print("Media Altura:", mean_height)
print("Media Talla calzado:", mean_shoesize)
print("Desviación estándar Altura:", std_dev_height)
print("Desviación estándar Talla calzado:", std_dev_shoesize)
print("La probabilidad de que una persona tomada al azar esté dentro de la primera desviación estándar para Estatura es:", prob_first_std_height*100)
print("La probabilidad de que una persona tomada al azar esté dentro de la primera desviación estándar para Talla calzado es:", prob_first_std_shoesize*100)
print("Coeficiente de correlación R =", r2)
print(f"Ecuación de la recta y = {b:.2f}x + {a:.2f}")
print('Regresión lineal para hombres:', p_male)
print("Coeficiente de correlación hombres R=", corr_male)
print('Regresión lineal para mujeres:', p_female)
print("Coeficiente de correlación mujeres R=", corr_female)



fig, axs = plt.subplots(2, 2)

axs[0,0].hist(height_standarized)
axs[0,0].set_title('Histograma de Estatura')
axs[0,0].set_xlabel('Estatura [cm]')
axs[0,0].set_ylabel('Frecuencia')

axs[0,1].hist(shoe_size)
axs[0,1].set_title('Histograma de Talla de calzado')
axs[0,1].set_xlabel('Talla de calzado')
axs[0,1].set_ylabel('Frecuencia')

axs[1,0].plot(x_height, y_height)
axs[1,0].set_xlabel('X')
axs[1,0].set_ylabel('P(X)')
axs[1,0].set_title('Distribución normal de probabilidad para Estatura')

axs[1,1].plot(x_shoe_size, y_shoe_size)
axs[1,1].set_xlabel('X')
axs[1,1].set_ylabel('P(X)')
axs[1,1].set_title('Distribución normal de probabilidad para Talla de Calzado')

plt.show()


plt.scatter(height_standarized, shoe_size, label='Total población')
x_values = [min(height_standarized), max(height_standarized)]
y_values = [b * x + a for x in x_values]
plt.plot(x_values, y_values, 'r', label='Regresión lineal')
plt.xlabel('Estatura [cm]')
plt.ylabel('Talla calzado')
plt.title('Estatura vs. Talla calzado')
plt.legend()
plt.show()


fig2, ax = plt.subplots()
ax.scatter(height_male_stand, shoe_size_male, label='Masculino')
ax.scatter(height_female_stand, shoe_size_female, label='Femenino')
ax.plot(x_male, p_male(x_male), color='blue', label='Regresión lineal hombres')
ax.plot(x_female, p_female(x_female), color='orange', label='Regresión lineal mujeres')
ax.plot(x_values, y_values, 'r--', label='Regresión lineal población total')
ax.set_xlabel('Estatura [cm]')
ax.set_ylabel('Talla calzado')
ax.legend()
plt.title('Estatura vs. Talla de calzado por Sexo')
plt.show()