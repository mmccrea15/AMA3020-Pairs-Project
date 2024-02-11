# AMA3020-Pairs-Project

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Estimating the volume of a sphere of unit length using the Monte Carlo method of integration (using 1000 randomly distributed points)#

def monte_carlo_volume(num_samples, dimension):
  points = np.random.uniform(-1, 1, size=(num_samples, dimension))
  distances = np.linalg.norm(points, axis=1)

  inside_sphere = np.sum(distances <= 1)

  inside_x = points[distances <= 1, 0]
  inside_y = points[distances <= 1, 1]
  inside_z = points[distances <= 1, 2] if dimension == 3 else []

  sphere_volume_estimate = (2**dimension) * inside_sphere / num_samples
  return sphere_volume_estimate, inside_x, inside_y, inside_z

if __name__ == "__main__":
  num_samples = 1000
  dimension = 3 
  volume_estimate, inside_x, inside_y, inside_z = monte_carlo_volume(num_samples, dimension)

  if dimension == 2:
      plt.scatter(inside_x, inside_y, color='blue', label='Inside the Sphere')
      plt.title(f'Monte Carlo Estimation of Sphere Volume (2D): {volume_estimate:.5f}')
      plt.xlabel('X-axis')
      plt.ylabel('Y-axis')
      plt.legend()
      plt.savefig('88.png')

  elif dimension == 3:
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ax.scatter(inside_x, inside_y, inside_z, c='blue', label='Inside the Sphere')
      ax.set_title(f'Monte Carlo Estimation of Sphere Volume (3D): {volume_estimate:.5f}')
      ax.set_xlabel('X-axis')
      ax.set_ylabel('Y-axis')
      ax.set_zlabel('Z-axis')
      ax.legend()
      plt.savefig('89.png', dpi=300)

  else:
      print(f"Plotting is supported only for dimensions 2 and 3.")



################# Plotting the three problem density functions against the test function for importance sampling ###############

x = np.linspace(0, 10, 1000)
plt.figure(figsize=(10, 5))

y = np.exp(-2*np.abs(x-5)) 
m = np.exp(-2*np.abs(x-5))-0.1
n = -np.exp(-2*np.abs(x-5))+1
o = 1/(np.sqrt(2*np.pi))*np.exp(-0.5*(x-5)**2)
p = -1/(np.sqrt(2*np.pi))*np.exp(-0.5*(x-5)**2)+0.5
q = 0.3

plt.subplot(1, 3, 2)
plt.plot(x, y, label='$f(x) = e^{-2|x-5|}$')
plt.plot(x, uniform_pdf, 'g--', label='Uniform PDF')
plt.title('Simple Uniform Sampling')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.xlim(3,7)
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
#plt.legend()

plt.subplot(1, 3, 1)
plt.plot(x, y, label='$f(x) = e^{-2|x-5|}$')
plt.plot(x, p, 'g--', label='PDF')
plt.title('Increased Variance')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.xlim(3,7)
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
#plt.legend()

plt.subplot(1, 3, 3)
plt.plot(x, y, label='$f(x) = e^{-2|x-5|}$')
plt.plot(x, o, 'g--', label='PDF')
plt.title('Variance Reduction')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.xlim(3,7)
#plt.legend()
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))


plt.tight_layout()
plt.savefig('86.png',dpi=300)


############## Plotting the function we used for the standard test with the chosen probabilty density function #################

x = np.linspace(0, 10, 1000)
y = np.exp(-2*np.abs(x-5)) 
n = -np.exp(-2*np.abs(x-5))+1
o = 1/(np.sqrt(2*np.pi))*np.exp(-0.5*(x-5)**2)
p = -1/(np.sqrt(2*np.pi))*np.exp(-0.5*(x-5)**2)+0.5

plt.plot(x, y, label='$f(x) = e^{-2|x-5|}$')
plt.plot(x, o, 'g--', label='$w(x)=\\frac{1}{\\sqrt{2\\pi}} e^{-0.5(x-5)^2}$')
plt.title('Functions used in the standard test')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.xlim(3,7)
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))


plt.tight_layout()
plt.savefig('87.png',dpi=300)
