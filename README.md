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



# Plotting the three problem density functions against the test function for importance sampling #

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


# Plotting the function we used for the standard test with the chosen probabilty density function #

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



# code used for project #

a = 0
b = np.pi
N = 1000
#xrand = np.random.uniform(a, b, N)

def func(x):
  return np.sin(x)




#answer = (b-a)/float(N)*integral
#print("The integral is ", answer)

areas = []

for ii in range(N):
  xrand = np.zeros(N)
  for ii in range(len(xrand)):
    xrand[ii] = np.random.uniform(a, b)
  integral = 0.0
  for ii in range(N):
    integral += func(xrand[ii])
  answer = (b-a)/float(N)*integral
  areas.append(answer)

plt.title("Distribution of Areas Calculated")
plt.hist(areas, bins=30, ec = 'black')
plt.xlabel("Areas")
plt.savefig("Histogram")

plt.close()

'''

'''

theta=np.linspace(0,np.pi,100)
y=np.sin(theta)
plt.plot(theta,y,'r')
plt.axis('equal')

xsq=[0,np.pi,np.pi,0,0]
ysq=[0,0,1,1,0]
plt.plot(xsq,ysq,'b')

plt.savefig("shapes.png")

xlist=[]
ylist=[]

for ii in range(1000):
 x=np.random.uniform(0,np.pi)
 xlist.append(x)
 y=np.random.uniform(0,1)
 ylist.append(y)

#plt.plot(xlist,ylist,'k.')

plt.savefig("shapes+dots.png")

xin=[]
yin=[]
xout=[]
yout=[]

for ii in range(1000):
 x=np.random.uniform(0,np.pi)
 y=np.random.uniform(0,1)
 if y < np.sin(x):
  xin.append(x)
  yin.append(y)
 else:
   xout.append(x)
   yout.append(y)

plt.plot(xin,yin,'r.') 
plt.plot(xout,yout,'b.')

plt.savefig("shapes+circle_dots.png")

print(len(xin))
print(len(xout))

print((len(xin)/1000)*np.pi)

plt.close()

def monte_carlo_integration(func, a, b, N):
  xrand = np.random.uniform(a, b, N)
  yrand = func(xrand)
  integral_estimate = np.mean(yrand)*(b-a)
  return integral_estimate

result = monte_carlo_integration(np.sin,0, np.pi, 1000)
print(result)

##################################################################################################################### Graph of function plus weight ##############################
#########################################################################################

p = 1/10*x
x = np.linspace(3, 7, 100)
h = np.exp(-2*np.abs(x-5))
w = 1/(np.sqrt(2*np.pi))*np.exp(-0.5*(x-5)**2)
plt.plot(x,w, 'r', label='w(x)')
plt.plot(x,h, 'y', label='h(x)')
plt.plot(x,h/w, 'b', label='h(x)/w(x)')


#plt.axhline(y=1/10, color='r', label='p(x)')
plt.legend()
plt.savefig("exp.png")
plt.close()

############################################################################################################################## MC of function ####################################
#########################################################################################

x = np.linspace(0, 10, 100)
h = np.exp(-2*np.abs(x-5))

plt.legend()
plt.plot(x,h, 'r', label='h(x)')

#plt.axis('equal')
xsq=[0,10,10,0,0]
ysq=[0,0,1,1,0]
plt.plot(xsq,ysq,'b')

#plt.savefig("act.png")

xin=[]
yin=[]
xout=[]
yout=[]

for ii in range(1000):
 x=np.random.uniform(0,10)
 y=np.random.uniform(0,1)
 if y < np.exp(-2*np.abs(x-5)):
  xin.append(x)
  yin.append(y)
 else:
   xout.append(x)
   yout.append(y)

plt.plot(xin,yin,'r.') 
plt.plot(xout,yout,'b.')

plt.savefig("actual_dots.png")

print(len(xin))
print(len(xout))

print((len(xin)/1000)*10)

plt.close()


#########################################################################################
#######################################Improved##########################################
#########################################################################################

x = np.linspace(0, 10, 100)
h = np.exp(-2*np.abs(x-5))


plt.plot(x,h, 'r', label='h(x)')

#plt.axis('equal')

xsq=[0,10,10,0,0]
ysq=[0,0,1,1,0]
plt.plot(xsq,ysq,'b')
xin=[]
yin=[]
xout=[]
yout=[]

h = np.exp(-2*np.abs(x-5))
w = 1/(np.sqrt(2*np.pi))*np.exp(-0.5*(x-5)**2)

for ii in range(1000):
 x=np.random.uniform(0,10)
 y=np.random.uniform(0,1)
 if y < np.exp(-2*abs(x-5)):
  xin.append(x)
  yin.append(y)
 else:
   xout.append(x)
   yout.append(y)

plt.plot(xin,yin,'r.') 
plt.plot(xout,yout,'b.')

#plt.savefig("actual_dots_improved.png")

print(len(xin))
print(len(xout))

print((len(xin)/1000)*10)

plt.close()
'''
#############################################################################################################

import scipy
'''
# Define the function h(x) 
def h(x): 
  return (np.cos(50*x) + np.sin(20*x))**2
  
# Integrate h(x) from 0 to 1 
integral_value, error = scipy.integrate.quad(h, 0, 1) 

# Calculate the average value of h(x) within the interval [0, 1] 
n = 1000000 
x_values = np.random.uniform(0, 1, n) 
average_h = np.mean(h(x_values)) 

# Calculate the integral I using the average value 
I = (1-0) * average_h 



# Importance Sampling 

# Define the target function f(x) 
def f(x): 
  return np.exp(-2*np.abs(x-5)) 

# Integrate f(x) from 0 to 10 
integral_f, error_f = scipy.integrate.quad(f, 0, 10) 

# Generate samples from g(x) = N(5, 1) 
x_samples = np.random.normal(5, 1, n) 

# Calculate h(x)*f(x)/g(x) for each sample 
weights = 10 * np.exp(-2*np.abs(x_samples-5)) * (1/10) / (1/np.sqrt(2*np.pi) * np.exp(-(x_samples-5)**2/2))

# Estimate the expectation using importance sampling 
I_importance_sampling = np.mean(weights) 
print(I_importance_sampling)
# Variance estimation 
variance_importance_sampling = np.var(weights)
print(variance_importance_sampling)

'''
# Define the function h(x) 
def f(x):
  return np.exp(-2*np.abs(x-5))

integral_value, error = scipy.integrate.quad(f, 0, 10)

# Calculate the average value of f(x) within the interval [0, 10] 
n = 1000000 
x_values = np.random.uniform(0, 10, n) 
average_f = np.mean(f(x_values)) 

# Calculate the integral I using the average value 
I = (10-0) * average_f
print("Monte Carlo value:",I)
variance_I = np.var(10*f(x_values))
print("Variance of MC:",variance_I)

# Importance Sampling 

# Define the target function f(x) 
def f(x): 
  return np.exp(-2*np.abs(x-5)) 

# Integrate f(x) from 0 to 10 
integral_f, error_f = scipy.integrate.quad(f, 0, 10) 

# Generate samples from g(x) = N(5, 1) 
x_samples = np.random.normal(5, 1, n) 

# Calculate h(x)*f(x)/g(x) for each sample 
weights = 10 * np.exp(-2*np.abs(x_samples-5)) * (1/10) / (1/np.sqrt(2*np.pi) * np.exp(-(x_samples-5)**2/2))

# Estimate the expectation using importance sampling 
I_importance_sampling = np.mean(weights) 
print("Weighted Value:",I_importance_sampling)
# Variance estimation 
variance_importance_sampling = np.var(weights)
print("Variance of weighted:",variance_importance_sampling)

#############################################################################################################
'''
n_values = [100, 1000, 10000, 100000, 1000000]
variances = []


for n in n_values:
    x_values = np.random.uniform(0, 10, n)
    variance_I = np.var(10*np.exp(-2*np.abs(x_values-5)))
    variances.append(variance_I)

plt.plot(n_values, variances, marker='o')
plt.title('Variance of Monte Carlo Integration over Different n Values')
plt.xlabel('n values')
plt.ylabel('Variance_I')
plt.savefig("variance_I.png")
plt.close()



n_values = np.arange(100, 1000001, 1000)
variances = []

for n in n_values:
    x_values = np.random.uniform(0, 10, n)
    variance_I = np.var(10*np.exp(-2*np.abs(x_values-5)))
    variances.append(variance_I)

plt.plot(n_values, variances)
plt.title('Variance of Monte Carlo Integration over Different n Values')
plt.xlabel('n values')
plt.ylabel('Variance_I')
plt.savefig("variance_I2.png")

plt.close()



n_values = np.arange(100, 1000001, 1000)
variances = []

for n in n_values:
    x_samples = np.random.normal(5, 1, n)
    weights = 10 * np.exp(-2*np.abs(x_samples-5)) * (1/10) / (1/np.sqrt(2*np.pi) * np.exp(-(x_samples-5)**2/2))
    variance = np.var(weights)
    variances.append(variance)

plt.plot(n_values, variances)
plt.title('Variance of Importance Sampling over Different n Values')
plt.xlabel('n values')
plt.ylabel('Variance')
plt.savefig("variance_weight.png")

plt.close()
'''
######################################################################################################################################## Code for plotiing both variances ################################################
'''
n_values = np.arange(100, 1000001, 1000)
variances1 = []
variances2 = []

for n in n_values:
  x_values = np.random.uniform(0, 10, n)
  variance_I = np.var(10*np.exp(-2*np.abs(x_values-5)))
  variances1.append(variance_I)

for n in n_values:
  x_samples = np.random.normal(5, 1, n)
  weights = 10 * np.exp(-2*np.abs(x_samples-5)) * (1/10) / (1/np.sqrt(2*np.pi) * np.exp(-(x_samples-5)**2/2))
  variance = np.var(weights)
  variances2.append(variance)

plt.plot(n_values, variances1, label='Monte Carlo')
plt.plot(n_values, variances2, label='Importance Sampling')
plt.title('Comparison of Variance over Different n Values')
plt.ylim(0,5)
plt.xlabel('n values')
plt.ylabel('Variance')
plt.legend()
plt.savefig("variance_total5.png")
'''

############################################################################################################################################  Metropolis Algorithm #######################################################

import numpy as np
from scipy.stats import norm


"""
We want to get an exponential decay integral approx using importance sampling.
We will try to integrate x^2exp(-x^2) over the real line.
Metropolis-hasting alg will generate configuartions (in this case, single numbers) such that 
the probablity of a given configuration x^a ~ p(x^a) for p(x) propto exp(-x^2).

Once configs  = {x^a} generated, the apporximation, Q_N, of the integral, I, will be given by 
Q_N = 1/N sum_(configs) x^2
lim (N-> inf) Q_N -> I
"""

"""
Implementing metropolis-hasting algorithm
"""

# Setting up the initial config for our chain
x_0 = np.random.uniform(-20, -10)

# Defining function that generates the next N steps in the chain, given a starting config x
# Works by iteratively taking the last element in the chain, generating a new candidate configuration from it and accepting/rejecting according to the algorithm
# Success and failures implemented to see roughly the success rate of each step
def next_steps(x, N):
    Success = 0
    Failures = 0
    Data = np.empty((N,))
    d = 1.5  # Spread of (normal) transition function
    for i in range(N):
        r = np.random.uniform(0, 1)
        delta = np.random.normal(0, d)
        x_new = x + delta
        hasting_ratio = np.exp(-(x_new ** 2 - x ** 2))
        if hasting_ratio > r:
            x = x_new
            Success = Success + 1
        else:
            Failures = Failures + 1
        Data[i] = x
    print(Success)
    print(Failures)
    return Data


# Number of steps in the chain
N_iteration = 50000

# Generating the data
Data = next_steps(x_0, N_iteration)

# Obtaining tail end data and obtaining the standard deviation of resulting gaussian distribution
Data = Data[-40000:]
(mu, sigma) = norm.fit(Data)
print(sigma)

######################################################################################################################################### Visualising Metropolis Algorithm ###############################################


from numpy.random import uniform
import math
from typing import Tuple
from typing import Callable
from typing import Iterator
from tqdm import tqdm
'''
x0 = 0


def propose_sample(current_sample: float) -> float:
    return current_sample + uniform() - 0.5


proposed_sample = propose_sample(x0)

def score(x: float, mu: float = 0, sigma: float = 1) -> float:
  norm_x = (x - mu) / sigma
  return math.exp(-(norm_x ** 2) / 2)


def get_next_sample(current_sample: float, current_sample_score: float) -> Tuple[float, float]:
  # Calculate proposed value for x_{t+1}
  proposed_sample = propose_sample(current_sample)
  proposed_sample_score = score(proposed_sample)

  # NOTE: This code was written with simplicity in mind, but there is no reason to
  # sample from the uniform distribution if proposed_sample_score > self.current_sample_score
  if uniform() <= (proposed_sample_score / current_sample_score):
      current_sample = proposed_sample
      current_sample_score = proposed_sample_score

  return current_sample, current_sample_score


# x0 is the arbitrary starting point
x0 = 0
x0_score = score(x0)

x1, x1_score = get_next_sample(x0, x0_score)
x2, x2_score = get_next_sample(x1, x1_score)

class MetropolisHastings:
  def __init__(self, x0: float = 0) -> None:
      self.sample = x0
      self.sample_score = score(self.sample)

  def __call__(self) -> float:
      self.sample, self.sample_score = get_next_sample(self.sample, self.sample_score)
      return self.sample


def gen_samples(f: Callable[[], float]) -> Iterator[float]:
  for _ in tqdm(range(1000000)):
      yield f()

metropolis = MetropolisHastings()
gaussian_samples = list(gen_samples(metropolis))

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

count, bins, ignored = plt.hist(gaussian_samples, 30, density=True)
mu, sigma = norm.fit(gaussian_samples)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2)), linewidth=2, color='r')
plt.savefig('Metrpolis.png')

'''


def propose_sample(current_sample: float) -> float:
    return current_sample + uniform() - 0.5


proposed_sample = propose_sample(x0)

def score(x: float, mu: float = 0, sigma: float = 1) -> float:
  norm_x = (x - mu) / sigma
  #return (0.5*math.exp(-((norm_x-2) ** 2) / 2) + 0.5*math.exp(-((norm_x+2) ** 2) / 2))
  return math.exp(-2*(np.abs(norm_x - 5)))
  #return math.exp(-(norm_x ** 2) / 2)




def get_next_sample(current_sample: float, current_sample_score: float) -> Tuple[float, float]:
  # Calculate proposed value for x_{t+1}
  proposed_sample = propose_sample(current_sample)
  proposed_sample_score = score(proposed_sample)

  # NOTE: This code was written with simplicity in mind, but there is no reason to
  # sample from the uniform distribution if proposed_sample_score > self.current_sample_score
  if uniform() <= (proposed_sample_score / current_sample_score):
      current_sample = proposed_sample
      current_sample_score = proposed_sample_score

  return current_sample, current_sample_score


# x0 is the arbitrary starting point
x0 = 0
x0_score = score(x0)

x1, x1_score = get_next_sample(x0, x0_score)
x2, x2_score = get_next_sample(x1, x1_score)

class MetropolisHastings:
  def __init__(self, x0: float = 0) -> None:
      self.sample = x0
      self.sample_score = score(self.sample)

  def __call__(self) -> float:
      self.sample, self.sample_score = get_next_sample(self.sample, self.sample_score)
      return self.sample


def gen_samples(f: Callable[[], float]) -> Iterator[float]:
  for _ in tqdm(range(1000000)):
      yield f()

metropolis = MetropolisHastings()
gaussian_samples = list(gen_samples(metropolis))

count, bins, ignored = plt.hist(gaussian_samples, 30, density=True)
mu, sigma = norm.fit(gaussian_samples)
#plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2)), linewidth=2, color='r')
#plt.plot(bins, (0.5*math.exp(-((bins-2) ** 2) / 2) + 0.5*math.exp(-((bins+2) ** 2) / 2)), linewidth=2, color='r')
plt.plot(bins, math.exp(-2*(np.abs(bins - 5))), linewidth=2, color='r')
plt.savefig('Metrpolis_actual2.png')


for i in range(4):
  for j in range(4):
    for k in range(4):
      for l in range(4):
        print(i, j, k, l)
