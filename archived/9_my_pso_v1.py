# Import library
import numpy as np
import matplotlib.pyplot as plt

#Sphere function
def Sphere(x):
  z=np.sum(np.square(x), axis=0)
  return z

# Parameter setting
d = 10 # dimension
xMin, xMax = -100, 100  # minimum and maximum of search space
vMin, vMax = -0.2*(xMax - xMin), 0.2*(xMax - xMin) # often 20 percent of xMin and xMax for velocity initialization
MaxIt = 3000 # Maximum iteration
ps = 10 # population size
c1 = 2
c2 = 2
w = 0.9 - ((0.9 - 0.4)/MaxIt)*np.linspace(0, MaxIt, MaxIt) # w should vary from 0.4 to 0.9 based on the number of iterations

def limitV(V):
  for i in range(len(V)):
    if V[i] > vMax:
      V[i] = vMax
    if V[i]<vMin:
      V[i] = vMin
  return V

def limitX(X):
  for i in range(len(X)):
    if X[i] > xMax:
      X[i] = xMax
    if X[i]<xMin:
      X[i] = xMin
  return X

#%% Algorithm
def Optimization():
  class Particle():
    def __init__(self):
      self.position = np.random.uniform(xMin, 50, [ps, d]) # random position
      self.velocity = np.random.uniform(vMin, vMax, [ps, d]) # random velocity

      self.cost = np.zeros(ps) # fitness value 
      self.cost[:] = Sphere(self.position[:]) # fitness value for position of each particle
      # personal best position (we assume that the first position we created is the best personal position so we copy the value of first position)
      self.pbest = np.copy(self.position)
      # personal best fitness value ()
      self.pbest_cost = np.copy(self.cost)
      # now we need to find the value of global best 
      # so we create a variable index(self.index) to find the value of personal best cost. So we return the index number of the minimum value in the personal best cost array
      self.index=np.argmin(self.pbest_cost)
      # and we need this index to define the global best position
      self.gbest = self.pbest[self.index]
      self.gbest_cost = self.pbest_cost[self.index]
      
      # So till here we got the personal best (pbest) and global best (gbest)

      # now we define an array to collect all the data from every iteration (BestCost)
      self.BestCost = np.zeros(MaxIt)

      # Now we got the first value, we can start the first iteration
    def Evaluate(self):
      # so the algorithm start from the 1st iteration. now we need a loop function
      # so we need one loop from 1st iteration to MaxIt and another for iterating from 1st particle to the last one
      for it in range(MaxIt):
        for i in range(ps):
          # now we can update the velocity by using the pso equation and also we will update the position of each particle
          # the formula is to start with the initial weight and velocity
          self.velocity[i] = (w[it]*self.velocity[i]
                              +c1*np.random.rand(d)*(self.pbest[i] - self.position[i])
                              +c2*np.random.rand(d)*(self.gbest[i] - self.position[i]))
          # before updating position we need to check velocity and position are within the same space or not(so we defined a function to control the limit of velocity limitV)
          self.velocity[i] = limitV(self.velocity[i])

          # Now we update position
          self.position[i] = self.position[i] + self.velocity[i]

          self.position[i] = limitX(self.position[i])

          # Now we can compare the fitness value of the particle vs previous value and if it is better than personal best then we compare to global best
          self.cost[i] = Sphere(self.position[i])
          # if fitness of corrent position is smaller than fitness of personal best then update the personal best position and cost
          if self.cost[i] < self.pbest_cost[i]:
            # personal best will equal to current position
            self.pbest[i] = self.position[i]
            # and personal cost will equal to current cost
            self.pbest_cost[i] = self.cost[i]
            # Now we compare the fitness to global best cost
            if self.pbest_cost[i] < self.gbest_cost:
              self.gbest = self.pbest[i]
              self.gbest_cost = self.pbest_cost[i]
        
        # Now we stop collecting the data:
        self.BestCost[it] = self.gbest_cost
    
    # ploting the gbest fitness value for every iteration:
    def Plot(self):
      plt.semilogy(self.BestCost)
      plt.ylim([10e-120, 10e20])
      plt.xlim([0, 3000])
      plt.ylabel('Best Function Value')
      plt.xlabel('Number of Iteration')
      plt.title('Particle Swarm Optimization of Sphere Function')
      print('Best fitness value =', self.gbest_cost)
      print('Best position value =', self.gbest)
  
  a = Particle()
  a.Evaluate()
  a.Plot()

#%%Run
Optimization()