import diffusion_model as dm
import random
import math

def move_particle(density):
  """Moves random particle randomly to the left or right."""
  # random position with particle(s)
  while True:
    pos = random.randint(0, len(density)-1)
    if density[pos] != 0:
      break
      
  # random direction (-1 - left, 1 - right)
  direction = random.choice([-1,1])
  
  # no periodic boundary conditions
  if pos == 0:
    direction = 1
  elif pos == len(density):
    direction = -1
    
  # move particle
  new_density = density
  new_density[pos] -= 1
  new_density[pos + direction] += 1
  
  return new_density


def accept_higher_energy(E_0, E_1, temp):
  """If random number [0,1) is smaller than Boltzmann factor accept new state."""
  boltzmann_factor = math.exp(- (E_1 - E_0)/temp)
  return random.random() < boltzmann_factor

  
def monte_carlo_sim(density, energy_fun, temp, steps=100):
  """Monte Carlo (Metropolis-Hastings-algorithm) simulation of 1D diffusion.
     density - 1D particle density
     energy_fun - function to calculate energy
     temp - simulation temperature
     steps - number of iterations
  """
  energies = [energy_fun(density)]
  densities = [density]
  
  for i in xrange(steps):
    E_0 = energy_fun(density)
    new_density = move_particle(density)
    E_1 = energy_fun(new_density)
    if E_1 < E_0 or accept_higher_energy(E_0, E_1, temp):
      density = new_density
      energies.append(E_1)
      densities.append(density)
  return energies, densities
      
  
if __name__ == "__main__"  :
  pass
      
  