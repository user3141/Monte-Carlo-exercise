import diffusion_model as dm
import random
import math
import numpy as np

def accumulate(lis):
  total = 0
  for x in lis:
    total += x
    yield total

def move_particle(density):
  """Moves random particle randomly to the left or right."""

  while True:
    # choose random particle
    particle = random.randint(1, np.sum(density))
    # find position of particle
    cum_particles = list(accumulate(density))
    pos = [i for i in xrange(len(cum_particles)) if cum_particles[i] >= particle][0]

    # random direction (-1 - left, 1 - right)
    direction = random.choice([-1,1])

    # no periodic boundary conditions
    if pos == 0 and direction == -1:
      continue
    elif pos == len(density) - 1 and direction == 1:
      continue

    # move particle
    new_density = np.array(density)
    new_density[pos] -= 1
    new_density[pos + direction] += 1
    break

  return new_density


def accept_higher_energy(E_0, E_1, temp):
  """If random number [0,1) is smaller than Boltzmann factor accept new state."""
  boltzmann_factor = math.exp(- (E_1 - E_0)/temp)
  return random.random() < boltzmann_factor


def monte_carlo_sim(density, energy_fun, temp=300, steps=50000):
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
  import matplotlib
  import matplotlib.pyplot as plt
  matplotlib.interactive(True)

  density = [0] * 8 + [10] * 6 + [0] * 8
  density = np.array(density)
  #random.shuffle(density)
  temp = 300
  energies, densities = monte_carlo_sim(density, dm.energy, temp, steps=5000)
  plt.plot(energies)
  plt.savefig('monte_carlo.png')

  densities = np.array(densities)
  plt.clf()
  heatmap = plt.pcolor(densities)
  plt.savefig('heatmap.png')


