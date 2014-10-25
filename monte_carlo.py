import diffusion_model as dm

def move_particle(density):
  """Moves random particle randomly to the left or right."""
  pass
  return density


def accept_higher_energy(E_0, E_1, temp):
  """If random number [0,1] is smaller than Boltzmann factor accept new state."""
  pass
  return

  
def monte_carlo_sim(density, energy_fun, temp, steps=100):
  """Monte Carlo (Metropolisalgorithm) simulation of 1D diffusion.
     density - 1D particle density
     energy_fun - function to calculate energy
     temp - simulation temperature
     steps - number of iterations
  """
  energies = [energy_fun(density)]
  
  for i in xrange(steps):
    E_0 = energy_fun(density)
    new_density = move_particle(density)
    E_1 = energy_fun(new_density)
    if E_1 < E_0 or accept_higher_energy(E_0, E_1, temp):
      density = new_density
      energies.append(E_1)
      
  
if __name__ == "__main__"  :
  pass
      
  