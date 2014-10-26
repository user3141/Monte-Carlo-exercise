import monte_carlo
from nose.tools import assert_equal, assert_raises, assert_true, assert_false
import mock
import numpy as np

def test_move_particle():
  density = np.array([0,0,0,1,0,0])
  new_density = monte_carlo.move_particle(density)
  np.testing.assert_allclose(density, new_density)
  assert_equal(np.sum(density), np.sum(new_density))
  assert_false(np.any(density < 0))
  
  density = np.array([1,0,0,0,0,0])
  new_density = monte_carlo.move_particle(density)
  np.testing.assert_allclose(new_density, np.array([0,1,0,0,0,0]))

  
@mock.patch('monte_carlo.random.random')
def test_accept_higher_energy(mock_random):
  mock_random.return_value = 0.35
  assert_true(monte_carlo.accept_higher_energy(E_0=10, E_1=18, temp=300))
  mock_random.return_value = 0.37
  assert_false(monte_carlo.accept_higher_energy(E_0=10, E_1=18, temp=300))
  

@mock.patch('monte_carlo.move_particle')
@mock.patch('monte_carlo.accept_higher_energy')
def test_monte_carlo_sim(mock_accept_higher_energy, mock_move_particle):

  density = [0,0,1,0]
  mock_move_particle.return_value = [0,1,0,0]
  temp=300

  # equal energy is accepted
  mock_accept_higher_energy.return_value = True
  energies, densities = monte_carlo.monte_carlo_sim(density, sum, temp, steps=1)
  assert_equal(energies, [1,1])
  assert_equal(densities, [[0, 0, 1, 0], [0, 1, 0, 0]])
  
  # equal energy is not accepted
  mock_accept_higher_energy.return_value = False
  energies, densities = monte_carlo.monte_carlo_sim(density, sum, temp, steps=1)
  assert_equal(energies, [1])
  assert_equal(densities, [[0, 0, 1, 0]])

if __name__ == "__main__":
  test_monte_carlo_sim()