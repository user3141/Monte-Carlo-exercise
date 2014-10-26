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

  

def test_accept_higher_energy():
  pass
  

@mock.patch('monte_carlo.move_particle')
@mock.patch('monte_carlo.accept_higher_energy')
def test_monte_carlo_sim(mock_move_particle, mock_accept_higher_energy):
  pass
  
