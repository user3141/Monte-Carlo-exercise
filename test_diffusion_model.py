from diffusion_model import energy
from nose.tools import assert_equal, assert_raises, assert_true

def test_energy():
  """ Optional description for nose reporting """
  density = [2,1]
  assert_equal(energy(density), 1.)
  assert_equal(energy(density, coeff=2), 2.)
  assert_raises(energy([]))
  assert_equal(energy([3,4,5]), 19.)
  assert_equal(type(energy([1])), type(1.))
  