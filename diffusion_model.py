def energy(density, coeff=1.0):
  """ Energy associated with the diffusion model

      Parameters
      ----------

      density: array of positive integers
          Number of particles at each position i in the array
      coeff: float
          Diffusion coefficient.
  """
  from numpy import array, any, sum
  
  # Make sure input is an array
  density = array(density)
  # of the right kind (integer). Unless it is zero length, in which case type does not matter.
  if density.dtype.kind != 'i' and len(density) > 0:
    raise TypeError("Density should be an array of *integers*.")
  # and the right values (positive or null)
  if any(density < 0):
    raise ValueError("Density should be an array of *positive* integers.")
  if density.ndim != 1:
    raise ValueError("Density should be an a *1-dimensional* array of positive integers.")
    
  return coeff * 0.5 * sum(density * (density - 1))
  

if __name__ == "__main__":
  energy([1,2,3,4,5])