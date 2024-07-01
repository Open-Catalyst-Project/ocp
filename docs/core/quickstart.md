---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

Using pre-trained models in ASE
----------

1. First, install `fairchem` in a fresh python environment using one of the approaches in [installation documentation](install).
2. See what pre-trained potentials are available 

```{code-cell} ipython3
from fairchem.core.models.model_registry import available_pretrained_models
print(available_pretrained_models)
```

3. Choose a checkpoint you want to use and download it automatically! We'll use the GemNet-OC potential, trained on both the OC20 and OC22 datasets.

```{code-cell} ipython3
from fairchem.core.models.model_registry import model_name_to_local_file
checkpoint_path = model_name_to_local_file('GemNet-OC-S2EFS-OC20+OC22', local_cache='/tmp/ocp_checkpoints/')
checkpoint_path
```

4. Finally, use this checkpoint in an ASE calculator for a simple relaxation!

```{code-cell} ipython3
from fairchem.core import FAIRChemCalculator
from ase.build import fcc111, add_adsorbate
from ase.optimize import BFGS
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

# Define the model atomic system, a Pt(111) slab with an *O adsorbate!
slab = fcc111('Pt', size=(2, 2, 5), vacuum=10.0)
add_adsorbate(slab, 'O', height=1.2, position='fcc')

# Load the pre-trained checkpoint!
calc = FAIRChemCalculator(checkpoint_path=checkpoint_path, cpu=False)
slab.set_calculator(calc)

# Run the optimization!
opt = BFGS(slab)
opt.run(fmax=0.05, steps=100)

# Visualize the result!
fig, axs = plt.subplots(1, 2)
plot_atoms(slab, axs[0]);
plot_atoms(slab, axs[1], rotation=('-90x'))
axs[0].set_axis_off()
axs[1].set_axis_off()
```

To learn more about what this simulation means and how it fits into catalysis, see the [catalysis tutorial](../tutorials/intro)!