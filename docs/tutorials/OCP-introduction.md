---
jupytext:
  formats: md:myst,ipynb
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

Simple simulations using the OCP ASE calculator
==================================================

To introduce OCP we start with using it to calculate adsorption energies for a simple, atomic adsorbate where we specify the site we want to the adsorption energy for. Conceptually, you do this like you would do it with density functional theory. You create a slab model for the surface, place an adsorbate on it as an initial guess, run a relaxation to get the lowest energy geometry, and then compute the adsorption energy using reference states for the adsorbate.

You do have to be careful in the details though. Some OCP model/checkpoint combinations return a total energy like density functional theory would, but some return an "adsorption energy" directly. You have to know which one you are using. In this example, the model we use returns an "adsorption energy".

+++

# Calculating adsorption energies

Calculating adsorption energy with OCP
Adsorption energies are different than you might be used to in OCP. For example, you may want the adsorption energy of oxygen, and conventionally you would compute that from this reaction:

    1/2 O2 + slab -> slab-O

This is not what is done in OCP. It is referenced to a different reaction

    x CO + (x + y/2 - z) H2 + (z-x) H2O + w/2 N2 + * -> CxHyOzNw*

Here, x=y=w=0, z=1, so the reaction ends up as

    -H2 + H2O + * -> O*

or alternatively,

    H2O + * -> O* + H2

It is possible through thermodynamic cycles to compute other reactions. If we can look up rH1 below and compute rH2

    H2 + 1/2 O2 -> H2O  re1 = -3.03 eV
    H2O + * -> O* + H2  re2  # Get from OCP as a direct calculation

Then, the adsorption energy for

    1/2O2 + * -> O*  

is just re1 + re2.

Based on https://atct.anl.gov/Thermochemical%20Data/version%201.118/species/?species_number=986, the formation energy of water is about -3.03 eV at standard state. You could also compute this using DFT.

The first step is getting a checkpoint for the model we want to use. eSCN is currently the state of the art model [`arXiv`](https://arxiv.org/abs/2302.03655). This next cell will download the checkpoint if you don't have it already. However, we're going to use an older GemNet model (which still works pretty well!) just to keep the resources lower for this tutorial.

The different models have different compute requirements. If you find your kernel is crashing, it probably means you have exceeded the allowed amount of memory. This checkpoint works fine in this example, but it may crash your kernel if you use it in the NRR example.

```{code-cell}
from fairchem.core.models.model_registry import model_name_to_local_file

checkpoint_path = model_name_to_local_file('EquiformerV2-31M-S2EF-OC20-All+MD', local_cache='/tmp/ocp_checkpoints/')
```

Next we load the checkpoint. The output is somewhat verbose, but it can be informative for debugging purposes.

```{code-cell}
from fairchem.core import FAIRChemCalculator
calc = FAIRChemCalculator(checkpoint_path=checkpoint_path, cpu=False)
# calc = FAIRChemCalculator(checkpoint_path=checkpoint_path, cpu=True)
```

Next we can build a slab with an adsorbate on it. Here we use the ASE module to build a Pt slab. We use the experimental lattice constant that is the default. This can introduce some small errors with DFT since the lattice constant can differ by a few percent, and it is common to use DFT lattice constants. In this example, we do not constrain any layers.

```{code-cell}
from ase.build import fcc111, add_adsorbate
from ase.optimize import BFGS
```

```{code-cell}
re1 = -3.03

slab = fcc111('Pt', size=(2, 2, 5), vacuum=10.0)
add_adsorbate(slab, 'O', height=1.2, position='fcc')

slab.set_calculator(calc)
opt = BFGS(slab)
opt.run(fmax=0.05, steps=100)
slab_e = slab.get_potential_energy()
slab_e + re1
```

It is good practice to look at your geometries to make sure they are what you expect.

```{code-cell}
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

fig, axs = plt.subplots(1, 2)
plot_atoms(slab, axs[0]);
plot_atoms(slab, axs[1], rotation=('-90x'))
axs[0].set_axis_off()
axs[1].set_axis_off()
```

How did we do? We need a reference point. In the paper below, there is an atomic adsorption energy for O on Pt(111) of about -4.264 eV. This is for the reaction O + * -> O*. To convert this to the dissociative adsorption energy, we have to add the reaction:

    1/2 O2 -> O   D = 2.58 eV (expt)

to get a comparable energy of about -1.68 eV. There is about ~0.6 eV difference (we predicted -2.3 eV above, and the reference comparison is -1.68 eV) to account for. The biggest difference is likely due to the differences in exchange-correlation functional. The reference data used the PBE functional, and eSCN was trained on *RPBE* data. To additional places where there are differences include:

1. Difference in lattice constant
2. The reference energy used for the experiment references. These can differ by up to 0.5 eV from comparable DFT calculations.
3. How many layers are relaxed in the calculation

Some of these differences tend to be systematic, and you can calibrate and correct these, especially if you can augment these with your own DFT calculations. 

See [convergence study](#Convergence-study) for some additional studies of factors that influence this number.

+++

### Exercises

1. Explore the effect of the lattice constant on the adsorption energy.
2. Try different sites, including the bridge and top sites. Compare the energies, and inspect the resulting geometries.

+++

# Trends in adsorption energies across metals.

Xu, Z., & Kitchin, J. R. (2014). Probing the coverage dependence of site and adsorbate configurational correlations on (111) surfaces of late transition metals. J. Phys. Chem. C, 118(44), 25597–25602. http://dx.doi.org/10.1021/jp508805h

[Supporting information](https://pubs.acs.org/doi/suppl/10.1021/jp508805h/suppl_file/jp508805h_si_001.pdf).

These are atomic adsorption energies:

    O + * -> O*

We have to do some work to get comparable numbers from OCP

    H2 + 1/2 O2 -> H2O  re1 = -3.03 eV
    H2O + * -> O* + H2  re2   # Get from OCP
    O -> 1/2 O2         re3 = -2.58 eV

Then, the adsorption energy for

    O + * -> O*  

is just re1 + re2 + re3.

Here we just look at the fcc site on Pt. First, we get the data stored in the paper.

Next we get the structures and compute their energies. Some subtle points are that we have to account for stoichiometry, and normalize the adsorption energy by the number of oxygens.

+++

First we get a reference energy from the paper (PBE, 0.25 ML O on Pt(111)).

```{code-cell}
import json

with open('energies.json') as f:
    edata = json.load(f)

with open('structures.json') as f:
    sdata = json.load(f)
    
edata['Pt']['O']['fcc']['0.25']
```

Next, we load data from the SI to get the geometry to start from.

```{code-cell}
with open('structures.json') as f:
    s = json.load(f)
    
sfcc = s['Pt']['O']['fcc']['0.25']    
```

Next, we construct the atomic geometry, run the geometry optimization, and compute the energy.

```{code-cell}
re3 = -2.58  # O -> 1/2 O2         re3 = -2.58 eV

from ase import Atoms

atoms = Atoms(sfcc['symbols'],
              positions=sfcc['pos'],
              cell=sfcc['cell'],
              pbc=True)
    
atoms.set_calculator(calc)
opt = BFGS(atoms)

opt.run(fmax=0.05, steps=100)
re2 = atoms.get_potential_energy()
    
nO = 0
for atom in atoms:
    if atom.symbol == 'O':
        nO += 1
        re2 += re1 + re3

print(re2 / nO)
```

## Site correlations

This cell reproduces a portion of a figure in the paper. We compare oxygen adsorption energies in the fcc and hcp sites across metals and coverages. These adsorption energies are highly correlated with each other because the adsorption sites are so similar.

At higher coverages, the agreement is not as good. This is likely because the model is extrapolating and needs to be fine-tuned.

```{code-cell}
from tqdm import tqdm
import time

t0 = time.time()

data = {'fcc': [],
       'hcp': []}

refdata = {'fcc': [],
           'hcp': []}


for metal in ['Cu', 'Ag', 'Pd', 'Pt', 'Rh', 'Ir']:
    print(metal)
    for site in ['fcc', 'hcp']:
        for adsorbate in ['O']:
            for coverage in tqdm(['0.25']):
                 
                
                entry = s[metal][adsorbate][site][coverage]
                
                atoms = Atoms(entry['symbols'],
                              positions=entry['pos'],
                              cell=entry['cell'],
                              pbc=True)
    
                atoms.set_calculator(calc)
                opt = BFGS(atoms, logfile=None)  # no logfile to suppress output

                opt.run(fmax=0.05, steps=100)
                
                re2 = atoms.get_potential_energy()
                nO = 0
                for atom in atoms:
                    if atom.symbol == 'O':
                        nO += 1
                        re2 += re1 + re3
                
                data[site] += [re2 / nO]
                refdata[site] += [edata[metal][adsorbate][site][coverage]]  
                
f'Elapsed time = {time.time() - t0} seconds'            
```

First, we compare the computed data and reference data. There is a systematic difference of about 0.5 eV due to the difference between RPBE and PBE functionals, and other subtle differences like lattice constant differences and reference energy differences. This is pretty typical, and an expected deviation.

```{code-cell}
plt.plot(refdata['fcc'], data['fcc'], 'r.', label='fcc')
plt.plot(refdata['hcp'], data['hcp'], 'b.', label='hcp')
plt.plot([-5.5, -3.5], [-5.5, -3.5], 'k-')
plt.xlabel('Ref. data (DFT)')
plt.ylabel('OCP prediction');
```

Next we compare the correlation between the hcp and fcc sites. Here we see the same trends. The data falls below the parity line because the hcp sites tend to be a little weaker binding than the fcc sites.

```{code-cell}
plt.plot(refdata['hcp'], refdata['fcc'], 'r.')
plt.plot(data['hcp'], data['fcc'], '.')
plt.plot([-6, -1], [-6, -1], 'k-')
plt.xlabel('$H_{ads, hcp}$')
plt.ylabel('$H_{ads, fcc}$')
plt.legend(['DFT (PBE)', 'OCP']);
```

### Exercises

1. You can also explore a few other adsorbates: C, H, N. 
2. Explore the higher coverages. The deviations from the reference data are expected to be higher, but relative differences tend to be better. You probably need fine tuning to improve this performance. This data set doesn't have forces though, so it isn't practical to do it here.

+++

# Next steps

In the next step, we consider some more complex adsorbates in nitrogen reduction, and how we can leverage OCP to automate the search for the most stable adsorbate geometry. See [the next step](./NRR/NRR_example-gemnet).

+++

# Convergence study

In [Calculating adsorption energies](#Calculating-adsorption-energies) we discussed some possible reasons we might see a discrepancy. Here we investigate some factors that impact the computed energies.

In this section, the energies refer to the reaction 1/2 O2 -> O*.

+++

## Effects of number of layers

Slab thickness could be a factor. Here we relax the whole slab, and see by about 4 layers the energy is converged to ~0.02 eV.

```{code-cell}
for nlayers in [3, 4, 5, 6, 7, 8]:
    slab = fcc111('Pt', size=(2, 2, nlayers), vacuum=10.0)
    add_adsorbate(slab, 'O', height=1.2, position='fcc')

    slab.set_calculator(calc)
    opt = BFGS(slab, logfile=None)
    opt.run(fmax=0.05, steps=100)
    slab_e = slab.get_potential_energy()
    print(f'nlayers = {nlayers}: {slab_e + re1:1.2f} eV')
```

## Effects of relaxation

It is common to only relax a few layers, and constrain lower layers to bulk coordinates. We do that here. We only relax the adsorbate and the top layer.

This has a small effect (0.1 eV).

```{code-cell}
from ase.constraints import FixAtoms

for nlayers in [3, 4, 5, 6, 7, 8]:
    slab = fcc111('Pt', size=(2, 2, nlayers), vacuum=10.0)
    add_adsorbate(slab, 'O', height=1.2, position='fcc')
    
    slab.set_constraint(FixAtoms(mask=[atom.tag > 1 for atom in slab]))

    slab.set_calculator(calc)
    opt = BFGS(slab, logfile=None)
    opt.run(fmax=0.05, steps=100)
    slab_e = slab.get_potential_energy()
    print(f'nlayers = {nlayers}: {slab_e + re1:1.2f} eV')
```

## Unit cell size

Coverage effects are quite noticeable with oxygen. Here we consider larger unit cells. This effect is large, and the results don't look right, usually adsorption energies get more favorable at lower coverage, not less. This suggests fine-tuning could be important even at low coverages.

```{code-cell}
for size in [1, 2, 3, 4, 5]:
    slab = fcc111('Pt', size=(size, size, 5), vacuum=10.0)
    add_adsorbate(slab, 'O', height=1.2, position='fcc')
    
    slab.set_constraint(FixAtoms(mask=[atom.tag > 1 for atom in slab]))

    slab.set_calculator(calc)
    opt = BFGS(slab, logfile=None)
    opt.run(fmax=0.05, steps=100)
    slab_e = slab.get_potential_energy()
    print(f'({size}x{size}): {slab_e + re1:1.2f} eV')
```

## Summary

As with DFT, you should take care to see how these kinds of decisions affect your results, and determine if they would change any interpretations or not.
