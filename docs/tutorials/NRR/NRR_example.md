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

Using OCP to enumerate adsorbates on alloy catalyst surfaces
======================================================



In the previous example, we constructed slab models of adsorbates on desired sites. Here we leverage code to automate this process. The goal in this section is to generate candidate structures, compute energetics, and then filter out the most relevant ones.


```{code-cell} ipython3
from fairchem.core import FAIRChemCalculator
import ase.io
from ase.optimize import BFGS
import sys
from scipy.stats import linregress
import pickle
import matplotlib.pyplot as plt
import time

from fairchem.data.oc.core import Adsorbate, AdsorbateSlabConfig, Bulk, Slab
import os
from glob import glob
import pandas as pd
from fairchem.data.oc.utils import DetectTrajAnomaly
```

```{code-cell} ipython3
from fairchem.core.models.model_registry import model_name_to_local_file

checkpoint_path = model_name_to_local_file('EquiformerV2-31M-S2EF-OC20-All+MD', local_cache='/tmp/ocp_checkpoints/')
checkpoint_path
```

# Introduction

We will reproduce Fig 6b from the following paper: Zhou, Jing, et al. "Enhanced Catalytic Activity of Bimetallic Ordered Catalysts for Nitrogen Reduction Reaction by Perturbation of Scaling Relations." ACS Catalysis 134 (2023): 2190-2201 (https://doi.org/10.1021/acscatal.2c05877).

The gist of this figure is a correlation between H* and NNH* adsorbates across many different alloy surfaces. Then, they identify a dividing line between these that separates surfaces known for HER and those known for NRR.

To do this, we will enumerate adsorbate-slab configurations and run ML relaxations on them to find the lowest energy configuration. We will assess parity between the model predicted values and those reported in the paper. Finally we will make the figure and assess separability of the NRR favored and HER favored domains.

+++

# Enumerate the adsorbate-slab configurations to run relaxations on

+++

Be sure to set the path in `fairchem/data/oc/configs/paths.py` to point to the correct place or pass the paths as an argument. The database pickles can be found in `fairchem/data/oc/databases/pkls`. We will show one explicitly here as an example and then run all of them in an automated fashion for brevity.

```{code-cell} ipython3
import fairchem.data.oc
from pathlib import Path
db = Path(fairchem.data.oc.__file__).parent / Path('databases/pkls/adsorbates.pkl')
db
```

## Work out a single example

We load one bulk id, create a bulk reference structure from it, then generate the surfaces we want to compute.

```{code-cell} ipython3
bulk_src_id = "oqmd-343039"
adsorbate_smiles_nnh = "*N*NH"
adsorbate_smiles_h = "*H"

bulk = Bulk(bulk_src_id_from_db = bulk_src_id, bulk_db_path="NRR_example_bulks.pkl")
adsorbate_H = Adsorbate(adsorbate_smiles_from_db = adsorbate_smiles_h, adsorbate_db_path=db)
adsorbate_NNH = Adsorbate(adsorbate_smiles_from_db = adsorbate_smiles_nnh, adsorbate_db_path=db)
slab = Slab.from_bulk_get_specific_millers(bulk= bulk, specific_millers = (1, 1, 1))
slab
```

We now need to generate potential placements. We use two kinds of guesses, a heuristic and a random approach. This cell generates 13 potential adsorption geometries.

```{code-cell} ipython3
# Perform heuristic placements
heuristic_adslabs = AdsorbateSlabConfig(slab[0], adsorbate_H, mode="heuristic")

# Perform random placements
# (for AdsorbML we use `num_sites = 100` but we will use 4 for brevity here)
random_adslabs = AdsorbateSlabConfig(slab[0], adsorbate_H, mode="random_site_heuristic_placement", num_sites=4)

adslabs = [*heuristic_adslabs.atoms_list, *random_adslabs.atoms_list]
len(adslabs)
```

Let's see what we are looking at. It is a little tricky to see the tiny H atom in these figures, but with some inspection you can see there are ontop, bridge, and hollow sites in different places. This is not an exhaustive search; you can increase the number of random placements to check more possibilities. The main idea here is to *increase* the probability you find the most relevant sites.

```{code-cell} ipython3
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

fig, axs = plt.subplots(4, 4)

for i, slab in enumerate(adslabs):
    plot_atoms(slab, axs[i % 4, i // 4]);
    axs[i % 4, i // 4].set_axis_off()
    
for i in range(16):
    axs[i % 4, i // 4].set_axis_off()

plt.tight_layout()   
```

### Run an ML relaxation

We will use an ASE compatible calculator to run these.

+++

Running the model with BFGS prints at each relaxation step which is a lot to print. So we will just run one to demonstrate what happens on each iteration.

```{code-cell} ipython3
os.makedirs(f"data/{bulk_src_id}_{adsorbate_smiles_h}", exist_ok=True)

# Define the calculator
calc = FAIRChemCalculator(checkpoint_path=checkpoint_path, cpu=False)   # if you have a GPU
# calc = FAIRChemCalculator(checkpoint_path=checkpoint_path, cpu=True)  # If you have CPU only
```

Now we setup and run the relaxation.

```{code-cell} ipython3
t0 = time.time()
os.makedirs(f"data/{bulk_src_id}_H", exist_ok=True)
adslab = adslabs[0]
adslab.calc = calc
opt = BFGS(adslab, trajectory=f"data/{bulk_src_id}_H/test.traj")
opt.run(fmax=0.05, steps=100)

print(f'Elapsed time {time.time() - t0:1.1f} seconds')
```

With a GPU this runs pretty quickly. It is much slower on a CPU.

+++

# Run all the systems

In principle you can run all the systems now. It takes about an hour though, and we leave that for a later exercise if you want. For now we will run the first two, and for later analysis we provide a results file of all the runs. Let's read in our reference file and take a look at what is in it.

```{code-cell} ipython3
with open("NRR_example_bulks.pkl", "rb") as f:
    bulks = pickle.load(f)
    
bulks
```

We have 19 bulk materials we will consider. Next we extract the `src-id` for each one.

```{code-cell} ipython3
bulk_ids = [row['src_id'] for row in bulks]
```

In theory you would run all of these, but it takes about an hour with a GPU. We provide the relaxation logs and trajectories in the repo for the next step.

These steps are embarrassingly parallel, and can be launched that way to speed things up. The only thing you need to watch is that you don't exceed the available RAM, which will cause the Jupyter kernel to crash.

+++

The goal here is to relax each candidate adsorption geometry and save the results in a trajectory file we will analyze later. Each trajectory file will have the geometry and final energy of the relaxed structure. 

It is somewhat time consuming to run this, so in this cell we only run one example, and just the first 4 configurations for each adsorbate. 

```{code-cell} ipython3
import time
from tqdm import tqdm
tinit = time.time()

# Note we're just doing the first bulk_id! 
for bulk_src_id in tqdm(bulk_ids[:1]): 
    # Enumerate slabs and establish adsorbates
    bulk = Bulk(bulk_src_id_from_db=bulk_src_id, bulk_db_path="NRR_example_bulks.pkl")
    slab = Slab.from_bulk_get_specific_millers(bulk= bulk, specific_millers=(1, 1, 1))

    # Perform heuristic placements
    heuristic_adslabs_H = AdsorbateSlabConfig(slab[0], adsorbate_H, mode="heuristic")
    heuristic_adslabs_NNH = AdsorbateSlabConfig(slab[0], adsorbate_NNH, mode="heuristic")

    #Run relaxations
    os.makedirs(f"data/{bulk_src_id}_H", exist_ok=True)
    os.makedirs(f"data/{bulk_src_id}_NNH", exist_ok=True)

    print(f'{len(heuristic_adslabs_H.atoms_list)} H slabs to compute for {bulk_src_id}')
    print(f'{len(heuristic_adslabs_NNH.atoms_list)} NNH slabs to compute for {bulk_src_id}')

    # Set up the calculator, note we're doing just the first 4 configs to keep this fast for the online documentation!
    for idx, adslab in enumerate(heuristic_adslabs_H.atoms_list[:4]):
        t0 = time.time()
        adslab.calc = calc
        print(f'Running data/{bulk_src_id}_H/{idx}')
        opt = BFGS(adslab, trajectory=f"data/{bulk_src_id}_H/{idx}.traj", logfile=f"data/{bulk_src_id}_H/{idx}.log")
        opt.run(fmax=0.05, steps=20)
        print(f'  Elapsed time: {time.time() - t0:1.1f} seconds for data/{bulk_src_id}_H/{idx}')
        
    # Set up the calculator, note we're doing just the first 4 configs to keep this fast for the online documentation!
    for idx, adslab in enumerate(heuristic_adslabs_NNH.atoms_list[:4]):
        t0 = time.time()
        adslab.calc = calc
        print(f'Running data/{bulk_src_id}_NNH/{idx}')
        opt = BFGS(adslab, trajectory=f"data/{bulk_src_id}_NNH/{idx}.traj", logfile=f"data/{bulk_src_id}_NNH/{idx}.log")
        opt.run(fmax=0.05, steps=50)
        print(f'  Elapsed time: {time.time() - t0:1.1f} seconds for data/{bulk_src_id}_NNH/{idx}')

print(f'Elapsed time: {time.time() - tinit:1.1f} seconds')
```

This cell runs all the examples. I don't recommend you run this during the workshop. Instead, we have saved the results for the subsequent analyses so you can skip this one.

```{code-cell} ipython3
---
tags: ["skip-execution"]
---

import time
from tqdm import tqdm
tinit = time.time()

for bulk_src_id in tqdm(bulk_ids): 
    # Enumerate slabs and establish adsorbates
    bulk = Bulk(bulk_src_id_from_db=bulk_src_id, bulk_db_path="NRR_example_bulks.pkl")
    slab = Slab.from_bulk_get_specific_millers(bulk= bulk, specific_millers=(1, 1, 1))

    # Perform heuristic placements
    heuristic_adslabs_H = AdsorbateSlabConfig(slab[0], adsorbate_H, mode="heuristic")
    heuristic_adslabs_NNH = AdsorbateSlabConfig(slab[0], adsorbate_NNH, mode="heuristic")

    #Run relaxations
    os.makedirs(f"data/{bulk_src_id}_H", exist_ok=True)
    os.makedirs(f"data/{bulk_src_id}_NNH", exist_ok=True)

    print(f'{len(heuristic_adslabs_H.atoms_list)} H slabs to compute for {bulk_src_id}')
    print(f'{len(heuristic_adslabs_NNH.atoms_list)} NNH slabs to compute for {bulk_src_id}')

    # Set up the calculator
    for idx, adslab in enumerate(heuristic_adslabs_H.atoms_list):
        t0 = time.time()
        adslab.calc = calc
        print(f'Running data/{bulk_src_id}_H/{idx}')
        opt = BFGS(adslab, trajectory=f"data/{bulk_src_id}_H/{idx}.traj", logfile=f"data/{bulk_src_id}_H/{idx}.log")
        opt.run(fmax=0.05, steps=20)
        print(f'  Elapsed time: {time.time() - t0:1.1f} seconds for data/{bulk_src_id}_H/{idx}')
        
    for idx, adslab in enumerate(heuristic_adslabs_NNH.atoms_list):
        t0 = time.time()
        adslab.calc = calc
        print(f'Running data/{bulk_src_id}_NNH/{idx}')
        opt = BFGS(adslab, trajectory=f"data/{bulk_src_id}_NNH/{idx}.traj", logfile=f"data/{bulk_src_id}_NNH/{idx}.log")
        opt.run(fmax=0.05, steps=50)
        print(f'  Elapsed time: {time.time() - t0:1.1f} seconds for data/{bulk_src_id}_NNH/{idx}')

print(f'Elapsed time: {time.time() - tinit:1.1f} seconds')
```

# Parse the trajectories and post-process

As a post-processing step we check to see if:

1. the adsorbate desorbed
2. the adsorbate disassociated
3. the adsorbate intercalated
4. the surface has changed

We check these because they affect our referencing scheme and may result in energies that don't mean what we think, e.g. they aren't just adsorption, but include contributions from other things like desorption, dissociation or reconstruction. For (4), the relaxed surface should really be supplied as well. It will be necessary when correcting the SP / RX energies later. Since we don't have it here, we will ommit supplying it, and the detector will instead compare the initial and final slab from the adsorbate-slab relaxation trajectory. If a relaxed slab is provided, the detector will compare it and the slab after the adsorbate-slab relaxation. The latter is more correct!

In this loop we find the most stable (most negative) adsorption energy for each adsorbate on each surface and save them in a DataFrame.

```{code-cell} ipython3
# Iterate over trajs to extract results
min_E = []
for file_outer in glob("data/*"):
    ads = file_outer.split("_")[1]
    bulk = file_outer.split("/")[1].split("_")[0]
    results = []
    for file in glob(f"{file_outer}/*.traj"):
        rx_id = file.split("/")[-1].split(".")[0]
        traj = ase.io.read(file, ":")

        # Check to see if the trajectory is anomolous
        detector = DetectTrajAnomaly(traj[0], traj[-1], traj[0].get_tags())
        anom = (
            detector.is_adsorbate_dissociated()
            or detector.is_adsorbate_desorbed()
            or detector.has_surface_changed()
            or detector.is_adsorbate_intercalated()
        )
        rx_energy = traj[-1].get_potential_energy()
        results.append({"relaxation_idx": rx_id, "relaxed_atoms": traj[-1],
                        "relaxed_energy_ml": rx_energy, "anomolous": anom})
    df = pd.DataFrame(results)

    df = df[~df.anomolous].copy().reset_index()
    min_e = min(df.relaxed_energy_ml.tolist())
    min_E.append({"adsorbate":ads, "bulk_id":bulk, "min_E_ml": min_e})

df = pd.DataFrame(min_E)
df_h = df[df.adsorbate == "H"]
df_nnh = df[df.adsorbate == "NNH"]
df_flat = df_h.merge(df_nnh, on = "bulk_id")
```

# Make parity plots for values obtained by ML v. reported in the paper

```{code-cell} ipython3
# Add literature data to the dataframe
with open("literature_data.pkl", "rb") as f:
    literature_data = pickle.load(f)
df_all = df_flat.merge(pd.DataFrame(literature_data), on = "bulk_id")
```

```{code-cell} ipython3
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
f.set_figheight(15)
x = df_all.min_E_ml_x.tolist()
y = df_all.E_lit_H.tolist()
ax1.set_title("*H parity")
ax1.plot([-3.5, 2], [-3.5, 2], "k-", linewidth=3)
slope, intercept, r, p, se = linregress(x, y)
ax1.plot(
    [-3.5, 2],
    [
        -3.5 * slope + intercept,
        2 * slope + intercept,
    ],
    "k--",
    linewidth=2,
)

ax1.legend(
    [
        "y = x",
        f"y = {slope:1.2f} x + {intercept:1.2f}, R-sq = {r**2:1.2f}",
    ],
    loc="upper left",
)
ax1.scatter(x, y)
ax1.axis("square")
ax1.set_xlim([-3.5, 2])
ax1.set_ylim([-3.5, 2])
ax1.set_xlabel("dE predicted OCP [eV]")
ax1.set_ylabel("dE NRR paper [eV]");


x = df_all.min_E_ml_y.tolist()
y = df_all.E_lit_NNH.tolist()
ax2.set_title("*N*NH parity")
ax2.plot([-3.5, 2], [-3.5, 2], "k-", linewidth=3)
slope, intercept, r, p, se = linregress(x, y)
ax2.plot(
    [-3.5, 2],
    [
        -3.5 * slope + intercept,
        2 * slope + intercept,
    ],
    "k--",
    linewidth=2,
)

ax2.legend(
    [
        "y = x",
        f"y = {slope:1.2f} x + {intercept:1.2f}, R-sq = {r**2:1.2f}",
    ],
    loc="upper left",
)
ax2.scatter(x, y)
ax2.axis("square")
ax2.set_xlim([-3.5, 2])
ax2.set_ylim([-3.5, 2])
ax2.set_xlabel("dE predicted OCP [eV]")
ax2.set_ylabel("dE NRR paper [eV]");
f.set_figwidth(15)
f.set_figheight(7)
```

# Make figure 6b and compare to literature results

```{code-cell} ipython3
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
x = df_all[df_all.reaction == "HER"].min_E_ml_y.tolist()
y = df_all[df_all.reaction == "HER"].min_E_ml_x.tolist()
comp = df_all[df_all.reaction == "HER"].composition.tolist()

ax1.scatter(x, y,c= "r", label = "HER")
for i, txt in enumerate(comp):
    ax1.annotate(txt, (x[i], y[i]))

x = df_all[df_all.reaction == "NRR"].min_E_ml_y.tolist()
y = df_all[df_all.reaction == "NRR"].min_E_ml_x.tolist()
comp = df_all[df_all.reaction == "NRR"].composition.tolist()
ax1.scatter(x, y,c= "b", label = "NRR")
for i, txt in enumerate(comp):
    ax1.annotate(txt, (x[i], y[i]))


ax1.legend()
ax1.set_xlabel("dE *N*NH predicted OCP [eV]")
ax1.set_ylabel("dE *H predicted OCP [eV]")


x = df_all[df_all.reaction == "HER"].E_lit_NNH.tolist()
y = df_all[df_all.reaction == "HER"].E_lit_H.tolist()
comp = df_all[df_all.reaction == "HER"].composition.tolist()

ax2.scatter(x, y,c= "r", label = "HER")
for i, txt in enumerate(comp):
    ax2.annotate(txt, (x[i], y[i]))

x = df_all[df_all.reaction == "NRR"].E_lit_NNH.tolist()
y = df_all[df_all.reaction == "NRR"].E_lit_H.tolist()
comp = df_all[df_all.reaction == "NRR"].composition.tolist()
ax2.scatter(x, y,c= "b", label = "NRR")
for i, txt in enumerate(comp):
    ax2.annotate(txt, (x[i], y[i]))

ax2.legend()
ax2.set_xlabel("dE *N*NH literature [eV]")
ax2.set_ylabel("dE *H literature [eV]")
f.set_figwidth(15)
f.set_figheight(7)
```
