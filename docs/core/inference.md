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

Fast batched inference
------------------

The ASE calculator is not necessarily the most efficient way to run a lot of computations. It is better to do a "mass inference" using a command line utility. We illustrate how to do that here.

In this paper we computed about 10K different gold structures:

Boes, J. R., Groenenboom, M. C., Keith, J. A., & Kitchin, J. R. (2016). Neural network and Reaxff comparison for Au properties. Int. J. Quantum Chem., 116(13), 979–987. http://dx.doi.org/10.1002/qua.25115

You can retrieve the dataset below. In this notebook we learn how to do "mass inference" without an ASE calculator. You do this by creating a config.yml file, and running the `main.py` command line utility.

```{code-cell} ipython3
! wget https://figshare.com/ndownloader/files/11948267 -O data.db
```



Inference on this file will be fast if we have a gpu, but if we don't this could take a while. To keep things fast for the automated builds, we'll just select the first 10 structures so it's still approachable with just a CPU.
Comment or skip this block to use the whole dataset!

```{code-cell} ipython3
! mv data.db full_data.db

import ase.db
import numpy as np

with ase.db.connect('full_data.db') as full_db:
  with ase.db.connect('data.db',append=False) as subset_db:

    # Select 50 random points for the subset, ASE DB ids start at 1
    for i in np.random.choice(list(range(1,len(full_db)+1)),size=50,replace=False):
      atoms = full_db.get_atoms(f'id={i}', add_additional_information=True)

      if 'tag' in atoms.info['key_value_pairs']:
        atoms.info['key_value_pairs']['tag'] = int(atoms.info['key_value_pairs']['tag'])

      subset_db.write(atoms, **atoms.info['key_value_pairs'])
```

```{code-cell} ipython3
! ase db data.db
```

You have to choose a checkpoint to start with. The newer checkpoints may require too much memory for this environment.

```{code-cell} ipython3
from fairchem.core.models.model_registry import available_pretrained_models
print(available_pretrained_models)
```

```{code-cell} ipython3
from fairchem.core.models.model_registry import model_name_to_local_file

checkpoint_path = model_name_to_local_file('GemNet-dT-S2EFS-OC22', local_cache='/tmp/ocp_checkpoints/')
checkpoint_path

```

We have to update our configuration yml file with the test dataset.

```{code-cell} ipython3
from fairchem.core.common.tutorial_utils import generate_yml_config
yml = generate_yml_config(checkpoint_path, 'config.yml',
                   delete=['cmd', 'logger', 'task', 'model_attributes',
                           'dataset', 'slurm'],
                   update={'amp': True,
                           'gpus': 1,
                           'task.dataset': 'ase_db',
                           'task.prediction_dtype': 'float32',
                           'logger':'tensorboard', # don't use wandb!
                            # Test data - prediction only so no regression
                           'dataset.test.src': 'data.db',
                           'dataset.test.a2g_args.r_energy': False,
                           'dataset.test.a2g_args.r_forces': False,
                           'dataset.test.select_args.selection': 'natoms>5,xc=PBE',
                          })

yml
```

It is a good idea to redirect the output to a file. If the output gets too large here, the notebook may fail to save. Normally I would use a redirect like `2&>1`, but this does not work with the main.py method. An alternative here is to open a terminal and run it there.

```{code-cell} ipython3
%%capture inference
import time
from fairchem.core.common.tutorial_utils import fairchem_main

t0 = time.time()
! python {fairchem_main()} --mode predict --config-yml {yml} --checkpoint {checkpoint_path} --amp
print(f'Elapsed time = {time.time() - t0:1.1f} seconds')
```

```{code-cell} ipython3
with open('mass-inference.txt', 'wb') as f:
    f.write(inference.stdout.encode('utf-8'))
```

```{code-cell} ipython3
! grep "Total time taken:" 'mass-inference.txt'
```

The mass inference approach takes 1-2 minutes to run. See the output [here](./mass-inference.txt).

```{code-cell} ipython3
results = ! grep "  results_dir:" mass-inference.txt
d = results[0].split(':')[-1].strip()
```

```{code-cell} ipython3
import numpy as np
results = np.load(f'{d}/ocp_predictions.npz', allow_pickle=True)
results.files
```

It is not obvious, but the data from mass inference is not in the same order. We have to get an id from the mass inference, and then "resort" the results so they are in the same order.

```{code-cell} ipython3
inds = np.array([int(r.split('_')[0]) for r in results['ids']])
sind = np.argsort(inds)
inds[sind]
```

To compare this with the results, we need to get the energy data from the ase db.

```{code-cell} ipython3
from ase.db import connect
db = connect('data.db')

energies = np.array([row.energy for row in db.select('natoms>5,xc=PBE')])
natoms = np.array([row.natoms for row in db.select('natoms>5,xc=PBE')])
```

Now, we can see the predictions. They are only ok here; that is not surprising, the data set has lots of Au configurations that have never been seen by this model. Fine-tuning would certainly help improve this.

```{code-cell} ipython3
import matplotlib.pyplot as plt

plt.plot(energies / natoms, results['energy'][sind] / natoms, 'b.')
plt.xlabel('DFT')
plt.ylabel('OCP');
```

# The ASE calculator way

We include this here just to show that:

1. We get the same results
2. That this is much slower.

```{code-cell} ipython3
from fairchem.core import FAIRChemCalculator
calc = FAIRChemCalculator(checkpoint_path=checkpoint_path, cpu=False)
```

```{code-cell} ipython3
import time
from tqdm import tqdm
t0 = time.time()
OCP, DFT = [], []
for row in tqdm(db.select('natoms>5,xc=PBE')):
    atoms = row.toatoms()
    atoms.set_calculator(calc)
    DFT += [row.energy / len(atoms)]
    OCP += [atoms.get_potential_energy() / len(atoms)]
print(f'Elapsed time {time.time() - t0:1.1} seconds')
```

This takes at least twice as long as the mass-inference approach above. It is conceptually simpler though, and does not require resorting.

```{code-cell} ipython3
plt.plot(DFT, OCP, 'b.')
plt.xlabel('DFT (eV/atom)')
plt.ylabel('OCP (eV/atom)');
```

# Comparing ASE calculator and main.py

The results should be the same.

It is worth noting the default precision of predictions is float16 with main.py, but with the ASE calculator the default precision is float32. Supposedly you can specify `--task.prediction_dtype=float32` at the command line to or specify it in the config.yml like we do above, but as of the tutorial this does not resolve the issue.

As noted above (see also [Issue 542](https://github.com/FAIR-Chem/fairchem/issues/542)), the ASE calculator and main.py use different precisions by default, which can lead to small differences.

```{code-cell} ipython3
np.mean(np.abs(results['energy'][sind] - OCP * natoms))  # MAE
```

```{code-cell} ipython3
np.min(results['energy'][sind] - OCP * natoms), np.max(results['energy'][sind] - OCP * natoms)
```

```{code-cell} ipython3
plt.hist(results['energy'][sind] - OCP * natoms, bins=20);
```

Here we see many of the differences are very small. 0.0078125 = 1 / 128, and these errors strongly suggest some kind of mixed precision is responsible for these differences. It is an open issue to remove them and identify where the cause is.

```{code-cell} ipython3
(results['energy'][sind] - OCP * natoms)[0:400]
```
