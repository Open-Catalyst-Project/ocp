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

# Fine tuning a model

In this section we look at how to improve a model. We start by using a pre-trained model to predict the energies of several oxide polymorphs. The data here comes from this paper:

Identifying Potential BO<sub>2</sub> Oxide Polymorphs for Epitaxial Growth Candidates
Mehta, P., Salvador, P. A., & Kitchin, J. R. (2015). Identifying potential BO2 oxide polymorphs for epitaxial growth candidates. ACS Appl. Mater. Interfaces, 6(5), 3630–3639. http://dx.doi.org/10.1021/am4059149.

This data set shows equations of state for several oxide/polymorph combinations. I use this dataset to compare with predictions from OCP

+++

First we get the checkpoint that we want. According to the [MODELS](../../core/models) the GemNet-OC	OC20+OC22 combination has an energy MAE of 0.483 which seems like a good place to start. This model was trained on oxides.

We get this checkpoint here.

```{code-cell} ipython3
from fairchem.core.models.model_registry import model_name_to_local_file

checkpoint_path = model_name_to_local_file('GemNet-OC-S2EFS-OC20+OC22', local_cache='/tmp/ocp_checkpoints/')
```

The data we need is provided in `supporting-information.json`. That file is embedded in the supporting information for the article, and is provided here in the tutorial. We load this data and explore it a little. The json file provides a dictionary with the structure:

    [oxide][polymorph][xc][EOS][configurations]
    
The first key is a composition, the second is a string for the polymorph structure, the third indicates which XC functional was used (we focus on PBE here), the fourth key is for the Equation of State calculations, and the last key is a list of results for the EOS. 

```{code-cell} ipython3
import json
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms

with open('supporting-information.json', 'rb') as f:
     d = json.loads(f.read())
        

oxides = list(d.keys())
polymorphs = list(d['TiO2'].keys())
oxides, polymorphs
```

A specific calculation has a lot of details in it. You can use this to recreate the calculations. Shortly we will use these to compare the DFT results to OCP.

```{code-cell} ipython3
d['TiO2']['rutile']['PBE']['EOS']['calculations'][0]
```

For each result we can retrieve the atomic geometry, energy and forces from the json file. We use these to recreate an Atoms object. Here is an example.

```{code-cell} ipython3
c = d['TiO2']['rutile']['PBE']['EOS']['calculations'][0]
atoms = Atoms(symbols=c['atoms']['symbols'],
              positions=c['atoms']['positions'],
              cell=c['atoms']['cell'],
              pbc=c['atoms']['pbc'])
atoms.set_tags(np.ones(len(atoms)))
atoms, c['data']['total_energy'], c['data']['forces']
```

Next, we will create an OCP calculator that we can use to get predictions from.

```{code-cell} ipython3
from fairchem.core import FAIRChemCalculator
calc = FAIRChemCalculator(checkpoint_path=checkpoint_path, trainer='forces', cpu=False)
```

Now, we loop through each structure and accumulate the OCP predictions. Then, we plot the parity results.

```{code-cell} ipython3
import time
t0 = time.time()

eos_data = {}
for oxide in oxides:
    eos_data[oxide] = {}
    for polymorph in polymorphs:
        dft = []
        ocp = []
        vols = []
        calculations = d[oxide][polymorph]['PBE']['EOS']['calculations']
        for c in calculations:
                        
            atoms = Atoms(symbols=c['atoms']['symbols'],
                          positions=c['atoms']['positions'],
                          cell=c['atoms']['cell'],
                          pbc=c['atoms']['pbc'])
            atoms.set_tags(np.ones(len(atoms)))
            
            atoms.calc = calc
            ocp += [atoms.get_potential_energy() / len(atoms)]
            dft += [c['data']['total_energy'] / len(atoms)]
            vols += [atoms.get_volume()]
        plt.plot(dft, ocp, marker='s' if oxide == 'VO2' else '.',
                 alpha=0.5, label=f'{oxide}-{polymorph}')
        eos_data[oxide][polymorph] = (vols, dft, ocp)
        
plt.xlabel('DFT (eV/atom)')
plt.ylabel('OCP (eV/atom)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=3);
print(f'Elapsed time {time.time() - t0:1.1f} seconds.')
```

```{code-cell} ipython3
mae = np.mean(np.abs(np.array(dft) - np.array(ocp)))
print(f'MAE = {mae:1.3f} eV/atom')
```

The MAE is somewhat high compared to the reported value of 0.458 eV. That is not too surprising; although OC22 was trained on oxides, it was not trained on all of these structures. It is also evident on inspection that the main issues are all the VO2 structures, and these skew the MAE. Next we look at the EOS for each material. Here you can see offsets, and qualitative issues in the shapes. The offset is largely a PBE/RPBE difference.

Some notable issues are with fluorite structures. We show one here for VO2.  

```{code-cell} ipython3
oxide, polymorph = 'VO2', 'fluorite'
V, D, O = eos_data[oxide][polymorph]
        
plt.plot(V, D, label='dft')
plt.plot(V, O, label='ocp')
plt.title(f'pretrained {oxide} - {polymorph}')
plt.legend();
```

We can improve this by fine-tuning the model.

+++

# Fine tuning the checkpoint

There are severeal steps in fine-tuning:
1. Create train, test, val datasets
2. Create a configuration yml file
3. Run the training
4. Check the results

First we create the training databases. OCP supports ase databases, so we use those here. The first step is to make a database containing the data. We need to write an atoms object with an attached SinglePointCalculator for each structure.

```{code-cell} ipython3
from ase.db import connect
from ase.calculators.singlepoint import SinglePointCalculator

! rm -fr oxides.db  # start clean when you run this cell
db = connect('oxides.db')
```

```{code-cell} ipython3
for oxide in oxides:
    for polymorph in polymorphs:
        for c in d[oxide][polymorph]['PBE']['EOS']['calculations']:
            atoms = Atoms(symbols=c['atoms']['symbols'],
                          positions=c['atoms']['positions'],
                          cell=c['atoms']['cell'],
                          pbc=c['atoms']['pbc'])
            atoms.set_tags(np.ones(len(atoms)))
            calc = SinglePointCalculator(atoms, 
                                         energy=c['data']['total_energy'],
                                         forces=c['data']['forces'])
            atoms.set_calculator(calc)
            db.write(atoms)
```

Let's see what we made.

```{code-cell} ipython3
! ase db oxides.db
```

## Make the train, test, val splits

We need to split the ase-db into three separate databases, one for training (80%), one for testing (10%) and one for validation. We generate a list of ids and then shuffle them. Then we write the first 80% into `train.db`, the next 10% into `test.db`, and the remaining into `val.db`.

The train set is used for training. The test and val sets are used to check for overfitting.

You choose the splits you want, 80:10:10 is common. We take a simple approach to split the database here. We make an array of integers that correspond to the ids, randomly shuffle them, and then get each row in the randomized order and write them to a new db.

We provide some helper functions in `fairchem.core.common.tutorial_utils` to streamline this process. 

```{code-cell} ipython3
from fairchem.core.common.tutorial_utils import train_test_val_split
! rm -fr train.db test.db val.db

train, test, val = train_test_val_split('oxides.db')
train, test, val
```

## Setting up the configuration yaml file

We have to create a yaml configuration file for the model we are using. The pre-trained checkpoints contain their config data, so we use this to get the base configuration, and then remove pieces we don't need, and update pieces we do need.

```{code-cell} ipython3
from fairchem.core.common.tutorial_utils import generate_yml_config 

yml = generate_yml_config(checkpoint_path, 'config.yml',
                   delete=['slurm', 'cmd', 'logger', 'task', 'model_attributes',
                           'optim.loss_force', # the checkpoint setting causes an error
                           'dataset', 'test_dataset', 'val_dataset'],
                   update={'gpus': 1,
                           'task.dataset': 'ase_db',
                           'optim.eval_every': 10,
                           'optim.max_epochs': 1,
                           'optim.batch_size': 4,
                            'logger':'tensorboard', # don't use wandb!
                           # Train data
                           'dataset.train.src': 'train.db',
                           'dataset.train.a2g_args.r_energy': True,
                           'dataset.train.a2g_args.r_forces': True,
                            # Test data - prediction only so no regression
                           'dataset.test.src': 'test.db',
                           'dataset.test.a2g_args.r_energy': False,
                           'dataset.test.a2g_args.r_forces': False,
                           # val data
                           'dataset.val.src': 'val.db',
                           'dataset.val.a2g_args.r_energy': True,
                           'dataset.val.a2g_args.r_forces': True,
                          })

yml
```

```{code-cell} ipython3
! cat config.yml
```

## Running the training job

`fairchem` provides a `main.py` file that is used for training. Here we construct the Python command you need to run, and run it. `main.py` is not executable, so we have to run it with python, and you need the absolute path to it, which we get from the `fairchem_main()` that is defined in the fairchem.core.common.tutorial_utils.

you must set a `mode` and provide a `config-yml`. We provide a checkpoint for a starting point, if you don't do this, it will start from scratch. 

By default the files that are created go in a directories called `checkpoints`, `logs` and `results`. You can change that with a `--run-dir preferred-place` option.

You can also add an identifier tag to the end of the timestamps with `--identifier tag`. This can make it easier to find them later.

The cell below uses some IPython magic to put Python variables in the shell command.

This command takes some time to run even on a GPU, e.g. about 30 minutes. 

It is advisable to redirect the outputs to files. The reason is that if the outputs are very large, the notebook may not be able to be saved. This also makes your notebooks more reproducible. The checkpoints are stored in time-stamped directories that change everytime you run them. Below we show how to reproducibly retrieve this directory name in a way that allows you to run the notebook again later, while automatically updating the directory name.

You can follow how the training is going by opening a terminal and running

    tail -f train.txt
    
You can also visit it in a browser at [train.txt](./train.txt). You have to periodically refresh the view to see updates though.

This can take up to 30 minutes for 80 epochs, so we only do a few here to see what happens.

```{code-cell} ipython3
:tags: [hide-output]

import time
from fairchem.core.common.tutorial_utils import fairchem_main

t0 = time.time()
! python {fairchem_main()} --mode train --config-yml {yml} --checkpoint {checkpoint_path} --run-dir fine-tuning --identifier ft-oxides --amp > train.txt 2>&1 
print(f'Elapsed time = {time.time() - t0:1.1f} seconds')
```

Now since we have a file, we can find the training results in it. See [train.txt](./train.txt). At the top, the config is printed, so we can get the checkpoint directory. I use shell commands and Python to get the line, split and strip it here.

```{code-cell} ipython3
cpline = !grep "checkpoint_dir:" train.txt
cpdir = cpline[0].split(':')[-1].strip()
cpdir
```

There will be two files in there: `checkpoint.pt` and `best_checkpoint.pt`. 

The `best_checkpoint.pt` is the one that performs best on the validation dataset. The `checkpoint.pt` is the most recently saved one. Probably it has the lowest loss in training, but this could be an indication of overfitting. You have to use some judgement in determining which one to use, and if it is sufficiently accurate for your needs. It may need additional training to further improve it.

```{code-cell} ipython3
newckpt = cpdir + '/checkpoint.pt'
newcalc = FAIRChemCalculator(checkpoint_path=newckpt, cpu=False)
```

```{code-cell} ipython3
eos_data = {}
for oxide in oxides:
    eos_data[oxide] = {}
    for polymorph in polymorphs:
        dft = []
        ocp = []
        vols = []
        calculations = d[oxide][polymorph]['PBE']['EOS']['calculations']
        for c in calculations:
                        
            atoms = Atoms(symbols=c['atoms']['symbols'],
                          positions=c['atoms']['positions'],
                          cell=c['atoms']['cell'],
                          pbc=c['atoms']['pbc'])
            atoms.set_tags(np.ones(len(atoms)))
            
            atoms.calc = newcalc
            ocp += [atoms.get_potential_energy() / len(atoms)]
            dft += [c['data']['total_energy'] / len(atoms)]
            vols += [atoms.get_volume()]
        plt.plot(dft, ocp, marker='s' if oxide == 'VO2' else '.',
                 alpha=0.5, label=f'{oxide}-{polymorph}')
        eos_data[oxide][polymorph] = (vols, dft, ocp)
        
plt.xlabel('DFT (eV/atom)')
plt.ylabel('OCP (eV/atom)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=3);
```

```{code-cell} ipython3
mae = np.mean(np.abs(np.array(dft) - np.array(ocp)))
print(f'New MAE = {mae:1.3f} eV/atom')
```

This is a substantial improvement! We can see what it means in terms of the Equations of State. There is still not perfect agreement, but the curves are closer together. Additional fine tuning, or a better model could probably still improve this.

```{code-cell} ipython3
oxide, polymorph = 'VO2', 'fluorite'
V, D, O = eos_data[oxide][polymorph]
        
plt.plot(V, D, label='dft')
plt.plot(V, O, label='ocp')
plt.title(f'fine-tuned {oxide} - {polymorph}')
plt.legend();
```

It is possible to continue refining the fit. The simple things to do are to use more epochs of training. Eventually the MAE will stabilize, and then it may be necessary to adjust other optimization parameters like the learning rate (usually you decrease it). 

Depending on what is important to you, you may consider changing the relative importances of energy and forces; you can often trade off accuracy of one for the other. It may be necessary to add additional data to cover the composition and configuration space more thoroughly.

There are also other models you could consider. Newer models tend to be more accurate, but they may also be more expensive to run. These are all compromises you can consider.

+++

