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

Fine-tuning with Python
----------------------------

The recommended way to do training is with the `main.py` script in ocp. One of the reasons for that is training often takes a long time and is better suited for queue systems like slurm. However, you can submit Python scripts too, and it is possible to run notebooks in Slurm too. Here we work out a proof of concept in training from Python and a Jupyter notebook.

```{code-cell} ipython3
import logging
from fairchem.core.common.utils import SeverityLevelBetween

root = logging.getLogger()


root.setLevel(logging.INFO)

log_formatter = logging.Formatter(
            "%(asctime)s (%(levelname)s): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",)

# Send INFO to stdout
handler_out = logging.FileHandler('out.txt', 'w')
handler_out.addFilter(
            SeverityLevelBetween(logging.INFO, logging.WARNING)
        )
handler_out.setFormatter(log_formatter)
root.addHandler(handler_out)

# Send WARNING (and higher) to stderr
handler_err = logging.FileHandler('out.txt', 'w+')
handler_err.setLevel(logging.WARNING)
handler_err.setFormatter(log_formatter)
root.addHandler(handler_err)
```

```{code-cell} ipython3
! ase db ../../core/fine-tuning/oxides.db
```

```{code-cell} ipython3
from fairchem.core.models.model_registry import model_name_to_local_file

checkpoint_path = model_name_to_local_file('GemNet-OC-S2EFS-OC20+OC22', local_cache='/tmp/ocp_checkpoints/')
from fairchem.core import FAIRChemCalculator
calc = FAIRChemCalculator(checkpoint_path=checkpoint_path, trainer='forces', cpu=False)
```

## Split the data into train, test, val sets

```{code-cell} ipython3
! rm -fr train.db test.db val.db

from fairchem.core.common.tutorial_utils import train_test_val_split

train, test, val = train_test_val_split('../../core/fine-tuning/oxides.db')
train, test, val
```

# Setup the training code

We start by making the config.yml. We build this from the calculator checkpoint.

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
                           'logger': 'tensorboard', # don't use wandb unless you already are logged in
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

## Setup the training task

This essentially allows several opportunities to define and override the config. You start with the base config.yml, and then via "command-line" arguments you specify changes you want to make.

The code is build around `submitit`, which is often used with Slurm, but also works locally.

+++

We have to mimic the `main.py` setup to get the arguments and config setup. Here is a minimal way to do this.

```{code-cell} ipython3
from fairchem.core.common.flags import flags
parser = flags.get_parser()
args, args_override = parser.parse_known_args(["--mode=train",
                                               "--config-yml=config.yml",
                                               f"--checkpoint={checkpoint_path}",
                                               "--amp"])
args, args_override
```

Next, we build the first stage in our config. This starts with the file config.yml, then updates it with the args

```{code-cell} ipython3
from fairchem.core.common.utils import build_config, new_trainer_context

config = build_config(args=args, args_override={})
config
```

# Run the training task

It is still annoying that if your output is too large the notebook will not be able to be saved. On the other hand, it is annoying to simply capture the output.

We are able to redirect most logging to a file above, but not all of it. The link below will open the file in a browser, and the subsequent cell captures all residual output. We do not need any of that, so it is ultimately discarded.

Alternatively, you can open a Terminal and use `tail -f out.txt` to see the progress.

```{code-cell} ipython3
from IPython.display import display, FileLink
display(FileLink('out.txt'))
```

```{code-cell} ipython3
with new_trainer_context(config=config) as ctx:
    config = ctx.config
    task = ctx.task
    trainer = ctx.trainer
    task.setup(trainer)
    task.run()
```

```{code-cell} ipython3
! head out.txt
! tail out.txt
```

Now, you are all set to carry on with what ever subsequent analysis you want to do.
