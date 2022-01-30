## Link to colab
[https://colab.research.google.com/drive/1BKk4dS9gP-AoTYa3Aa1g7xvK585VD7Cr?usp=sharing](https://colab.research.google.com/drive/1BKk4dS9gP-AoTYa3Aa1g7xvK585VD7Cr?usp=sharing)

## Setup

You can run this code on your own machine or on Google Colab. 

1. **Local option:** If you choose to run locally, you will need to install MuJoCo and some Python packages; see [installation.md](installation.md) for instructions.
2. **Colab is possible but not supported:** The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badge below:

## Complete the code

Fill in sections marked with `TODO`. In particular, see
 - [infrastructure/rl_trainer.py](ift6163/infrastructure/rl_trainer.py)
 - [policies/MLP_policy.py](ift6163/policies/MLP_policy.py)
 - [infrastructure/replay_buffer.py](ift6163/infrastructure/replay_buffer.py)
 - [infrastructure/utils.py](ift6163/infrastructure/utils.py)
 - [infrastructure/pytorch_util.py](ift6163/infrastructure/pytorch_util.py)

Look for sections maked with `HW1` to see how the edits you make will be used.
Some other files that you may find relevant
 - [run_hw1.py](ift6163/run_hw1.py) (if running locally) or [scripts/run_hw1.ipynb](ift6163/scripts/run_hw1.ipynb) (if running on Colab)
 - [agents/bc_agent.py](ift6163/agents/bc_agent.py)

See the homework pdf for more details.

## Run the code

Tip: While debugging, you probably want to keep the flag `render: false` in the `conf/config.yaml` file which will disable video logging and speed up the experiment. However, feel free to remove it to save videos of your awesome policy!

### Section 1 (Behavior Cloning)
Command for problem 1:

```
python run_hw1.py 
```

Make sure to also try another environment.
See the homework PDF for more details on what else you need to run.

### Section 2 (DAgger)
Command for section 1:
(Note the `--do_dagger` flag, and the higher value for `n_iter`)

Edit the `conf/config.yaml` file `alg.do_dagger` to `true` and use the command

```
python run_hw1.py \
```

Make sure to also try another environment.
See the homework PDF for more details on what else you need to run.

## Visualization the saved tensorboard event file:

You can visualize your runs using tensorboard:
```
tensorboard --logdir output/
```

You will see scalar summaries as well as videos of your trained policies (in the 'images' tab).

You can choose to visualize specific runs with a comma-separated list:
```
tensorboard --logdir data/run1,data/run2,data/run3...
```

If using on Colab, you may use the `%tensorboard` [line magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html) to do the same thing; see the [notebook](ift6163/scripts/run_hw1.ipynb) for more details.

