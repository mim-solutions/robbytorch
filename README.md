# Installation

`Robbytorch` requires Pytorch, however it's not specified in the dependencies - we recommend installing Pytorch manually via conda and only later installing Robbytorch by pip. Pytorch has to be in version `1.6` or higher.

Use your conda env or create a new one:

```
conda create --name <ENV NAME> python=3.8 pip
conda activate <ENV NAME>
```

Install [Pytorch](https://pytorch.org/). If you have older drivers for GPU you may require older version of CUDA, i.e.:

```
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch -c conda-forge
```

or even [older Pytorch version](https://pytorch.org/get-started/previous-versions/):

```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
```

Then run:

```
pip install robbytorch
```

# Usage

The basics of the `Robbytorch` library are explained in the [ipython/RobbytorchTutorial.ipynb](https://github.com/mim-solutions/robbytorch/blob/master/ipython/RobbytorchTutorial.ipynb) juputer notebook.

# TODO

- Streamline logging - currently we need to pass arguments to `MLFLowWriter` at init which we want to do outside of the `Trainer#train_model` function to keep reference to the `Writer` instance. We'd like to invent better flow that would allow us to pass config dict just once;
- Create helper for iterating over rows and cols and printing tensor:

```python
diffs = show - show.roll(shifts=1, dims=0)

cols, rows = 3, 6
for i in range(rows):
    curr = (show[i*cols:(i+1)*cols])
    diff = vis_grad(diffs[i*cols:(i+1)*cols])
    robby.get_image_table(curr, diff, size=(10,10))

robby.widen_outputs()
```

# Packaging

## Development mode

Run from this repository's root dir:
```
python setup.py develop
```

Now you can use the package as if it was installed by `pip`.

When you’re done with a given development task, you can remove the project source from a staging area using:

```
python setup.py develop --uninstall
```

## Creating python package

Use [this tutorial](https://packaging.python.org/tutorials/packaging-projects/). In short:

1. put your source files into `src/<PACKAGE NAME>/` dir and create files: `src/<PACKAGE NAME>/__init__.py`, `setup.py`, `pyproject.toml`, `README.md` and `LICENSE` (also optionally an empty `test` dir)
2. install required packages:
```
python3 -m pip install --upgrade twine build
```
3. build the package, i.e. run from root folder:
```
python3 -m build
```
4. upload the package:
```
python3 -m twine upload dist/*
```

In order to perform the last step you need to have a https://pypi.org/ account and create the API token, which could be conveniently placed in `$HOME/.pypirc`:

```
[pypi]
  username = __token__
  password = pypi-<API TOKEN>
```


## Warnings:
If you update the package you should increment the `version` param in `setup.py` and empty the `dist` dir.