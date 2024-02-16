# Summary
![threads_low_Res](https://github.com/drisspg/driss_torch/assets/32754868/df5f6286-ef38-411e-9bc5-90042671a4fd)


Simple CMAKE make project for extending PyTorch with custom cuda kernels for doing
all the things that I wanna try.


## Build
### New scikit-build core way
``` Shell
pip install scikit_build_core
pip install pyproject-metadata

# Install in editable mode
pip install -v --no-build-isolation -e .
```

This will invoke the `pyproject.toml` file and build the project with the `scikit-build` package.
Including, calling cmake and building the shared library.

### What does this do?
So this will go and build the libdriss_torch.so shared library. In which there will be a number of ops that have been registered.
In the python package `driss_torch` there will be small wrappers around the cpp ops with better type hints and documentation.

Could use .ini but this more fun.


#### Tip (Maybe unsafe?)
The total install command from scratch can take a second. If you want a faster dev flow cd into build and once you change your sources of the lib just run `ninja` to rebuild the shared library. This will not work if you are adding and deleting files.
## Intended Usage
You should be able to grab the ops by doing something like

```Python
import torch
import driss_torch

a = torch.randn(10, 10, device='cuda')
b = driss_torch.add_one(b)
print (b)
```

### Install pre-commit hooks
```Shell
pre-commit install
```
### Running Tests
```Shell
pytest test/
```


##### Old build way
This calls cmake directly in the build dir.
``` Shell
# In the root directory
mkdir build && cd build

# Configure the build, to build with compute arch
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja  -DTORCH_CUDA_ARCH_LIST=9.0 ..

```
