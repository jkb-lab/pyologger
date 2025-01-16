See the [Scientific Python Developer Guide][spc-dev-intro] for a detailed
description of best practices for developing scientific packages.

[spc-dev-intro]: https://learn.scientific-python.org/development/

# Contributing
Instructions for developers seeking to contribute to `pyologger`.

## Environment setup
1. Create virtual environment from which to run your code:

```python -m venv venv``` 

This creates a folder `venv`, ignored by git by default, that contains all library-related files.

2. Activate your virtual environment:

```source venv/Scripts/activate``` on PC
```source venv/bin/activate``` on Mac

3. Install the package `pyologger` localling using: 

```pip install .``` 

4. As you develop, please remember to add any new packages used into the `pyproject.toml` file. 

## Updating the documentation
For any new functions, please use the sphinx documentation syntax:

```
"""[Summary]

:param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
:type [ParamName]: [ParamType](, optional)
...
:raises [ErrorType]: [ErrorDescription]
...
:return: [ReturnDescription]
:rtype: [ReturnType]
"""
```

From the `docs` folder, run the following command to update the documentation upon adding a new function. This will re-generate the automated documentation of the package and any sub-packages or sub-modules.

`sphinx-apidoc -o ./source ../pyologger`

Then re-create the `html` using:

`make clean`

followed by:

`make html`