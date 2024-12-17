# Poetry
___
Poetry URL: https://python-poetry.org/docs/

## 1. Installation

Install Pyenv:
```
curl https://pyenv.run | bash
```

Add this lines to your terminal shrc config file (.bashrc, .zshrc, etc.):
```
bashCopyexport PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Install Poetry:
```
curl -sSL https://install.python-poetry.org | python3 -
```

In Linux/Unix, Poetry is installed into the following user-specific directory: `~/.local/share/pypoetry`

Add Poetry to your PATH:

The installer creates a poetry wrapper in a well-known, platform-specific directory: `$HOME/.local/bin` on Unix.

```
echo $PATH
export PATH="/home/<tu_usuario>/.local/bin:$PATH"
```

## 2. Create a Project

Install desired Python version for the project with Pyenv:
```
pyenv install 3.X.X
```

Installed python version is often stored in `~/.pyenv/versions/3.X.X/bin/python`.


Init a project with Poetry. This command will help you create a pyproject.toml file interactively by prompting you to provide basic information about your package. It will interactively ask you to fill in the fields, while using some smart defaults.
```
mkdir <your_project>
cd <your_project>
```
```
pyenv local 3.X.X
```
```
poetry init
```

You can check the python version in your project with:
```
echo $(pyenv which python)
```

Configure Poetry to use Pyenv Python version:
```
poetry env use 3.X.X
```

Add dependencies to project in a generic way. The add command adds required packages to your pyproject.toml and installs them. If you do not specify a version constraint, poetry will choose a suitable one based on the available package versions.
```
poetry add session-info pandas jupyter ipykernel <library_1> <library_1> ... <library_n>
```

Or specify the exact version of a package:
```
poetry add numpy==1.24
```

By default, Poetry will try to use the Python version used during Poetry’s installation to create the virtual environment for the current project.

However, for various reasons, this Python version might not be compatible with the python range supported by the project. In this case, Poetry will try to find one that is and use it. If it’s unable to do so then you will be prompted to activate one explicitly. For this specific purpose, you can use the env use command to tell Poetry which Python version to use for the current project.

```
poetry env use /full/path/to/python
```

Dependencies for a project can be specified in various forms, which depend on the type of the dependency and on the optional constraints that might be needed for it to be installed.
https://python-poetry.org/docs/dependency-specification/#using-the--operator


## 3. Manage Poetry environments

If a project is started without any Poetry environment activated, Poetry creates it in the following path: 
```
/home/user/.cache/pypoetry/virtualenvs/
```

To manange poetry envs you can:

Show info of the current environment:
```
poetry env info
```

List all Poetry environments:
```
poetry env list
```

Remove a Poetry environment:
```
poetry env remove
```

## 4. Activate/deactivate Poetry environments

Activate a Poetry environment:
```
poetry shell
```

Deactivate a Poetry environment:
```
exit
```

## 5. Use a Poetry environment inside Jupyter Lab

Create a kernel linked to a virtual environement to use it inside Jupyter Lab.

Activate a Poetry environment:
```
poetry shell
```

Create a kernel linked to the environment:
```
poetry run ipython kernel install --user --name=<your_kernel>
```

Open Jupyter Lab from another shell, and select the kernel created in previous step:
```
jupyter lab
```
