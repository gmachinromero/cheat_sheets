# Poetry
___
Poetry URL: https://python-poetry.org/docs/

## 1. Installation

Install Poetry:
```
curl -sSL https://install.python-poetry.org | python3 -
```

Add Poetry to your PATH:
```
echo $PATH
export PATH="/home/<tu_usuario>/.local/bin:$PATH"
```

## 2. Create a Project

Init a Data Science project with Poetry. This command will help you create a pyproject.toml file interactively by prompting you to provide basic information about your package. It will interactively ask you to fill in the fields, while using some smart defaults.
```
mkdir <your_project>
cd <your_project>
poetry init
```

Add dependencies to project in a generic way. The add command adds required packages to your pyproject.toml and installs them. If you do not specify a version constraint, poetry will choose a suitable one based on the available package versions.
```
poetry add session-info pandas jupyter ipykernel <library_1> <library_1> ... <library_n>
```

Or specify the exact version of a package:
```
poetry add numpy==1.24
```

Dependencies for a project can be specified in various forms, which depend on the type of the dependency and on the optional constraints that might be needed for it to be installed.
https://python-poetry.org/docs/dependency-specification/#using-the--operator


## 3. Manage Poetry environments

If a project is started without any Poetry environment activated, Poetry creates it in
the following path: `/home/user/.cache/pypoetry/virtualenvs/`. To manange poetry
envs you can:

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
