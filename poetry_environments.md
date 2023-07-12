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

Init a Data Science project with Poetry:
```
mkdir <your_project>
cd <your_project>
poetry init
```

Add dependencies to project in a generic way:
```
poetry add session-info pandas jupyter ipykernel <library_1> <library_1> ... <library_n>
```

Or specify the exact version of a package:
```
poetry add numpy==1.24
```


## 3. Manage Poetry environments

If a project is started without any Poetry environment activated, Poetry creates it in
the following path: `/home/user/.cache/pypoetry/virtualenvs/`. To manange poetry
envs:
```
poetry env info
```
```
poetry env list
```
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
poetry run ipython kernel install --user --name=<KERNEL_NAME>
```

Open Jupyter Lab from another shell:
```
jupyter lab
```
