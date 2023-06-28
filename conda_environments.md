# Enviroments
___

## 1. Creation of enviroments

Create a clean enviroment ready for Python 3 or later.
```
conda create --name <env> python=3
```

Create an enviroment using a environment.yml conda file.
```
conda env create -f environment.yml
```

Create an enviroment using a requirements.txt conda file.
```
conda create --name <env> --file requirements.txt
```

Create an enviroment using a requirements.txt pip file.
```
pip install -r requirements.txt
```

## 2. Information about enviroments

List all available enviroments
```
conda info --envs
```

Info about an enviroment (execute inside desired enviroment)
```
conda info
```

Libraries inside an enviroment (execute inside desired enviroment)
```
conda list
```

## 3. Manage enviroments

Activate and deactivate enviroments
```
conda activate <env>
conda deactivate
```

Add a library inside an enviroment
```
conda install --name <env> <library>
conda install <library>
```

Update libraries from an enviroment
```
conda update <library>
conda update --all
```

Delete an enviroment
```
conda remove --name <env> --all
```

## 4. Environments in Jupyter

From the base environment, list kernels:
```
jupyter kernelspec list
```

Create a kernel linked to with an environment:
```
conda activate <env>
conda install ipykernel
ipython kernel install --user --name=<your_kernal_name>
conda deactivate
```

Delete kernel:
```
jupyter kernelspec uninstall <your_kernal_name>
```


## 5. Generate requirements.txt files
```
conda list -e > requirements_conda.txt
pip list --format=freeze > requirements_pip.txt
```
```
conda env export --no-builds > env_neural_prophet.yml
```
