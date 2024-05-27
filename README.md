# PYCARET_REGRESSION
Repositório com uma aplicação do *Pycaret* para problemas de regressão.

# Instruções de instalação

Caso queira executar este notebook, recomenda-se instalar o arquivo *requirements*. Adicionalmente, deve-se abrir o arquivo em CSV chamado *base_dados*. 


## Dinâmica

A estrutura de instalação foi baseada através do Anaconda, ou seja, a criação do *ambiente virtual*. Assim, as etapas a serem seguidas são:

1. Abrir o **Anaconda Prompot**.
2. Criar um ambiente virtual. Escreva  o nome que deseja no lugar de *nome_do_ambiente*.
```bash 
conda create --name nome_do_ambiente
```
3. Ative o ambiente virtual.
```bash 
conda activate nome_do_ambiente
```
4. Instalar o arquivo *requirements*. Nele estão todas as bibliotecas e versões utilizados na modelagem.
```bash 
conda install --file requirements.txt
```

## Nota

1. A versão do *Python* instalada é a 3.11.8.
2. A versão do *Pycaret* instalada é a 3.3.0.

Se não for instalar pelo Anaconda, recomenda-se instalar, primeiramente, a versão exposta do *Python* e, posteriormente, a versão do *Pycaret*. Ao instalar o *Pycaret*, outras bibliotecas do *Python* (*Pandas*, *Numpy*, ...) também serão instaladas.