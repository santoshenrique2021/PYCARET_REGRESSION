# PYCARET_REGRESSION
Este repositório apresenta os códigos da biblioteca *PyCaret* para problemas de regressão.

# Finalidade
*PyCaret* é uma biblioteca de machine learning de low-code que automatiza o workflow de modelagem. Este repositório funciona como um tutorial para problemas de regressão, com um exemplo numérico de estimação de preços de imóveis baseado em variáveis explicativas (categóricas e quantitativas).

# Instruções de instalação

## Pré-requisitos
1. Ter o [Anaconda](https://www.anaconda.com/download) ou [Miniconda](https://docs.conda.io/en/latest/miniconda.html) instalado.
2. Baixar os arquivos `requirements.txt` e `base_dados.csv` do repositório.

## Configuração do Ambiente Virtual
Recomenda-se a criação de um ambiente virtual isolado. Siga estes passos:

1. Abrir o **Anaconda Prompt** (Windows) ou terminal (Linux/Mac)
2. Criar um novo ambiente (substitua `nome_do_ambiente` por um nome de sua escolha):
```bash 
conda create --name nome_do_ambiente python=3.11.8 -y
```
3. Ativar o ambiente virtual.
```bash 
conda activate nome_do_ambiente
```
4. Instalar as dependências:
```bash 
pip install -r requirements.txt
```
5. (Opcional) Caso o pip não funcione, você pode tentar com conda:
```bash 
while read requirement; do conda install --yes $requirement; done < requirements.txt
```


## Notas importantes

1. A versão do *Python* instalada é a 3.11.8.
2. A versão do *Pycaret* instalada é a 3.3.0.
* Para evitar conflitos, sempre ative o ambiente virtual antes de trabalhar no projeto
* Para desativar o ambiente após o uso: conda deactivate

