# PYCARET_REGRESSION

**Versão**: 1.0 | **Data**: Junho de 2026 | **Tecnologia**: PyCaret 3.3.0 · Python 3.11.8. 

## Autor
**Henrique Santos** Economista e Cientista de Dados

* **Formação acadêmica**:
   * **Graduação**: Ciências econômicas (UFPE - Universidade Federal de Pernambuco);
   * **Mestrado**: Engenharia de produção (UFPE);
   * **Doutorado**: Biometria e estatística aplicada (UFRPE - Universidade Federal Rural de Pernambuco);
   * **Pós-doutorado**: Estatística (UFPE).

* **Atuação profissional**:
    * Desde 2020, trabalho na área de ciência de dados, analytics e modelagem. Atualmente trabalho na FSBR (Fábrica de Software do Brasil), alocado na SEFAZ-PE realizando projetos de inteligência artificial visando a otimização de processos de tributação fiscal.

### Contato
1. E-mail: santos.henrique624@gmail.com
2. Github: https://github.com/santoshenrique2021
3. Linkedin: https://www.linkedin.com/in/henriquesantos2021/

## Finalidade

Este repositório corresponde a um tutorial de como utilizar a biblioteca  [PyCaret](https://pycaret.gitbook.io/docs), AutoML low-code, que **automatiza todo o workflow de modelagem**. 

Por meio de **poucas linhas de código**, é possível construir um pipeline de um projeto de machine learning e comparar o **poder preditivo de vários algoritmos**.

Um exemplo sobre a estimação de preço de imóveis (**regressão supervisionada**) é utilizado para ilustrar o potencial de uso desta biblioteca.

> **Aprendizado Supervisionado**: Previsão de uma variável alvo contínua a partir de um conjunto de variáveis independentes, que podem ser quantitativas ou qualitativas. Assim, a partir de dados históricos é possível realizar este mapeamento preditivo.


## Instruções de instalação

### Pré-requisitos
1. Ter o [Anaconda](https://www.anaconda.com/download) ou [Miniconda](https://docs.conda.io/en/latest/miniconda.html) instalado.
2. Baixar os arquivos `requirements.txt` e `base_dados.csv` do repositório.

### Configuração do Ambiente Virtual
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
6. Execução do notebook.
```bash 
jupyter notebook
```

## Notas importantes

* Para evitar conflitos, sempre ative o ambiente virtual antes de trabalhar no projeto.
* Para desativar o ambiente após o uso: `conda deactivate`.

