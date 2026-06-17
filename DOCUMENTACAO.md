# Documentação Técnica — Precificação de Imóveis com PyCaret

**Versão:** 1.0 | **Data:** Junho de 2026 | **Tecnologia:** PyCaret 3.3.0 · Python 3.11.8

**Autor:** Henrique Santos — Economista e Cientista de Dados  
📧 santos.henrique624@gmail.com · [GitHub](https://github.com/santoshenrique2021) · [LinkedIn](https://www.linkedin.com/in/henriquesantos2021/)

---

## 1. Visão Geral

Este projeto é um tutorial prático de **regressão supervisionada** utilizando a biblioteca [PyCaret](https://pycaret.gitbook.io/docs) — uma solução AutoML low-code que automatiza o workflow completo de modelagem preditiva.

**Contexto de negócio:** Uma imobiliária focada na venda de imóveis de terceiros busca estimar o preço de residências a partir de características estruturais e de localização.

**Objetivo analítico:** Construir um modelo preditivo capaz de estimar o preço (`PRECO`) de um imóvel com base em um conjunto de variáveis independentes quantitativas e qualitativas.

---

## 2. Aplicações de Negócio

| Estratégia | Descrição |
|---|---|
| **Precificação de Oferta** | Usar a estimativa do modelo para definir o preço ao consumidor final |
| **Balizamento de Captação** | Ajustar o preço sugerido pelo proprietário ao valor estimado |
| **Priorização de Carteira** | Ordenar imóveis por faixa de preço estimada para priorizar vendas |
| **Alocação de Equipe** | Direcionar corretores especializados (ex.: alto padrão) conforme a faixa estimada |

---

## 3. Estrutura do Dataset

- **Fonte:** Base histórica de imóveis comercializados pela corretora
- **Dimensão original:** 545 registros × 14 colunas
- **Variável alvo:** `PRECO` (contínua, em R$)
- **Missings:** 56,5% das linhas contêm ao menos um valor ausente

### 3.1 Dicionário de Variáveis

| Variável | Tipo | Descrição |
|---|---|---|
| `ID_RESIDENCIA` | Identificador | Código único do imóvel (ignorado na modelagem) |
| `PRECO` | Numérica (alvo) | Preço do imóvel em R$ |
| `AREA` | Numérica | Área total do imóvel em m² |
| `QUARTOS` | Numérica | Número de quartos |
| `BANHEIROS` | Numérica | Número de banheiros |
| `ANDARES` | Numérica | Número de andares |
| `VAGAS_ESTACIONAMENTO` | Numérica | Número de vagas de garagem |
| `FLAG_CENTRO` | Categórica (SIM/NAO) | Imóvel localizado em área central |
| `FLAG_QUARTO_HOSPEDE` | Categórica (SIM/NAO) | Possui quarto de hóspedes |
| `FLAG_PORAO` | Categórica (SIM/NAO) | Possui porão |
| `FLAG_AGUA_MORNA` | Categórica (SIM/NAO) | Possui aquecimento de água |
| `FLAG_AR_CONDICIONADO` | Categórica (SIM/NAO) | Possui ar-condicionado |
| `FLAG_AREA_PREFERENCIAL` | Categórica (SIM/NAO) | Localizado em área preferencial |
| `MOBILIADA` | Ordinal (NAO/PARCIALMENTE/SIM) | Nível de mobília do imóvel |

---

## 4. Configuração do Experimento (PyCaret `setup`)

| Parâmetro | Valor |
|---|---|
| Session ID | 1935 |
| Train set | 354 registros (65%) |
| Test set | 191 registros (35%) |
| Folds (Cross-Validation) | 10 (KFold) |
| Imputação numérica | Mediana |
| Imputação categórica | Moda |
| Encoding ordinal | `FLAG_*` → OrdinalEncoder |
| Encoding nominal | `MOBILIADA` → OneHotEncoder |
| Normalização | MinMaxScaler |
| Nome do experimento | `EXP_REGRESSAO` |

---

## 5. Pipeline de Pré-processamento

O PyCaret monta automaticamente o seguinte pipeline:

```
1. SimpleImputer (mediana) ........... variáveis numéricas
2. SimpleImputer (moda) .............. variáveis categóricas
3. OrdinalEncoder .................... FLAG_* (NAO=0, SIM=1)
                                       MOBILIADA (NAO=0, PARCIALMENTE=1, SIM=2)
4. OneHotEncoder ..................... MOBILIADA → 3 colunas dummies
5. MinMaxScaler ...................... normalização [0, 1]
6. Ridge Regressor .................. estimador final
```

---

## 6. Comparação de Modelos (`compare_models`)

25 algoritmos foram avaliados via cross-validation de 10 folds. Os principais resultados ordenados por **R²**:

| Ranking | Modelo | MAE | R² | MAPE |
|---|---|---|---|---|
| 🥇 1º | Ridge Regression | 798.445 | **0,6434** | 17,73% |
| 🥈 2º | Lasso Regression | 801.915 | 0,6411 | 17,77% |
| 🥉 3º | Least Angle Regression | 801.916 | 0,6411 | 17,77% |
| 4º | Linear Regression | 801.439 | 0,6415 | 17,78% |
| 5º | Random Forest | 825.827 | 0,6056 | 18,17% |
| ... | CatBoost | 832.400 | 0,5983 | 18,20% |
| ... | Dummy Regressor (baseline) | 1.434.530 | -0,038 | 33,42% |

> **Melhor modelo:** Ridge Regression, com R² de 0,6434 e MAPE de 17,73%, superando modelos ensemble como Random Forest e CatBoost.

---

## 7. Modelo Final — Ridge Regression

### 7.1 Avaliação no Conjunto de Teste

| Métrica | Valor |
|---|---|
| MAE | R$ 853.069 |
| RMSE | R$ 1.211.693 |
| R² | 0,5940 |
| MAPE | 18,76% |

### 7.2 Avaliação Pós-Finalização (full dataset)

Após `finalize_model` (retreinamento com 100% dos dados):

| Métrica | Valor |
|---|---|
| MAE | R$ 784.614 |
| RMSE | R$ 1.079.267 |
| R² | **0,6664** |
| MAPE | 17,39% |

### 7.3 Feature Importance (Coeficientes Ridge)

| # | Feature | Coeficiente (R$) |
|---|---|---|
| 1 | AREA | +2.966.935 |
| 2 | BANHEIROS | +2.181.051 |
| 3 | ANDARES | +1.766.142 |
| 4 | FLAG_AGUA_MORNA | +920.712 |
| 5 | VAGAS_ESTACIONAMENTO | +887.097 |
| 6 | FLAG_AR_CONDICIONADO | +831.382 |
| 7 | FLAG_AREA_PREFERENCIAL | +685.372 |
| 8 | FLAG_QUARTO_HOSPEDE | +601.221 |
| 9 | FLAG_PORAO | +487.606 |
| 10 | QUARTOS | +422.934 |
| 11 | FLAG_CENTRO | +363.404 |
| 12 | MOBILIADA_SIM | +218.213 |
| 13 | MOBILIADA_NAO | -210.570 |

> As variáveis com maior impacto no preço são **Área**, **Banheiros** e **Andares**, reforçando a intuição de mercado imobiliário.

---

## 8. Dependências e Ambiente

### 8.1 Versões principais

| Biblioteca | Versão |
|---|---|
| Python | 3.11.8 |
| PyCaret | 3.3.0 |
| scikit-learn | — |
| pandas | — |
| numpy | — |
| matplotlib | — |
| lightgbm | — |
| xgboost | — |
| catboost | — |

### 8.2 Instalação

```bash
# 1. Criar ambiente virtual
conda create --name nome_do_ambiente python=3.11.8 -y
conda activate nome_do_ambiente

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Executar o notebook
jupyter notebook
```

> ⚠️ **Nota:** Sempre ative o ambiente virtual antes de executar o projeto. Para desativar: `conda deactivate`.

---

## 9. Estrutura do Repositório

```
PYCARET_REGRESSION/
├── MACHINE_LEARNING_PYCARET.ipynb   # Notebook principal com todo o tutorial
├── base_dados.csv                   # Dataset de imóveis
├── requirements.txt                 # Dependências do projeto
└── README.md                        # Documentação resumida
```

---

## 10. Referências

- [Documentação oficial PyCaret](https://pycaret.gitbook.io/docs)
- [PyCaret — Regression Module](https://pycaret.readthedocs.io/en/latest/api/regression.html)
- [scikit-learn — Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
