# Documentação Técnica — Machine Learning com PyCaret
## Precificação de Imóveis Residenciais

**Versão:** 1.0 | **Data:** Junho 2026 | **Tecnologia:** PyCaret 3.3.0 · Python 3.11.8

---

## 1. Visão Geral do Projeto

### 1.1 Objetivo

Desenvolver um modelo de regressão supervisionada para estimar o preço de imóveis residenciais a partir de suas características estruturais e de localização. O projeto serve como estudo de caso aplicado da biblioteca **PyCaret**, demonstrando o fluxo completo de AutoML low-code.

### 1.2 Estratégia de Negócio

O modelo de precificação tem três aplicações diretas:

- **Precificação ao consumidor final:** usar a estimativa do modelo como referência de preço justo
- **Priorização de portfólio:** alocar esforços de venda a imóveis com maior valor estimado
- **Alocação de corretores:** direcionar profissionais especializados por faixa de preço

### 1.3 Tecnologias Utilizadas

| Biblioteca | Versão | Finalidade |
|-----------|--------|-----------|
| Python | 3.11.8 | Linguagem base |
| PyCaret | 3.3.0 | AutoML — treinamento, avaliação e deploy |
| Pandas | — | Manipulação de dados |
| NumPy | — | Operações numéricas |
| Seaborn / Matplotlib | — | Visualizações |
| scikit-learn | — | Extração de coeficientes |
| sktime | — | Cálculo do MAPE |

---

## 2. Base de Dados

### 2.1 Descrição Geral

| Atributo | Valor |
|---------|-------|
| Formato | CSV (separador `,`) |
| Observações | 545 imóveis |
| Variáveis | 14 (1 ID + 12 preditoras + 1 alvo) |
| Linhas com missing | 56,5% (ao menos 1 campo ausente) |

### 2.2 Dicionário de Variáveis

| Variável | Tipo | Descrição | Missing? |
|---------|------|-----------|---------|
| IDRESIDENCIA | Texto (ID) | Identificador único do imóvel | Não |
| **PRECO** | Inteiro | **Variável alvo — preço em R$** | Não |
| AREA | Inteiro | Área total em m² | Não |
| QUARTOS | Decimal | Número de quartos | Sim |
| BANHEIROS | Decimal | Número de banheiros | Sim |
| ANDARES | Inteiro | Número de andares | Não |
| FLAGCENTRO | Categórico (SIM/NAO) | Localizado em área central? | Sim |
| FLAGQUARTOHOSPEDE | Categórico (SIM/NAO) | Possui quarto de hóspedes? | Sim |
| FLAGPORAO | Categórico (SIM/NAO) | Possui porão? | Sim |
| FLAGAGUAMORNA | Categórico (SIM/NAO) | Possui aquecimento de água? | Sim |
| FLAGARCONDICIONADO | Categórico (SIM/NAO) | Possui ar-condicionado? | Sim |
| VAGASESTACIONAMENTO | Decimal | Número de vagas | Sim |
| FLAGAREAPREFERENCIAL | Categórico (SIM/NAO) | Em área preferencial? | Sim |
| MOBILIADA | Ordinal (NAO/PARCIALMENTE/SIM) | Nível de mobília | Sim |

> **Nota sobre MOBILIADA:** variável tratada como **ordinal** com ordenação implícita: NAO < PARCIALMENTE < SIM, refletindo o impacto crescente sobre o valor do imóvel.

---

## 3. Pipeline de Machine Learning

O pipeline implementado com PyCaret segue seis etapas sequenciais:

```
Dados Brutos
    ↓
[1] setup()          — Pré-processamento automático
    ↓
[2] compare_models() — Seleção do melhor algoritmo
    ↓
[3] create_model()   — Treinamento com cross-validation
    ↓
[4] tune_model()     — Otimização de hiperparâmetros
    ↓
[5] finalize_model() — Retreino com 100% dos dados
    ↓
[6] save_model()     — Serialização para produção
```

---

## 4. Pré-processamento — `setup()`

### 4.1 Parâmetros Utilizados

```python
exp = setup(
    df,
    target                 = 'PRECO',
    session_id             = 1935,          # semente para reprodutibilidade
    train_size             = 0.65,          # 354 treino / 191 teste
    ignore_features        = ['IDRESIDENCIA'],
    normalize              = True,
    normalize_method       = 'minmax',      # escala [0, 1]
    numeric_imputation     = 'median',
    categorical_imputation = 'mode',
    ordinal_features       = {
        'MOBILIADA': ['NAO', 'PARCIALMENTE', 'SIM'],
        # demais FLAGS tratadas como ordinais binárias
    },
    experiment_name        = 'EXP_REGRESSAO'
)
```

### 4.2 Transformações Realizadas

| Etapa | Técnica | Detalhe |
|------|---------|---------|
| Imputação numérica | Mediana | Robusta a outliers |
| Imputação categórica | Moda | Valor mais frequente |
| Encoding ordinal | Mapeamento numérico | NAO=0, PARCIALMENTE=1, SIM=2 |
| Normalização | MinMax | Transforma para [0, 1] |
| Divisão treino/teste | Holdout | 65% / 35% estratificado |

**Exemplo de transformação (variável AREA):**

| Estatística | Original | Normalizado |
|------------|---------|-------------|
| Mínimo | 1.650 m² | 0.00 |
| Mediana | 4.600 m² | 0.20 |
| Máximo | 16.200 m² | 1.00 |

---

## 5. Modelagem

### 5.1 Comparação de Modelos

A função `compare_models(sort='MAPE', fold=10)` treina e avalia todos os algoritmos disponíveis via 10-fold cross-validation na base de treino. **Esta etapa é obrigatória antes de escolher um algoritmo** — ela evidencia qual modelo performa melhor nos dados.

### 5.2 Modelo Selecionado: Ridge Regression

**Ridge** é um modelo de regressão linear com regularização L2. Ele minimiza a função de perda:

\[ \mathcal{L}(\beta) = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} \beta_j^2 \]

onde `α` (alpha) controla a força da regularização:
- **α → 0**: equivale a OLS (sem regularização)
- **α → ∞**: coeficientes tendem a zero

O Ridge é adequado quando há **multicolinearidade** entre as features, pois distribui os coeficientes de forma mais estável que OLS puro.

### 5.3 Resultados do Modelo

#### Cross-Validation (base de treino — 10 folds)

| Métrica | Média | Desvio Padrão |
|--------|------|--------------|
| MAE | R$ 798.445 | ±R$ 106.082 |
| RMSE | R$ 1.064.003 | ±R$ 197.151 |
| R² | 0.6434 | ±0.0875 |
| MAPE | 17.73% | ±2.72% |

#### Avaliação na Base de Teste (holdout final)

| Métrica | Valor | Interpretação |
|--------|------|--------------|
| MAE | R$ 853.069 | Erro médio absoluto de ~R$ 853 mil |
| RMSE | R$ 1.211.693 | Penaliza erros grandes |
| R² | 0.5940 | Explica 59,4% da variância do preço |
| MAPE | **18.76%** | Erra em média ~18,8% do preço real |

> **Contextualização:** o MAPE do modelo (18.76%) deve ser comparado com um **modelo ingênuo** (que sempre prevê a média dos preços, MAPE ≈ 28-32%). A melhora sobre o baseline é de ~10-12 pontos percentuais.

---

## 6. Feature Importance

Os coeficientes do Ridge (em valor absoluto) revelam a importância relativa de cada variável:

| Rank | Feature | Coeficiente (abs) | Interpretação |
|-----|---------|------------------|--------------|
| 1 | AREA | 2.966.935 | Preditor dominante |
| 2 | BANHEIROS | 2.181.051 | 2ª maior influência |
| 3 | ANDARES | 1.766.142 | Impacto estrutural |
| 4 | FLAGAGUAMORNA | 920.712 | Infraestrutura |
| 5 | VAGASESTACIONAMENTO | 887.097 | Comodidade |
| 6 | FLAGARCONDICIONADO | 831.382 | Conforto |
| 7 | FLAGAREAPREFERENCIAL | 685.372 | Localização |
| 8 | FLAGQUARTOHOSPEDE | 601.221 | Espaço extra |

**Insights econômicos:**
- A **AREA** em m² é o preditor mais relevante, como esperado no mercado imobiliário.
- Features de **conforto e infraestrutura** (AC, água morna, vagas) têm peso expressivo.
- A variável **MOBILIADA** (dummy 0.0) apresenta coeficiente **negativo** — imóveis sem mobília têm preço menor em relação à referência, coerente economicamente.

---

## 7. Tunagem de Hiperparâmetros

```python
param_grid = {'alpha': [0.0001, 0.01, 0.1, 1, 2, 5, 10]}

tuned_model = tune_model(
    model_ridge,
    optimize    = 'MAPE',
    fold        = 10,
    custom_grid = param_grid,
    n_iter      = 10
)
```

**Resultado:** o modelo original (α = 1.0) já era o ótimo dentro do grid testado. A tunagem não trouxe melhora adicional — `tune_model()` retorna o modelo original automaticamente neste caso.

---

## 8. Finalização e Deploy

### 8.1 Finalização

```python
final_model = finalize_model(model_ridge)
```

`finalize_model()` retreina o modelo com **100% dos dados** (treino + teste). O R² melhora para ~0.666 ao utilizar toda a informação disponível.

### 8.2 Salvamento

```python
save_model(final_model, 'model_ridge_precificacao')
# → Gera: model_ridge_precificacao.pkl
```

O arquivo `.pkl` contém o **pipeline completo**: imputação + encoding + normalização + estimador. Não é necessário replicar nenhum pré-processamento em produção.

### 8.3 Carregamento em Produção

```python
from pycaret.regression import load_model, predict_model

# Carregar
modelo_prod = load_model('model_ridge_precificacao')

# Predição em novos imóveis (dados brutos — sem pré-processamento manual)
df_novos = pd.read_csv('novos_imoveis.csv')
predicoes = predict_model(modelo_prod, data=df_novos)
```

---

## 9. Diagnóstico e Limitações

### 9.1 Limitações do Modelo Atual

| Limitação | Impacto | Sugestão |
|----------|---------|---------|
| R² = 0.59 (moderado) | Explica apenas 59% da variância | Testar XGBoost/LightGBM |
| Ausência de localização georreferenciada | Bairro/CEP são fortes preditores | Adicionar coordenadas ou cluster de bairro |
| Distribuição assimétrica do preço | Ridge assume linearidade | Testar transformação log(PRECO) |
| 545 observações | Base pequena para ML complexo | Ampliar base se possível |

### 9.2 Próximos Passos Recomendados

1. **Comparação completa de algoritmos** via `compare_models()` antes de fixar o Ridge
2. **Feature engineering:** razão AREA/QUARTOS, indicador de "imóvel de luxo"
3. **Transformação log** na variável alvo para lidar com assimetria
4. **SHAP values** para interpretabilidade mais robusta que coeficientes Ridge
5. **Validação temporal** se os dados possuírem dimensão de tempo (preços variam ao longo dos anos)

---

## 10. Glossário de Métricas

| Métrica | Fórmula | Interpretação |
|--------|---------|--------------|
| MAE | \(\frac{1}{n}\sum|y_i - \hat{y}_i|\) | Erro médio em R$ — intuitivo, mesma unidade da variável alvo |
| MSE | \(\frac{1}{n}\sum(y_i - \hat{y}_i)^2\) | Penaliza erros grandes — sensível a outliers |
| RMSE | \(\sqrt{MSE}\) | Raiz do MSE — mesma unidade da variável alvo |
| R² | \(1 - \frac{SS_{res}}{SS_{tot}}\) | Proporção da variância explicada. R²=1 perfeito, R²=0 equivale à média |
| MAPE | \(\frac{100}{n}\sum\frac{|y_i - \hat{y}_i|}{y_i}\) | Erro em percentual — comparável entre diferentes escalas |

---

*Documentação gerada para fins didáticos — Seminário de Machine Learning com PyCaret*
