# Guia de Execução - Previsão de Preço de Pedidos

## Pré-requisitos

1. **Python 3.8+** instalado
2. **pip** atualizado

## Instalação

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

Isso instalará todas as bibliotecas necessárias:
- pandas
- numpy
- scikit-learn
- lightgbm
- matplotlib
- seaborn
- jupyter
- plotly
- shap

## Passo a Passo

### Passo 1: Gerar Dados Sintéticos

Execute o script para gerar dados de exemplo:

```bash
python data/generate_data.py
```

**Saída esperada:**
- `data/raw/customers.csv` - 1.000 clientes
- `data/raw/orders.csv` - 10.000 pedidos

### Passo 2: Análise Exploratória (Opcional)

Abra o notebook de EDA para entender os dados:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Este notebook contém:
- Estatísticas descritivas
- Análise por categoria
- Análise geográfica
- Análise temporal
- Correlações
- Insights para modelagem

### Passo 3: Criar Features

Execute o script de feature engineering:

```bash
cd src
python features.py
```

**Saída esperada:**
- `data/processed/orders_with_features.csv`

Features criadas:
- Históricas (RFM, pedidos anteriores, etc.)
- Temporais (sazonalidade, dia da semana, etc.)
- Geográficas (estado, região)
- Categoria (preços médios, popularidade)
- Interações entre variáveis

### Passo 4: Treinar Modelo

Execute o script de treinamento:

```bash
python src/train.py
```

**O que acontece:**
1. Carrega dados brutos
2. Aplica feature engineering
3. Divide em treino/teste (80/20)
4. Treina modelo LightGBM
5. Avalia com métricas (RMSE, MAE, R², MAPE)
6. Realiza validação cruzada (5-fold)
7. Mostra feature importance
8. Salva modelo e artefatos

**Saída esperada:**
- `models/lightgbm_model.txt` - Modelo treinado
- `models/label_encoders.pkl` - Encoders
- `models/feature_engineer.pkl` - Feature engineer
- `models/feature_names.json` - Nomes das features
- `models/metrics.json` - Métricas de avaliação

**Tempo estimado:** 2-5 minutos

### Passo 5: Fazer Predições

#### Modo 1: Exemplos Demonstrativos

```bash
python src/predict.py
```

Executa exemplos de:
- Predição única
- Predição em lote

#### Modo 2: Interativo

```bash
python src/predict.py --interactive
```

Permite inserir dados manualmente e obter predições em tempo real.

**Exemplo de interação:**
```
Categoria: Eletrônicos
Estado: SP
Segmento do Cliente: Ouro
Idade do Cliente: 35

PREÇO PREVISTO: R$ 1.245,67
```

#### Modo 3: Predição em Lote

```bash
python src/predict.py --batch caminho/para/arquivo.csv
```

Processa múltiplos pedidos de uma vez e salva resultado em `predictions.csv`.

### Passo 6: Explorar Modelagem (Opcional)

Abra o notebook de modelagem:

```bash
jupyter notebook notebooks/02_modeling.ipynb
```

Este notebook permite:
- Experimentar diferentes parâmetros
- Visualizar predições vs valores reais
- Analisar erros por categoria
- Entender feature importance

## Métricas Esperadas

Com os dados sintéticos, você deve obter aproximadamente:

- **RMSE:** R$ 80-120
- **MAE:** R$ 60-90
- **R²:** 0.85-0.95
- **MAPE:** 15-25%

## Fluxo Completo (Resumo)

```bash
# 1. Gerar dados
python data/generate_data.py

# 2. Criar features (opcional - o train.py já faz isso)
python src/features.py

# 3. Treinar modelo
python src/train.py

# 4. Fazer predições
python src/predict.py
# ou
python src/predict.py --interactive
```

## Estrutura dos Arquivos Gerados

```
.
├── data/
│   ├── raw/
│   │   ├── customers.csv          # Dados de clientes
│   │   └── orders.csv              # Dados de pedidos
│   └── processed/
│       └── orders_with_features.csv # Dados com features
├── models/
│   ├── lightgbm_model.txt          # Modelo treinado
│   ├── label_encoders.pkl          # Encoders
│   ├── feature_engineer.pkl        # Feature engineer
│   ├── feature_names.json          # Nomes das features
│   └── metrics.json                # Métricas
└── predictions.csv                 # Predições (se usar --batch)
```

## Solução de Problemas

### Erro: "Module not found"
```bash
pip install -r requirements.txt
```

### Erro: "File not found"
Certifique-se de estar no diretório raiz do projeto.

### Erro no training
Verifique se os dados foram gerados corretamente:
```bash
dir data\raw  # ou ls data/raw no Linux/Mac
```

## Para Uso em Produção

Para usar com dados reais:

1. **Substitua os dados sintéticos** por seus dados reais
2. **Ajuste o schema** em `generate_data.py` se necessário
3. **Re-treine o modelo** com seus dados
4. **Valide as métricas** e ajuste hiperparâmetros se necessário
5. **Implemente API REST** (opcional) para servir predições

## Conceitos Aplicados

- Feature Engineering avançado
- Regressão com LightGBM
- Validação cruzada
- Tratamento de dados categóricos
- Análise de importância de features
- Métricas de regressão
- Sazonalidade e tendências temporais

## Dicas

1. **Experimente diferentes parâmetros** no `train.py`
2. **Adicione mais features** no `features.py`
3. **Use os notebooks** para análise interativa
4. **Documente suas alterações** para manter rastreabilidade

---

**Bom projeto!**
