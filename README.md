# Previsão de Preço Médio de Pedido

## Descrição
Modelo de regressão que estima o valor médio do pedido com base em histórico, categoria e localização. Desenvolvido para auxiliar na estratégia de pricing.

## Objetivo
Prever o preço médio de pedidos utilizando:
- **Histórico de compras** do cliente
- **Categoria** do produto
- **Localização** geográfica
- **Features temporais** (sazonalidade, dia da semana, etc.)

## Tecnologias
- **Python 3.8+**
- **LightGBM** - Modelo de gradient boosting
- **Pandas** - Manipulação de dados
- **Scikit-learn** - Pré-processamento e métricas
- **Matplotlib/Seaborn** - Visualização

## Estrutura do Projeto
```
.
├── data/
│   ├── raw/                            # Dados brutos
│   │   ├── customers.csv               # Base de clientes
│   │   └── orders.csv                  # Histórico de pedidos
│   ├── processed/                      # Dados processados
│   │   └── orders_with_features.csv    # Com feature engineering
│   └── generate_data.py                # Gerador de dados sintéticos
├── src/
│   ├── __init__.py                     # Módulo Python
│   ├── features.py                     # Feature engineering avançado
│   ├── train.py                        # Pipeline de treinamento
│   └── predict.py                      # Sistema de predição
├── models/                             # Artefatos do modelo
│   ├── lightgbm_model.txt              # Modelo treinado
│   ├── label_encoders.pkl              # Encoders categóricos
│   ├── feature_engineer.pkl            # Feature transformer
│   ├── feature_names.json              # Nomes das features
│   └── metrics.json                    # Métricas de avaliação
├── config.py                           # Configurações centralizadas
├── examples.py                         # Exemplos de uso
├── tests.py                            # Testes automatizados
├── requirements.txt                    # Dependências
├── README.md                           # Este arquivo
```

## Como Usar

### 1. Instalação
```bash
pip install -r requirements.txt
```

### 2. Gerar Dados (Opcional - para teste)
```bash
python data/generate_data.py
```

### 3. Análise Exploratória
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 4. Treinar Modelo
```bash
python src/train.py
```

### 5. Fazer Predições
```bash
python src/predict.py
```

## Features Implementadas

### Features Históricas
- Número de pedidos anteriores
- Valor médio de pedidos passados
- Valor total gasto
- Tempo desde último pedido
- Recência, Frequência, Valor (RFM)

### Features Temporais
- Dia da semana
- Mês
- Trimestre
- Fim de semana (flag)
- Sazonalidade

### Features Geográficas
- Estado
- Região
- Densidade populacional da região

### Features de Categoria
- Categoria do produto
- Preço médio da categoria
- Popularidade da categoria

## Métricas de Avaliação
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coeficiente de Determinação)

## Aprendizados
- Feature engineering para dados de e-commerce
- Modelagem com LightGBM
- Otimização de hiperparâmetros
- Tratamento de sazonalidade

## Próximos Passos
- [ ] Implementar validação cruzada temporal
- [ ] Adicionar mais features de interação
- [ ] Criar API REST para predições
- [ ] Deploy em produção

## Arquivos Adicionais

- **GUIA_EXECUCAO.md** - Passo a passo detalhado de como executar o projeto
- **DOCUMENTACAO_TECNICA.md** - Documentação técnica completa (features, modelo, arquitetura)
- **config.py** - Configurações centralizadas (ajuste parâmetros aqui)
- **examples.py** - 6 exemplos práticos de uso do modelo
- **tests.py** - Suite de testes automatizados

## Executar Testes

```bash
python tests.py
```

Testa:
- Geração de dados
- Feature engineering
- Modelo treinado
- Predições
- Métricas
- Batch processing

