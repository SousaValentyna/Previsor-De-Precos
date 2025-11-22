"""
Arquivo de configuração centralizado para o projeto.
Ajuste os parâmetros aqui para experimentar diferentes configurações.
"""

# ============================================================================
# CONFIGURAÇÕES DE DADOS
# ============================================================================

DATA_CONFIG = {
    # Geração de dados sintéticos
    'num_customers': 1000,
    'num_orders': 10000,
    'start_date': '2022-01-01',
    'end_date': '2024-11-22',
    
    # Seed para reprodutibilidade
    'random_seed': 42,
    
    # Paths
    'raw_data_path': 'data/raw/',
    'processed_data_path': 'data/processed/',
    'model_path': 'models/',
}

# ============================================================================
# CONFIGURAÇÕES DE FEATURE ENGINEERING
# ============================================================================

FEATURE_CONFIG = {
    # Features históricas
    'historical_features': True,
    'rfm_features': True,
    'rolling_windows': [7, 30, 90],  # dias para features rolling
    
    # Features temporais
    'temporal_features': True,
    'cyclic_encoding': True,
    'seasonal_flags': True,
    
    # Features geográficas
    'location_features': True,
    'regional_stats': True,
    
    # Features de categoria
    'category_features': True,
    'category_stats': True,
    
    # Features de interação
    'interaction_features': True,
}

# ============================================================================
# CONFIGURAÇÕES DO MODELO LIGHTGBM
# ============================================================================

MODEL_CONFIG = {
    # Parâmetros principais
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',  # gbdt, dart, goss
    
    # Estrutura da árvore
    'num_leaves': 31,
    'max_depth': -1,  # -1 = sem limite
    'min_child_samples': 20,
    
    # Learning
    'learning_rate': 0.05,
    'num_boost_round': 1000,
    'early_stopping_rounds': 50,
    
    # Sampling
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    
    # Regularização
    'reg_alpha': 0.1,  # L1
    'reg_lambda': 0.1,  # L2
    
    # Outros
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,  # usar todos os cores
}

# ============================================================================
# CONFIGURAÇÕES DE TREINAMENTO
# ============================================================================

TRAINING_CONFIG = {
    # Split de dados
    'test_size': 0.2,
    'val_size': 0.2,  # do treino
    'shuffle': True,
    'stratify': None,
    
    # Validação cruzada
    'cv_folds': 5,
    'cv_shuffle': True,
    
    # Métricas
    'metrics': ['rmse', 'mae', 'r2', 'mape'],
    'primary_metric': 'rmse',
}

# ============================================================================
# CONFIGURAÇÕES DE PREDIÇÃO
# ============================================================================

PREDICTION_CONFIG = {
    # Defaults para novos clientes
    'default_historical_features': {
        'num_pedidos_anteriores': 0,
        'valor_medio_anterior': 0,
        'valor_total_anterior': 0,
        'dias_desde_ultimo_pedido': 999,
        'pedidos_ultimos_30_dias': 0,
        'pedidos_ultimos_90_dias': 0,
        'valor_medio_ultimos_30_dias': 0,
        'std_valor_anterior': 0,
        'max_valor_anterior': 0,
        'min_valor_anterior': 0,
        'tendencia_gasto': 0,
    },
    
    # Limites de predição
    'min_price': 10.0,
    'max_price': 10000.0,
    
    # Batch processing
    'batch_size': 1000,
}

# ============================================================================
# CONFIGURAÇÕES DE VISUALIZAÇÃO
# ============================================================================

VISUALIZATION_CONFIG = {
    'style': 'seaborn-v0_8-darkgrid',
    'palette': 'husl',
    'figure_dpi': 100,
    'figure_size': (12, 6),
    'font_size': 12,
}

# ============================================================================
# CONFIGURAÇÕES DE CATEGORIAS E LOCALIZAÇÃO
# ============================================================================

CATEGORIES = {
    'Eletrônicos': {'base_price': 800, 'std': 400},
    'Roupas': {'base_price': 150, 'std': 80},
    'Livros': {'base_price': 50, 'std': 25},
    'Casa e Jardim': {'base_price': 200, 'std': 100},
    'Esportes': {'base_price': 180, 'std': 90},
    'Beleza': {'base_price': 120, 'std': 60},
    'Alimentos': {'base_price': 80, 'std': 40},
    'Brinquedos': {'base_price': 100, 'std': 50},
}

ESTADOS = {
    'SP': {'regiao': 'Sudeste', 'densidade': 1.3},
    'RJ': {'regiao': 'Sudeste', 'densidade': 1.2},
    'MG': {'regiao': 'Sudeste', 'densidade': 1.1},
    'RS': {'regiao': 'Sul', 'densidade': 1.15},
    'PR': {'regiao': 'Sul', 'densidade': 1.1},
    'SC': {'regiao': 'Sul', 'densidade': 1.1},
    'BA': {'regiao': 'Nordeste', 'densidade': 0.9},
    'PE': {'regiao': 'Nordeste', 'densidade': 0.85},
    'CE': {'regiao': 'Nordeste', 'densidade': 0.85},
    'GO': {'regiao': 'Centro-Oeste', 'densidade': 0.95},
    'DF': {'regiao': 'Centro-Oeste', 'densidade': 1.25},
    'AM': {'regiao': 'Norte', 'densidade': 0.8},
    'PA': {'regiao': 'Norte', 'densidade': 0.75},
}

SEGMENTOS = ['Bronze', 'Prata', 'Ouro', 'Platina']

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def get_config(config_name):
    """
    Retorna uma configuração específica.
    
    Args:
        config_name: Nome da configuração (data, feature, model, training, etc.)
    
    Returns:
        dict: Configuração solicitada
    """
    configs = {
        'data': DATA_CONFIG,
        'feature': FEATURE_CONFIG,
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'prediction': PREDICTION_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'categories': CATEGORIES,
        'estados': ESTADOS,
        'segmentos': SEGMENTOS,
    }
    
    return configs.get(config_name, {})

def print_config(config_name=None):
    """
    Imprime configurações de forma formatada.
    
    Args:
        config_name: Nome da configuração ou None para todas
    """
    if config_name:
        config = get_config(config_name)
        print(f"\n{'='*60}")
        print(f"CONFIGURAÇÃO: {config_name.upper()}")
        print('='*60)
        for key, value in config.items():
            print(f"{key:30s}: {value}")
    else:
        configs = ['data', 'feature', 'model', 'training', 'prediction']
        for name in configs:
            print_config(name)

if __name__ == "__main__":
    # Imprimir todas as configurações
    print("\n" + "="*70)
    print(" "*15 + "CONFIGURAÇÕES DO PROJETO")
    print("="*70)
    
    print_config('data')
    print_config('model')
    print_config('training')
    
    print("\n" + "="*70)
    print("Para usar estas configurações em seus scripts:")
    print("   from config import MODEL_CONFIG, TRAINING_CONFIG")
    print("="*70)
