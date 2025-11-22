"""
Script para gerar dados sintéticos de pedidos para treinamento do modelo.
Simula um e-commerce realista com histórico, categorias e localizações.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configurar seed para reprodutibilidade
np.random.seed(42)
random.seed(42)

# Configurações
NUM_CUSTOMERS = 1000
NUM_ORDERS = 10000
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2024, 11, 22)

# Categorias de produtos com preços base
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

# Estados brasileiros com regiões
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

def generate_customers():
    """Gera dados de clientes"""
    customers = []
    
    for i in range(NUM_CUSTOMERS):
        customer = {
            'customer_id': f'C{i:05d}',
            'estado': random.choice(list(ESTADOS.keys())),
            'idade': np.random.randint(18, 70),
            'segmento': random.choice(['Bronze', 'Prata', 'Ouro', 'Platina']),
        }
        customers.append(customer)
    
    return pd.DataFrame(customers)

def generate_orders(customers_df):
    """Gera dados de pedidos com base nos clientes"""
    orders = []
    
    for _ in range(NUM_ORDERS):
        # Selecionar cliente aleatório
        customer = customers_df.sample(1).iloc[0]
        
        # Gerar data do pedido
        days_diff = (END_DATE - START_DATE).days
        order_date = START_DATE + timedelta(days=random.randint(0, days_diff))
        
        # Selecionar categoria
        category = random.choice(list(CATEGORIES.keys()))
        
        # Calcular preço base
        cat_info = CATEGORIES[category]
        base_price = np.random.normal(cat_info['base_price'], cat_info['std'])
        
        # Aplicar modificadores
        # 1. Fator de localização
        estado_info = ESTADOS[customer['estado']]
        location_factor = estado_info['densidade']
        
        # 2. Fator de sazonalidade (Black Friday, Natal, etc.)
        seasonal_factor = 1.0
        if order_date.month == 11:  # Novembro (Black Friday)
            seasonal_factor = 0.8
        elif order_date.month == 12:  # Dezembro (Natal)
            seasonal_factor = 1.2
        elif order_date.month in [6, 7]:  # Inverno
            seasonal_factor = 1.05
        
        # 3. Fator de dia da semana
        weekday_factor = 1.0
        if order_date.weekday() in [5, 6]:  # Fim de semana
            weekday_factor = 1.1
        
        # 4. Fator de segmento do cliente
        segment_factors = {'Bronze': 0.8, 'Prata': 1.0, 'Ouro': 1.2, 'Platina': 1.5}
        segment_factor = segment_factors[customer['segmento']]
        
        # Preço final com todos os fatores
        final_price = base_price * location_factor * seasonal_factor * weekday_factor * segment_factor
        
        # Adicionar ruído
        final_price += np.random.normal(0, 20)
        final_price = max(10, final_price)  # Garantir preço mínimo
        
        order = {
            'order_id': f'O{len(orders):06d}',
            'customer_id': customer['customer_id'],
            'order_date': order_date,
            'categoria': category,
            'valor_pedido': round(final_price, 2),
            'estado': customer['estado'],
            'regiao': estado_info['regiao'],
            'idade_cliente': customer['idade'],
            'segmento_cliente': customer['segmento'],
        }
        orders.append(order)
    
    orders_df = pd.DataFrame(orders)
    orders_df = orders_df.sort_values('order_date').reset_index(drop=True)
    
    return orders_df

def add_temporal_features(df):
    """Adiciona features temporais"""
    df['dia_semana'] = df['order_date'].dt.dayofweek
    df['mes'] = df['order_date'].dt.month
    df['trimestre'] = df['order_date'].dt.quarter
    df['ano'] = df['order_date'].dt.year
    df['dia_mes'] = df['order_date'].dt.day
    df['fim_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
    
    return df

def main():
    """Função principal para gerar todos os dados"""
    print("Gerando dados sintéticos...")
    
    # Gerar clientes
    print("1. Gerando clientes...")
    customers_df = generate_customers()
    
    # Gerar pedidos
    print("2. Gerando pedidos...")
    orders_df = generate_orders(customers_df)
    
    # Adicionar features temporais
    print("3. Adicionando features temporais...")
    orders_df = add_temporal_features(orders_df)
    
    # Salvar dados
    print("4. Salvando dados...")
    customers_df.to_csv('data/raw/customers.csv', index=False)
    orders_df.to_csv('data/raw/orders.csv', index=False)
    
    # Estatísticas
    print("\n" + "="*50)
    print("DADOS GERADOS COM SUCESSO!")
    print("="*50)
    print(f"\nClientes: {len(customers_df):,}")
    print(f"Pedidos: {len(orders_df):,}")
    print(f"\nPeríodo: {orders_df['order_date'].min().date()} a {orders_df['order_date'].max().date()}")
    print(f"\nValor médio dos pedidos: R$ {orders_df['valor_pedido'].mean():.2f}")
    print(f"Valor mínimo: R$ {orders_df['valor_pedido'].min():.2f}")
    print(f"Valor máximo: R$ {orders_df['valor_pedido'].max():.2f}")
    print(f"\nCategorias: {orders_df['categoria'].nunique()}")
    print(f"Estados: {orders_df['estado'].nunique()}")
    print(f"Regiões: {orders_df['regiao'].nunique()}")
    
    print("\nDistribuição por categoria:")
    print(orders_df['categoria'].value_counts())
    
    print("\nDistribuição por região:")
    print(orders_df['regiao'].value_counts())
    
    print("\n[OK] Arquivos salvos em data/raw/")

if __name__ == "__main__":
    main()
