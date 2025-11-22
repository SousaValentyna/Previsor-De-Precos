"""
Módulo de Feature Engineering para previsão de preço médio de pedidos.
Cria features históricas, temporais e geográficas.
"""

import pandas as pd
import numpy as np
from datetime import timedelta

class FeatureEngineer:
    """Classe para criar e transformar features"""
    
    def __init__(self):
        self.category_stats = {}
        self.region_stats = {}
        self.estado_stats = {}
    
    def create_historical_features(self, df):
        """
        Cria features baseadas no histórico de compras do cliente.
        Features RFM (Recency, Frequency, Monetary).
        """
        print("Criando features históricas...")
        df = df.sort_values(['customer_id', 'order_date']).reset_index(drop=True)
        
        # Para cada pedido, calcular estatísticas dos pedidos ANTERIORES do cliente
        historical_features = []
        
        for customer_id in df['customer_id'].unique():
            customer_orders = df[df['customer_id'] == customer_id].copy()
            
            for idx, row in customer_orders.iterrows():
                # Pedidos anteriores a este
                previous_orders = customer_orders[customer_orders['order_date'] < row['order_date']]
                
                if len(previous_orders) == 0:
                    # Primeiro pedido do cliente
                    features = {
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
                    }
                else:
                    # Calcular features históricas
                    last_order_date = previous_orders['order_date'].max()
                    days_since_last = (row['order_date'] - last_order_date).days
                    
                    # Pedidos recentes
                    date_30_days_ago = row['order_date'] - timedelta(days=30)
                    date_90_days_ago = row['order_date'] - timedelta(days=90)
                    
                    recent_30 = previous_orders[previous_orders['order_date'] >= date_30_days_ago]
                    recent_90 = previous_orders[previous_orders['order_date'] >= date_90_days_ago]
                    
                    # Tendência de gasto (últimos 3 vs primeiros 3 pedidos)
                    if len(previous_orders) >= 6:
                        recent_3 = previous_orders.tail(3)['valor_pedido'].mean()
                        first_3 = previous_orders.head(3)['valor_pedido'].mean()
                        tendencia = (recent_3 - first_3) / first_3 if first_3 > 0 else 0
                    else:
                        tendencia = 0
                    
                    features = {
                        'num_pedidos_anteriores': len(previous_orders),
                        'valor_medio_anterior': previous_orders['valor_pedido'].mean(),
                        'valor_total_anterior': previous_orders['valor_pedido'].sum(),
                        'dias_desde_ultimo_pedido': days_since_last,
                        'pedidos_ultimos_30_dias': len(recent_30),
                        'pedidos_ultimos_90_dias': len(recent_90),
                        'valor_medio_ultimos_30_dias': recent_30['valor_pedido'].mean() if len(recent_30) > 0 else 0,
                        'std_valor_anterior': previous_orders['valor_pedido'].std() if len(previous_orders) > 1 else 0,
                        'max_valor_anterior': previous_orders['valor_pedido'].max(),
                        'min_valor_anterior': previous_orders['valor_pedido'].min(),
                        'tendencia_gasto': tendencia,
                    }
                
                features['order_id'] = row['order_id']
                historical_features.append(features)
        
        historical_df = pd.DataFrame(historical_features)
        df = df.merge(historical_df, on='order_id', how='left')
        
        return df
    
    def create_category_features(self, df):
        """Cria features baseadas na categoria do produto"""
        print("Criando features de categoria...")
        
        # Preço médio por categoria
        category_avg = df.groupby('categoria')['valor_pedido'].agg(['mean', 'std', 'count']).reset_index()
        category_avg.columns = ['categoria', 'categoria_preco_medio', 'categoria_preco_std', 'categoria_popularidade']
        
        df = df.merge(category_avg, on='categoria', how='left')
        
        # Proporção do pedido em relação à média da categoria
        df['ratio_preco_categoria'] = df['valor_pedido'] / df['categoria_preco_medio']
        
        # Salvar para uso em predict
        self.category_stats = category_avg
        
        return df
    
    def create_location_features(self, df):
        """Cria features baseadas na localização"""
        print("Criando features de localização...")
        
        # Preço médio por estado
        estado_avg = df.groupby('estado')['valor_pedido'].agg(['mean', 'std', 'count']).reset_index()
        estado_avg.columns = ['estado', 'estado_preco_medio', 'estado_preco_std', 'estado_num_pedidos']
        
        df = df.merge(estado_avg, on='estado', how='left')
        
        # Preço médio por região
        regiao_avg = df.groupby('regiao')['valor_pedido'].agg(['mean', 'std']).reset_index()
        regiao_avg.columns = ['regiao', 'regiao_preco_medio', 'regiao_preco_std']
        
        df = df.merge(regiao_avg, on='regiao', how='left')
        
        # Salvar para uso em predict
        self.estado_stats = estado_avg
        self.region_stats = regiao_avg
        
        return df
    
    def create_temporal_features(self, df):
        """Cria features temporais avançadas"""
        print("Criando features temporais...")
        
        # Já temos dia_semana, mes, trimestre do generate_data
        # Vamos adicionar mais features
        
        # É início/meio/fim do mês?
        df['inicio_mes'] = (df['dia_mes'] <= 10).astype(int)
        df['meio_mes'] = ((df['dia_mes'] > 10) & (df['dia_mes'] <= 20)).astype(int)
        df['fim_mes'] = (df['dia_mes'] > 20).astype(int)
        
        # Sazonalidade
        df['black_friday'] = ((df['mes'] == 11) & (df['dia_mes'] >= 20)).astype(int)
        df['natal'] = ((df['mes'] == 12) & (df['dia_mes'] >= 15)).astype(int)
        df['ano_novo'] = ((df['mes'] == 1) & (df['dia_mes'] <= 7)).astype(int)
        
        # Features cíclicas para dia da semana e mês
        df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
        df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
        
        return df
    
    def create_interaction_features(self, df):
        """Cria features de interação entre variáveis"""
        print("Criando features de interação...")
        
        # Interações importantes
        df['idade_x_num_pedidos'] = df['idade_cliente'] * df['num_pedidos_anteriores']
        df['categoria_x_regiao'] = df['categoria'] + '_' + df['regiao']
        df['fim_semana_x_categoria'] = df['fim_semana'].astype(str) + '_' + df['categoria']
        
        # Segmento x Categoria
        df['segmento_x_categoria'] = df['segmento_cliente'] + '_' + df['categoria']
        
        return df
    
    def fit_transform(self, df):
        """Aplica todas as transformações de feature engineering"""
        print("\n" + "="*50)
        print("INICIANDO FEATURE ENGINEERING")
        print("="*50)
        
        df = df.copy()
        
        # Aplicar todas as transformações
        df = self.create_historical_features(df)
        df = self.create_category_features(df)
        df = self.create_location_features(df)
        df = self.create_temporal_features(df)
        df = self.create_interaction_features(df)
        
        print("\n[OK] Feature Engineering concluído!")
        print(f"Total de features: {df.shape[1]}")
        
        return df
    
    def transform(self, df):
        """
        Aplica transformações em novos dados (para predição).
        Usa estatísticas calculadas no fit_transform.
        """
        df = df.copy()
        
        # Aplicar features que não dependem de estatísticas globais
        df = self.create_temporal_features(df)
        
        # Aplicar features que usam estatísticas pré-calculadas
        if self.category_stats is not None:
            df = df.merge(self.category_stats, on='categoria', how='left')
        
        if self.estado_stats is not None:
            df = df.merge(self.estado_stats, on='estado', how='left')
        
        if self.region_stats is not None:
            df = df.merge(self.region_stats, on='regiao', how='left')
        
        df = self.create_interaction_features(df)
        
        return df

def main():
    """Teste do módulo de feature engineering"""
    print("Carregando dados...")
    orders_df = pd.read_csv('data/raw/orders.csv', parse_dates=['order_date'])
    
    # Criar features
    fe = FeatureEngineer()
    orders_with_features = fe.fit_transform(orders_df)
    
    # Salvar
    print("\nSalvando dados processados...")
    orders_with_features.to_csv('data/processed/orders_with_features.csv', index=False)
    
    print("\n" + "="*50)
    print("DADOS PROCESSADOS COM SUCESSO!")
    print("="*50)
    print(f"\nShape: {orders_with_features.shape}")
    print(f"\nPrimeiras features criadas:")
    feature_cols = [col for col in orders_with_features.columns if col not in orders_df.columns]
    for col in feature_cols[:10]:
        print(f"  - {col}")
    print(f"  ... e mais {len(feature_cols) - 10} features")
    
    print("\n[OK] Arquivo salvo em data/processed/orders_with_features.csv")

if __name__ == "__main__":
    main()
