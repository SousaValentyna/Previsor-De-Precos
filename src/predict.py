"""
Script para fazer predições de preço médio de pedidos usando o modelo treinado.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PricePredictor:
    """Classe para fazer predições de preço"""
    
    def __init__(self, model_path='models/lightgbm_model.txt'):
        """Inicializa o preditor carregando o modelo e artefatos"""
        print("Carregando modelo e artefatos...")
        
        # Carregar modelo
        self.model = lgb.Booster(model_file=model_path)
        print("[OK] Modelo carregado")
        
        # Carregar label encoders
        self.label_encoders = joblib.load('models/label_encoders.pkl')
        print("[OK] Label encoders carregados")
        
        # Carregar estatísticas das features
        self.feature_stats = joblib.load('models/feature_stats.pkl')
        print("[OK] Estatísticas carregadas")
        
        # Carregar nomes das features
        with open('models/feature_names.json', 'r') as f:
            self.feature_names = json.load(f)
        print(f"[OK] {len(self.feature_names)} features carregadas")
        
        print("\n[OK] Preditor pronto para uso!\n")
    
    def prepare_input(self, input_data):
        """Prepara os dados de entrada para predição"""
        # Converter para DataFrame se necessário
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Garantir que order_date é datetime
        if 'order_date' in df.columns:
            df['order_date'] = pd.to_datetime(df['order_date'])
        
        # Aplicar feature engineering
        # Nota: Para predição real, precisaríamos do histórico do cliente
        # Aqui vamos simular com valores padrão para novos clientes
        
        # Adicionar features temporais básicas
        if 'order_date' in df.columns:
            df['dia_semana'] = df['order_date'].dt.dayofweek
            df['mes'] = df['order_date'].dt.month
            df['trimestre'] = df['order_date'].dt.quarter
            df['ano'] = df['order_date'].dt.year
            df['dia_mes'] = df['order_date'].dt.day
            df['fim_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
            
            # Features temporais avançadas
            df['inicio_mes'] = (df['dia_mes'] <= 10).astype(int)
            df['meio_mes'] = ((df['dia_mes'] > 10) & (df['dia_mes'] <= 20)).astype(int)
            df['fim_mes'] = (df['dia_mes'] > 20).astype(int)
            df['black_friday'] = ((df['mes'] == 11) & (df['dia_mes'] >= 20)).astype(int)
            df['natal'] = ((df['mes'] == 12) & (df['dia_mes'] >= 15)).astype(int)
            df['ano_novo'] = ((df['mes'] == 1) & (df['dia_mes'] <= 7)).astype(int)
            
            # Features cíclicas
            df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
            df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
            df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
            df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
        
        # Features históricas padrão para novos clientes
        historical_features = {
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
        
        for feature, default_value in historical_features.items():
            if feature not in df.columns:
                df[feature] = default_value
        
        # Adicionar features de categoria
        category_stats = self.feature_stats['category_stats']
        df = df.merge(category_stats, on='categoria', how='left')
        
        # Se categoria não encontrada, usar média geral
        df['categoria_preco_medio'].fillna(df['categoria_preco_medio'].mean(), inplace=True)
        df['categoria_preco_std'].fillna(df['categoria_preco_std'].mean(), inplace=True)
        df['categoria_popularidade'].fillna(0, inplace=True)
        
        # Criar ratio (usando valor médio como proxy se não houver valor real)
        df['ratio_preco_categoria'] = df.get('valor_pedido', df['categoria_preco_medio']) / df['categoria_preco_medio']
        
        # Adicionar features de localização
        estado_stats = self.feature_stats['estado_stats']
        df = df.merge(estado_stats, on='estado', how='left')
        
        region_stats = self.feature_stats['region_stats']
        df = df.merge(region_stats, on='regiao', how='left')
        
        # Preencher valores faltantes
        df['estado_preco_medio'].fillna(df['estado_preco_medio'].mean(), inplace=True)
        df['estado_preco_std'].fillna(df['estado_preco_std'].mean(), inplace=True)
        df['estado_num_pedidos'].fillna(0, inplace=True)
        df['regiao_preco_medio'].fillna(df['regiao_preco_medio'].mean(), inplace=True)
        df['regiao_preco_std'].fillna(df['regiao_preco_std'].mean(), inplace=True)
        
        # Adicionar features de interação ANTES do label encoding
        df['idade_x_num_pedidos'] = df['idade_cliente'] * df['num_pedidos_anteriores']
        df['categoria_x_regiao'] = df['categoria'].astype(str) + '_' + df['regiao'].astype(str)
        df['fim_semana_x_categoria'] = df['fim_semana'].astype(str) + '_' + df['categoria'].astype(str)
        df['segmento_x_categoria'] = df['segmento_cliente'].astype(str) + '_' + df['categoria'].astype(str)
        
        # Aplicar label encoding
        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        # Garantir que todas as features necessárias existem
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Selecionar apenas as features do modelo na ordem correta
        df = df[self.feature_names]
        
        return df
    
    def predict(self, input_data):
        """Faz predição do preço"""
        # Preparar dados
        X = self.prepare_input(input_data)
        
        # Fazer predição
        prediction = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        return prediction[0] if len(prediction) == 1 else prediction
    
    def predict_batch(self, csv_path):
        """Faz predições em lote a partir de um CSV"""
        print(f"Carregando dados de {csv_path}...")
        df = pd.read_csv(csv_path, parse_dates=['order_date'] if 'order_date' in pd.read_csv(csv_path, nrows=1).columns else None)
        
        print(f"Fazendo predições para {len(df)} registros...")
        predictions = self.predict(df)
        
        df['predicted_price'] = predictions
        
        return df

def example_single_prediction():
    """Exemplo de predição única"""
    print("\n" + "="*60)
    print(" " * 15 + "EXEMPLO: PREDIÇÃO ÚNICA")
    print("="*60 + "\n")
    
    # Criar preditor
    predictor = PricePredictor()
    
    # Dados de entrada para predição
    new_order = {
        'customer_id': 'C00001',
        'order_date': '2024-11-22',
        'categoria': 'Eletrônicos',
        'estado': 'SP',
        'regiao': 'Sudeste',
        'idade_cliente': 35,
        'segmento_cliente': 'Ouro',
    }
    
    print("Dados do pedido:")
    for key, value in new_order.items():
        print(f"  {key}: {value}")
    
    # Fazer predição
    predicted_price = predictor.predict(new_order)
    
    print("\n" + "="*60)
    print(f"PREÇO PREVISTO: R$ {predicted_price:.2f}")
    print("="*60)

def example_batch_prediction():
    """Exemplo de predição em lote"""
    print("\n" + "="*60)
    print(" " * 15 + "EXEMPLO: PREDIÇÃO EM LOTE")
    print("="*60 + "\n")
    
    # Criar dados de exemplo
    sample_orders = pd.DataFrame([
        {
            'customer_id': 'C00001',
            'order_date': '2024-11-22',
            'categoria': 'Eletrônicos',
            'estado': 'SP',
            'regiao': 'Sudeste',
            'idade_cliente': 35,
            'segmento_cliente': 'Ouro',
        },
        {
            'customer_id': 'C00002',
            'order_date': '2024-11-22',
            'categoria': 'Roupas',
            'estado': 'RJ',
            'regiao': 'Sudeste',
            'idade_cliente': 28,
            'segmento_cliente': 'Prata',
        },
        {
            'customer_id': 'C00003',
            'order_date': '2024-11-22',
            'categoria': 'Livros',
            'estado': 'RS',
            'regiao': 'Sul',
            'idade_cliente': 42,
            'segmento_cliente': 'Bronze',
        },
        {
            'customer_id': 'C00004',
            'order_date': '2024-12-25',
            'categoria': 'Brinquedos',
            'estado': 'MG',
            'regiao': 'Sudeste',
            'idade_cliente': 31,
            'segmento_cliente': 'Platina',
        },
    ])
    
    # Criar preditor
    predictor = PricePredictor()
    
    # Fazer predições
    results = predictor.predict(sample_orders)
    
    # Adicionar predições ao DataFrame
    sample_orders['predicted_price'] = results
    
    print("Resultados das predições:")
    print("-" * 60)
    for idx, row in sample_orders.iterrows():
        print(f"\nPedido {idx + 1}:")
        print(f"  Cliente: {row['customer_id']}")
        print(f"  Categoria: {row['categoria']}")
        print(f"  Estado: {row['estado']}")
        print(f"  Segmento: {row['segmento_cliente']}")
        print(f"  Preço Previsto: R$ {row['predicted_price']:.2f}")
    
    # Estatísticas
    print("\n" + "="*60)
    print("ESTATÍSTICAS DAS PREDIÇÕES")
    print("="*60)
    print(f"Preço médio previsto: R$ {sample_orders['predicted_price'].mean():.2f}")
    print(f"Preço mínimo previsto: R$ {sample_orders['predicted_price'].min():.2f}")
    print(f"Preço máximo previsto: R$ {sample_orders['predicted_price'].max():.2f}")

def interactive_prediction():
    """Modo interativo para fazer predições"""
    print("\n" + "="*60)
    print(" " * 15 + "MODO INTERATIVO")
    print("="*60 + "\n")
    
    # Criar preditor
    predictor = PricePredictor()
    
    print("Digite as informações do pedido:")
    print("-" * 60)
    
    # Coletar inputs
    categoria = input("Categoria (Eletrônicos/Roupas/Livros/Casa e Jardim/Esportes/Beleza/Alimentos/Brinquedos): ")
    estado = input("Estado (ex: SP, RJ, MG, RS, etc.): ")
    segmento = input("Segmento do Cliente (Bronze/Prata/Ouro/Platina): ")
    idade = int(input("Idade do Cliente: "))
    
    # Mapear região
    regioes = {
        'SP': 'Sudeste', 'RJ': 'Sudeste', 'MG': 'Sudeste', 'ES': 'Sudeste',
        'RS': 'Sul', 'PR': 'Sul', 'SC': 'Sul',
        'BA': 'Nordeste', 'PE': 'Nordeste', 'CE': 'Nordeste', 'RN': 'Nordeste',
        'GO': 'Centro-Oeste', 'DF': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste',
        'AM': 'Norte', 'PA': 'Norte', 'RO': 'Norte', 'AC': 'Norte'
    }
    
    regiao = regioes.get(estado.upper(), 'Sudeste')
    
    # Criar dados do pedido
    new_order = {
        'customer_id': 'NEW_CUSTOMER',
        'order_date': datetime.now().strftime('%Y-%m-%d'),
        'categoria': categoria,
        'estado': estado.upper(),
        'regiao': regiao,
        'idade_cliente': idade,
        'segmento_cliente': segmento,
    }
    
    # Fazer predição
    predicted_price = predictor.predict(new_order)
    
    print("\n" + "="*60)
    print(f"PREÇO PREVISTO: R$ {predicted_price:.2f}")
    print("="*60)

def main():
    """Função principal"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--interactive':
            interactive_prediction()
        elif sys.argv[1] == '--batch':
            if len(sys.argv) < 3:
                print("Uso: python src/predict.py --batch <caminho_do_csv>")
                return
            predictor = PricePredictor()
            results = predictor.predict_batch(sys.argv[2])
            output_path = 'predictions.csv'
            results.to_csv(output_path, index=False)
            print(f"\n[OK] Predições salvas em {output_path}")
        else:
            print("Opções: --interactive | --batch <arquivo.csv>")
    else:
        # Modo padrão: exemplos
        example_single_prediction()
        example_batch_prediction()
        
        print("\n" + "="*60)
        print("OUTROS MODOS DE USO:")
        print("="*60)
        print("Modo interativo:")
        print("  python src/predict.py --interactive")
        print("\nPredição em lote:")
        print("  python src/predict.py --batch caminho/para/arquivo.csv")

if __name__ == "__main__":
    main()
