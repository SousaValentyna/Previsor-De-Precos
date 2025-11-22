"""
Testes básicos para validar o funcionamento do projeto.
Execute: python tests.py
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

def test_1_data_generation():
    """Testa se os dados foram gerados corretamente"""
    print("\n" + "="*60)
    print("TEST 1: Verificando geração de dados")
    print("="*60)
    
    try:
        # Verificar se arquivos existem
        assert os.path.exists('data/raw/orders.csv'), "orders.csv não encontrado"
        assert os.path.exists('data/raw/customers.csv'), "customers.csv não encontrado"
        
        # Carregar dados
        orders = pd.read_csv('data/raw/orders.csv')
        customers = pd.read_csv('data/raw/customers.csv')
        
        # Verificar estrutura
        assert len(orders) > 0, "orders.csv está vazio"
        assert len(customers) > 0, "customers.csv está vazio"
        
        # Verificar colunas essenciais
        required_cols = ['order_id', 'customer_id', 'valor_pedido', 'categoria', 'estado']
        for col in required_cols:
            assert col in orders.columns, f"Coluna {col} não encontrada"
        
        # Verificar valores
        assert orders['valor_pedido'].min() > 0, "Valores de pedido inválidos"
        assert orders['valor_pedido'].isnull().sum() == 0, "Valores nulos encontrados"
        
        print("✅ PASSOU: Dados gerados corretamente")
        return True
        
    except AssertionError as e:
        print(f"❌ FALHOU: {e}")
        return False
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return False

def test_2_feature_engineering():
    """Testa o feature engineering"""
    print("\n" + "="*60)
    print("TEST 2: Verificando feature engineering")
    print("="*60)
    
    try:
        from src.features import FeatureEngineer
        
        # Carregar dados
        df = pd.read_csv('data/raw/orders.csv', parse_dates=['order_date'])
        
        # Aplicar feature engineering
        fe = FeatureEngineer()
        df_features = fe.fit_transform(df)
        
        # Verificar se features foram criadas
        assert df_features.shape[1] > df.shape[1], "Nenhuma feature nova criada"
        
        # Verificar features específicas
        expected_features = [
            'num_pedidos_anteriores',
            'valor_medio_anterior',
            'dia_semana_sin',
            'mes_sin',
            'categoria_preco_medio',
        ]
        
        for feature in expected_features:
            assert feature in df_features.columns, f"Feature {feature} não criada"
        
        # Verificar valores
        assert df_features['num_pedidos_anteriores'].min() >= 0, "Valores negativos em features"
        
        print(f"✅ PASSOU: {df_features.shape[1]} features criadas")
        return True
        
    except AssertionError as e:
        print(f"❌ FALHOU: {e}")
        return False
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return False

def test_3_model_training():
    """Testa se o modelo foi treinado"""
    print("\n" + "="*60)
    print("TEST 3: Verificando modelo treinado")
    print("="*60)
    
    try:
        # Verificar se arquivos do modelo existem
        assert os.path.exists('models/lightgbm_model.txt'), "Modelo não encontrado"
        assert os.path.exists('models/label_encoders.pkl'), "Label encoders não encontrados"
        assert os.path.exists('models/feature_engineer.pkl'), "Feature engineer não encontrado"
        assert os.path.exists('models/feature_names.json'), "Feature names não encontrado"
        
        # Carregar modelo
        import lightgbm as lgb
        model = lgb.Booster(model_file='models/lightgbm_model.txt')
        
        # Verificar propriedades do modelo
        assert model.num_trees() > 0, "Modelo sem árvores"
        
        print(f"✅ PASSOU: Modelo com {model.num_trees()} árvores carregado")
        return True
        
    except AssertionError as e:
        print(f"❌ FALHOU: {e}")
        print("💡 Execute: python src/train.py")
        return False
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return False

def test_4_prediction():
    """Testa predições"""
    print("\n" + "="*60)
    print("TEST 4: Verificando predições")
    print("="*60)
    
    try:
        from src.predict import PricePredictor
        
        # Criar preditor
        predictor = PricePredictor()
        
        # Dados de teste
        test_order = {
            'customer_id': 'TEST_001',
            'order_date': datetime.now().strftime('%Y-%m-%d'),
            'categoria': 'Eletrônicos',
            'estado': 'SP',
            'regiao': 'Sudeste',
            'idade_cliente': 30,
            'segmento_cliente': 'Ouro',
        }
        
        # Fazer predição
        prediction = predictor.predict(test_order)
        
        # Verificar resultado
        assert isinstance(prediction, (int, float)), "Predição não é numérica"
        assert prediction > 0, "Predição negativa"
        assert prediction < 10000, "Predição muito alta"
        
        print(f"✅ PASSOU: Predição = R$ {prediction:.2f}")
        return True
        
    except AssertionError as e:
        print(f"❌ FALHOU: {e}")
        return False
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return False

def test_5_metrics():
    """Testa se as métricas estão dentro do esperado"""
    print("\n" + "="*60)
    print("TEST 5: Verificando métricas do modelo")
    print("="*60)
    
    try:
        import json
        
        # Carregar métricas
        assert os.path.exists('models/metrics.json'), "Arquivo de métricas não encontrado"
        
        with open('models/metrics.json', 'r') as f:
            metrics = json.load(f)
        
        # Verificar estrutura
        assert 'test' in metrics, "Métricas de teste não encontradas"
        assert 'rmse' in metrics['test'], "RMSE não encontrado"
        assert 'r2' in metrics['test'], "R² não encontrado"
        
        # Verificar valores razoáveis
        test_metrics = metrics['test']
        
        assert test_metrics['rmse'] > 0, "RMSE inválido"
        assert 0 <= test_metrics['r2'] <= 1, "R² fora do range [0,1]"
        
        # Verificar se modelo não está muito ruim
        assert test_metrics['r2'] > 0.5, f"R² muito baixo: {test_metrics['r2']:.4f}"
        
        print(f"✅ PASSOU: RMSE={test_metrics['rmse']:.2f}, R²={test_metrics['r2']:.4f}")
        return True
        
    except AssertionError as e:
        print(f"❌ FALHOU: {e}")
        return False
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return False

def test_6_batch_prediction():
    """Testa predição em lote"""
    print("\n" + "="*60)
    print("TEST 6: Verificando predição em lote")
    print("="*60)
    
    try:
        from src.predict import PricePredictor
        
        predictor = PricePredictor()
        
        # Criar múltiplos pedidos
        test_orders = pd.DataFrame([
            {
                'customer_id': f'TEST_{i:03d}',
                'order_date': datetime.now().strftime('%Y-%m-%d'),
                'categoria': 'Eletrônicos',
                'estado': 'SP',
                'regiao': 'Sudeste',
                'idade_cliente': 25 + i,
                'segmento_cliente': 'Ouro',
            }
            for i in range(10)
        ])
        
        # Fazer predições
        predictions = predictor.predict(test_orders)
        
        # Verificar resultados
        assert len(predictions) == len(test_orders), "Número de predições incorreto"
        assert all(p > 0 for p in predictions), "Predições negativas"
        assert all(p < 10000 for p in predictions), "Predições muito altas"
        
        print(f"✅ PASSOU: {len(predictions)} predições realizadas")
        print(f"   Média: R$ {np.mean(predictions):.2f}")
        print(f"   Min: R$ {np.min(predictions):.2f}")
        print(f"   Max: R$ {np.max(predictions):.2f}")
        return True
        
    except AssertionError as e:
        print(f"❌ FALHOU: {e}")
        return False
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return False

def run_all_tests():
    """Executa todos os testes"""
    print("\n" + "="*70)
    print(" "*20 + "EXECUTANDO TESTES")
    print("="*70)
    
    tests = [
        ("Geração de Dados", test_1_data_generation),
        ("Feature Engineering", test_2_feature_engineering),
        ("Modelo Treinado", test_3_model_training),
        ("Predição Única", test_4_prediction),
        ("Métricas", test_5_metrics),
        ("Predição em Lote", test_6_batch_prediction),
    ]
    
    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
    
    # Resumo
    print("\n" + "="*70)
    print(" "*25 + "RESUMO DOS TESTES")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{name:30s} {status}")
    
    print("\n" + "="*70)
    print(f"Resultado Final: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 TODOS OS TESTES PASSARAM!")
    else:
        print("⚠️  Alguns testes falharam. Verifique os erros acima.")
    print("="*70)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
