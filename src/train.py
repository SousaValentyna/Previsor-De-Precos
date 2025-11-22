"""
Script de treinamento do modelo LightGBM para previsão de preço médio de pedidos.
Inclui feature engineering, validação cruzada e otimização de hiperparâmetros.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import sys
import os

# Adicionar diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.features import FeatureEngineer

class ModelTrainer:
    """Classe para treinar modelo de previsão de preço"""
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_engineer = FeatureEngineer()
        self.feature_names = None
        
    def load_and_prepare_data(self):
        """Carrega e prepara os dados"""
        print("\n" + "="*50)
        print("CARREGANDO E PREPARANDO DADOS")
        print("="*50)
        
        # Carregar dados brutos
        print("\n1. Carregando dados brutos...")
        df = pd.read_csv('data/raw/orders.csv', parse_dates=['order_date'])
        print(f"   [OK] {len(df):,} pedidos carregados")
        
        # Aplicar feature engineering
        print("\n2. Aplicando feature engineering...")
        df = self.feature_engineer.fit_transform(df)
        print(f"   ✓ {df.shape[1]} features criadas")
        
        # Salvar dados processados
        df.to_csv('data/processed/orders_with_features.csv', index=False)
        print("   ✓ Dados processados salvos")
        
        return df
    
    def prepare_features(self, df, is_training=True):
        """Prepara features para o modelo"""
        print("\n3. Preparando features para modelagem...")
        
        # Separar target
        target = df['valor_pedido'].copy()
        
        # Remover colunas não utilizadas
        drop_cols = [
            'order_id', 'customer_id', 'order_date', 'valor_pedido',
            'ano_mes',  # Se existir da EDA
        ]
        
        # Remover apenas colunas que existem
        drop_cols = [col for col in drop_cols if col in df.columns]
        features_df = df.drop(columns=drop_cols)
        
        # Identificar colunas categóricas
        categorical_cols = features_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Label encoding para categóricas
        for col in categorical_cols:
            if is_training:
                le = LabelEncoder()
                features_df[col] = le.fit_transform(features_df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Tratar valores novos
                    features_df[col] = features_df[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        
        # Salvar nomes das features
        if is_training:
            self.feature_names = features_df.columns.tolist()
        
        print(f"   [OK] {len(self.feature_names)} features preparadas")
        print(f"   [OK] {len(categorical_cols)} features categóricas encodadas")
        
        return features_df, target, categorical_cols
    
    def train_model(self, X_train, y_train, X_val, y_val, categorical_features):
        """Treina o modelo LightGBM"""
        print("\n" + "="*50)
        print("TREINANDO MODELO LIGHTGBM")
        print("="*50)
        
        # Parâmetros do LightGBM
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': -1,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbose': -1,
        }
        
        # Criar datasets LightGBM
        categorical_indices = [X_train.columns.get_loc(col) for col in categorical_features if col in X_train.columns]
        
        train_data = lgb.Dataset(
            X_train, 
            label=y_train,
            categorical_feature=categorical_indices,
            free_raw_data=False
        )
        
        val_data = lgb.Dataset(
            X_val, 
            label=y_val,
            categorical_feature=categorical_indices,
            reference=train_data,
            free_raw_data=False
        )
        
        # Treinar modelo
        print("\nTreinando...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )
        
        print(f"\n[OK] Modelo treinado com {self.model.best_iteration} iterações")
        
        return self.model
    
    def evaluate_model(self, X, y, dataset_name=''):
        """Avalia o modelo"""
        predictions = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        rmse = np.sqrt(mean_squared_error(y, predictions))
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        mape = np.mean(np.abs((y - predictions) / y)) * 100
        
        print(f"\n{dataset_name} Métricas:")
        print(f"  RMSE: R$ {rmse:.2f}")
        print(f"  MAE:  R$ {mae:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²:   {r2:.4f}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
    
    def cross_validate(self, X, y, categorical_features, n_splits=5):
        """Validação cruzada"""
        print("\n" + "="*50)
        print("VALIDAÇÃO CRUZADA")
        print("="*50)
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        cv_scores = {
            'rmse': [],
            'mae': [],
            'mape': [],
            'r2': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"\nFold {fold}/{n_splits}")
            
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Treinar modelo temporário
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': -1,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'verbose': -1,
            }
            
            categorical_indices = [X.columns.get_loc(col) for col in categorical_features if col in X.columns]
            
            train_data = lgb.Dataset(
                X_train_fold, 
                label=y_train_fold,
                categorical_feature=categorical_indices
            )
            
            val_data = lgb.Dataset(
                X_val_fold, 
                label=y_val_fold,
                categorical_feature=categorical_indices,
                reference=train_data
            )
            
            model_fold = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                ]
            )
            
            # Avaliar
            predictions = model_fold.predict(X_val_fold, num_iteration=model_fold.best_iteration)
            
            rmse = np.sqrt(mean_squared_error(y_val_fold, predictions))
            mae = mean_absolute_error(y_val_fold, predictions)
            r2 = r2_score(y_val_fold, predictions)
            mape = np.mean(np.abs((y_val_fold - predictions) / y_val_fold)) * 100
            
            cv_scores['rmse'].append(rmse)
            cv_scores['mae'].append(mae)
            cv_scores['mape'].append(mape)
            cv_scores['r2'].append(r2)
            
            print(f"  RMSE: R$ {rmse:.2f} | MAE: R$ {mae:.2f} | R²: {r2:.4f}")
        
        # Resultados finais
        print("\n" + "="*50)
        print("RESULTADOS DA VALIDAÇÃO CRUZADA")
        print("="*50)
        print(f"RMSE: R$ {np.mean(cv_scores['rmse']):.2f} ± {np.std(cv_scores['rmse']):.2f}")
        print(f"MAE:  R$ {np.mean(cv_scores['mae']):.2f} ± {np.std(cv_scores['mae']):.2f}")
        print(f"MAPE: {np.mean(cv_scores['mape']):.2f}% ± {np.std(cv_scores['mape']):.2f}%")
        print(f"R²:   {np.mean(cv_scores['r2']):.4f} ± {np.std(cv_scores['r2']):.4f}")
        
        return cv_scores
    
    def plot_feature_importance(self, top_n=20):
        """Plota importância das features"""
        print("\n" + "="*50)
        print(f"TOP {top_n} FEATURES MAIS IMPORTANTES")
        print("="*50)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        for idx, row in importance_df.iterrows():
            print(f"{row['feature']:<40} {row['importance']:>10.0f}")
        
        return importance_df
    
    def save_model(self):
        """Salva o modelo e artefatos"""
        print("\n" + "="*50)
        print("SALVANDO MODELO E ARTEFATOS")
        print("="*50)
        
        # Salvar modelo LightGBM
        self.model.save_model('models/lightgbm_model.txt')
        print("[OK] Modelo LightGBM salvo")
        
        # Salvar label encoders
        joblib.dump(self.label_encoders, 'models/label_encoders.pkl')
        print("[OK] Label encoders salvos")
        
        # Salvar feature engineer
        joblib.dump(self.feature_engineer, 'models/feature_engineer.pkl')
        print("[OK] Feature engineer salvo")
        
        # Salvar estatísticas para predição
        stats = {
            'category_stats': self.feature_engineer.category_stats,
            'estado_stats': self.feature_engineer.estado_stats,
            'region_stats': self.feature_engineer.region_stats,
        }
        joblib.dump(stats, 'models/feature_stats.pkl')
        print("[OK] Estatísticas das features salvas")
        
        # Salvar nomes das features
        with open('models/feature_names.json', 'w') as f:
            json.dump(self.feature_names, f)
        print("[OK] Nomes das features salvos")
        
        print("\n[OK] Todos os artefatos salvos em models/")

def main():
    """Função principal de treinamento"""
    print("\n" + "="*70)
    print(" " * 15 + "TREINAMENTO DO MODELO")
    print("="*70)
    
    # Criar instância do trainer
    trainer = ModelTrainer()
    
    # 1. Carregar e preparar dados
    df = trainer.load_and_prepare_data()
    
    # 2. Preparar features
    X, y, categorical_features = trainer.prepare_features(df, is_training=True)
    
    # 3. Split treino/teste
    print("\n4. Dividindo dados em treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"   [OK] Treino: {len(X_train):,} | Teste: {len(X_test):,}")
    
    # 4. Treinar modelo
    trainer.train_model(X_train, y_train, X_test, y_test, categorical_features)
    
    # 5. Avaliar modelo
    print("\n" + "="*50)
    print("AVALIAÇÃO DO MODELO")
    print("="*50)
    
    train_metrics = trainer.evaluate_model(X_train, y_train, "TREINO")
    test_metrics = trainer.evaluate_model(X_test, y_test, "TESTE")
    
    # 6. Validação cruzada
    cv_scores = trainer.cross_validate(X_train, y_train, categorical_features, n_splits=5)
    
    # 7. Feature importance
    importance_df = trainer.plot_feature_importance(top_n=20)
    
    # 8. Salvar modelo
    trainer.save_model()
    
    # Salvar métricas
    metrics = {
        'train': train_metrics,
        'test': test_metrics,
        'cross_validation': {
            'rmse_mean': float(np.mean(cv_scores['rmse'])),
            'rmse_std': float(np.std(cv_scores['rmse'])),
            'mae_mean': float(np.mean(cv_scores['mae'])),
            'mae_std': float(np.std(cv_scores['mae'])),
            'r2_mean': float(np.mean(cv_scores['r2'])),
            'r2_std': float(np.std(cv_scores['r2'])),
        }
    }
    
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "="*70)
    print(" " * 20 + "TREINAMENTO CONCLUÍDO!")
    print("="*70)
    print("\n[OK] Modelo treinado e salvo com sucesso!")
    print("[OK] Execute 'python src/predict.py' para fazer predições")

if __name__ == "__main__":
    main()
