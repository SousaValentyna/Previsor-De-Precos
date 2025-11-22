"""
Exemplos de uso do modelo de predição de preço.
Este arquivo demonstra diferentes cenários de uso.
"""

from src.predict import PricePredictor
import pandas as pd
from datetime import datetime, timedelta

def example_1_new_customer():
    """Exemplo 1: Predição para novo cliente"""
    print("\n" + "="*70)
    print(" "*15 + "EXEMPLO 1: NOVO CLIENTE")
    print("="*70 + "\n")
    
    predictor = PricePredictor()
    
    # Cliente novo fazendo primeiro pedido
    pedido = {
        'customer_id': 'NEW_001',
        'order_date': datetime.now().strftime('%Y-%m-%d'),
        'categoria': 'Eletrônicos',
        'estado': 'SP',
        'regiao': 'Sudeste',
        'idade_cliente': 28,
        'segmento_cliente': 'Bronze',  # Cliente novo começa no Bronze
    }
    
    preco_previsto = predictor.predict(pedido)
    
    print("📋 Cenário: Cliente novo comprando Eletrônicos em SP")
    print(f"Preço Previsto: R$ {preco_previsto:.2f}")
    print("\nInsight: Clientes novos tendem a ter pedidos menores.")

def example_2_premium_customer():
    """Exemplo 2: Cliente premium"""
    print("\n" + "="*70)
    print(" "*15 + "EXEMPLO 2: CLIENTE PREMIUM")
    print("="*70 + "\n")
    
    predictor = PricePredictor()
    
    # Cliente Platina fazendo compra de alto valor
    pedido = {
        'customer_id': 'PREMIUM_001',
        'order_date': datetime.now().strftime('%Y-%m-%d'),
        'categoria': 'Eletrônicos',
        'estado': 'SP',
        'regiao': 'Sudeste',
        'idade_cliente': 45,
        'segmento_cliente': 'Platina',  # Cliente top
    }
    
    preco_previsto = predictor.predict(pedido)
    
    print("📋 Cenário: Cliente Platina comprando Eletrônicos em SP")
    print(f"Preço Previsto: R$ {preco_previsto:.2f}")
    print("\nInsight: Clientes Platina gastam significativamente mais.")

def example_3_seasonal_comparison():
    """Exemplo 3: Comparação sazonal (Black Friday vs Normal)"""
    print("\n" + "="*70)
    print(" "*15 + "EXEMPLO 3: COMPARAÇÃO SAZONAL")
    print("="*70 + "\n")
    
    predictor = PricePredictor()
    
    # Pedido em dia normal
    pedido_normal = {
        'customer_id': 'C00100',
        'order_date': '2024-08-15',  # Agosto (mês normal)
        'categoria': 'Eletrônicos',
        'estado': 'SP',
        'regiao': 'Sudeste',
        'idade_cliente': 35,
        'segmento_cliente': 'Ouro',
    }
    
    # Pedido na Black Friday
    pedido_blackfriday = pedido_normal.copy()
    pedido_blackfriday['order_date'] = '2024-11-29'  # Black Friday
    
    preco_normal = predictor.predict(pedido_normal)
    preco_blackfriday = predictor.predict(pedido_blackfriday)
    
    diferenca = preco_normal - preco_blackfriday
    percentual = (diferenca / preco_normal) * 100
    
    print("📋 Cenário: Mesmo pedido em diferentes épocas")
    print(f"\n📅 Agosto (Normal):")
    print(f"   Preço Previsto: R$ {preco_normal:.2f}")
    
    print(f"\nBlack Friday:")
    print(f"   Preço Previsto: R$ {preco_blackfriday:.2f}")
    
    print(f"\nDiferença: R$ {diferenca:.2f} ({percentual:.1f}%)")
    print("Insight: Black Friday tem impacto significativo nos preços.")

def example_4_regional_comparison():
    """Exemplo 4: Comparação regional"""
    print("\n" + "="*70)
    print(" "*15 + "EXEMPLO 4: COMPARAÇÃO REGIONAL")
    print("="*70 + "\n")
    
    predictor = PricePredictor()
    
    # Pedido base
    pedido_base = {
        'customer_id': 'C00200',
        'order_date': datetime.now().strftime('%Y-%m-%d'),
        'categoria': 'Eletrônicos',
        'idade_cliente': 35,
        'segmento_cliente': 'Ouro',
    }
    
    # Testar em diferentes regiões
    regioes = [
        ('SP', 'Sudeste'),
        ('RS', 'Sul'),
        ('BA', 'Nordeste'),
        ('DF', 'Centro-Oeste'),
        ('AM', 'Norte'),
    ]
    
    print("📋 Cenário: Mesmo pedido em diferentes estados\n")
    
    resultados = []
    for estado, regiao in regioes:
        pedido = pedido_base.copy()
        pedido['estado'] = estado
        pedido['regiao'] = regiao
        
        preco = predictor.predict(pedido)
        resultados.append((estado, regiao, preco))
        print(f"{estado} ({regiao:15s}): R$ {preco:8.2f}")
    
    # Ordenar por preço
    resultados.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n🏆 Maior preço: {resultados[0][0]} - R$ {resultados[0][2]:.2f}")
    print(f"🔻 Menor preço: {resultados[-1][0]} - R$ {resultados[-1][2]:.2f}")
    print(f"Diferença: R$ {resultados[0][2] - resultados[-1][2]:.2f}")
    print("\nInsight: Localização tem impacto relevante no valor do pedido.")

def example_5_category_comparison():
    """Exemplo 5: Comparação entre categorias"""
    print("\n" + "="*70)
    print(" "*15 + "EXEMPLO 5: COMPARAÇÃO DE CATEGORIAS")
    print("="*70 + "\n")
    
    predictor = PricePredictor()
    
    # Pedido base
    pedido_base = {
        'customer_id': 'C00300',
        'order_date': datetime.now().strftime('%Y-%m-%d'),
        'estado': 'SP',
        'regiao': 'Sudeste',
        'idade_cliente': 35,
        'segmento_cliente': 'Ouro',
    }
    
    # Categorias para testar
    categorias = [
        'Eletrônicos',
        'Casa e Jardim',
        'Esportes',
        'Roupas',
        'Beleza',
        'Brinquedos',
        'Alimentos',
        'Livros',
    ]
    
    print("📋 Cenário: Mesmo cliente em diferentes categorias\n")
    
    resultados = []
    for categoria in categorias:
        pedido = pedido_base.copy()
        pedido['categoria'] = categoria
        
        preco = predictor.predict(pedido)
        resultados.append((categoria, preco))
        print(f"{categoria:20s}: R$ {preco:8.2f}")
    
    # Ordenar por preço
    resultados.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 Categoria mais cara: {resultados[0][0]} - R$ {resultados[0][1]:.2f}")
    print(f"Categoria mais barata: {resultados[-1][0]} - R$ {resultados[-1][1]:.2f}")
    print(f"Diferença: R$ {resultados[0][1] - resultados[-1][1]:.2f}")
    print("\nInsight: Categoria é o fator mais impactante no preço.")

def example_6_batch_pricing():
    """Exemplo 6: Precificação em lote para estratégia de marketing"""
    print("\n" + "="*70)
    print(" "*15 + "EXEMPLO 6: ANÁLISE DE LOTE")
    print("="*70 + "\n")
    
    predictor = PricePredictor()
    
    # Criar vários pedidos simulando uma campanha
    pedidos = []
    for i in range(100):
        pedido = {
            'customer_id': f'CAMP_{i:03d}',
            'order_date': datetime.now().strftime('%Y-%m-%d'),
            'categoria': 'Eletrônicos',
            'estado': 'SP',
            'regiao': 'Sudeste',
            'idade_cliente': 25 + (i % 40),  # Idades variadas
            'segmento_cliente': ['Bronze', 'Prata', 'Ouro', 'Platina'][i % 4],
        }
        pedidos.append(pedido)
    
    df_pedidos = pd.DataFrame(pedidos)
    precos = predictor.predict(df_pedidos)
    
    df_pedidos['preco_previsto'] = precos
    
    print("📋 Cenário: Campanha de marketing para 100 clientes\n")
    
    # Análise por segmento
    analise_segmento = df_pedidos.groupby('segmento_cliente')['preco_previsto'].agg([
        ('Quantidade', 'count'),
        ('Média', 'mean'),
        ('Mínimo', 'min'),
        ('Máximo', 'max'),
        ('Total', 'sum')
    ]).round(2)
    
    print("📊 Análise por Segmento:")
    print(analise_segmento)
    
    print(f"\n💰 Receita Prevista Total: R$ {df_pedidos['preco_previsto'].sum():,.2f}")
    print(f"📈 Ticket Médio: R$ {df_pedidos['preco_previsto'].mean():.2f}")
    
    print("\n💡 Insight: Use essas previsões para:")
    print("   - Definir orçamento de marketing")
    print("   - Segmentar ofertas")
    print("   - Prever receita da campanha")

def main():
    """Executa todos os exemplos"""
    print("\n" + "="*70)
    print(" "*10 + "EXEMPLOS DE USO - PREDIÇÃO DE PREÇO")
    print("="*70)
    
    try:
        example_1_new_customer()
        example_2_premium_customer()
        example_3_seasonal_comparison()
        example_4_regional_comparison()
        example_5_category_comparison()
        example_6_batch_pricing()
        
        print("\n" + "="*70)
        print(" "*20 + "EXEMPLOS CONCLUÍDOS!")
        print("="*70)
        print("\n[OK] Todos os exemplos foram executados com sucesso!")
        print("Use esses padrões para implementar suas próprias análises.")
        
    except FileNotFoundError:
        print("\n[ERRO] Modelo não encontrado!")
        print("Execute primeiro: python src/train.py")
    except Exception as e:
        print(f"\n[ERRO] {e}")

if __name__ == "__main__":
    main()
