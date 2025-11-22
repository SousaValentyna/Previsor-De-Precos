"""
Interface Gráfica com Gradio para Previsão de Preço de Pedidos

Este aplicativo fornece uma interface web interativa para fazer predições
de preço médio de pedidos usando o modelo LightGBM treinado.
"""

import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from predict import PricePredictor
import config

# Carregar o modelo uma vez no início
print("Carregando modelo...")
try:
    predictor = PricePredictor()
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    print("Execute 'python src/train.py' para treinar o modelo primeiro.")
    sys.exit(1)


def predict_price(customer_id, order_date, categoria, estado, regiao, 
                  idade_cliente, segmento_cliente):
    """
    Faz a predição de preço baseado nos inputs do usuário
    """
    try:
        # Criar dicionário com os dados
        order_data = {
            'customer_id': customer_id,
            'order_date': pd.to_datetime(order_date),
            'categoria': categoria,
            'estado': estado,
            'regiao': regiao,
            'idade_cliente': int(idade_cliente),
            'segmento_cliente': segmento_cliente
        }
        
        # Fazer predição
        predicted_price = predictor.predict(order_data)
        
        # Formatar resultado
        result = f"""
### Previsão de Preço
        
**Preço Previsto: R$ {predicted_price:,.2f}**

---

#### Detalhes do Pedido:
- **Cliente:** {customer_id}
- **Data:** {order_date}
- **Categoria:** {categoria}
- **Estado:** {estado}
- **Região:** {regiao}
- **Idade do Cliente:** {idade_cliente} anos
- **Segmento:** {segmento_cliente}

---

#### Informações Adicionais:
- O modelo considera mais de 45 features para fazer a predição
- Inclui histórico de compras, sazonalidade, localização e categoria
- Acurácia do modelo: R² = 0.9978, RMSE = R$ 15.40
"""
        
        return result
        
    except Exception as e:
        return f"Erro ao fazer predição: {str(e)}"


def predict_batch_from_file(file):
    """
    Faz predições em lote a partir de arquivo CSV
    """
    try:
        # Ler arquivo CSV
        df = pd.read_csv(file.name)
        
        # Validar colunas necessárias
        required_cols = ['customer_id', 'order_date', 'categoria', 'estado', 
                        'regiao', 'idade_cliente', 'segmento_cliente']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return f"Erro: Colunas faltando no CSV: {', '.join(missing_cols)}", None
        
        # Fazer predições
        results = predictor.predict_batch(df)
        
        # Salvar resultados
        output_path = 'predictions_gradio.csv'
        results.to_csv(output_path, index=False)
        
        # Criar resumo
        summary = f"""
### Predições em Lote Concluídas

**Total de pedidos processados:** {len(results)}

**Estatísticas das Predições:**
- Preço Médio: R$ {results['predicted_price'].mean():,.2f}
- Preço Mínimo: R$ {results['predicted_price'].min():,.2f}
- Preço Máximo: R$ {results['predicted_price'].max():,.2f}
- Desvio Padrão: R$ {results['predicted_price'].std():,.2f}

**Arquivo salvo:** {output_path}

#### Primeiras 5 predições:
"""
        
        # Adicionar primeiras predições ao resumo
        for idx, row in results.head().iterrows():
            summary += f"\n- Cliente {row['customer_id']}: R$ {row['predicted_price']:.2f}"
        
        return summary, output_path
        
    except Exception as e:
        return f"Erro ao processar arquivo: {str(e)}", None


def get_example_data():
    """Retorna dados de exemplo para teste"""
    return [
        "C00001",
        "2024-11-22",
        "Eletrônicos",
        "SP",
        "Sudeste",
        35,
        "Ouro"
    ]


# Criar interface com abas
with gr.Blocks(title="Previsão de Preço de Pedidos") as demo:
    
    gr.Markdown(
        """
        # Previsão de Preço Médio de Pedidos
        
        Sistema de Machine Learning para prever o preço médio de pedidos usando LightGBM.
        Desenvolvido com mais de 50 features incluindo histórico de compras, sazonalidade e localização.
        """
    )
    
    with gr.Tabs():
        # Aba 1: Predição Individual
        with gr.TabItem("Predição Individual"):
            gr.Markdown("### Preencha os dados do pedido para obter a predição de preço")
            
            with gr.Row():
                with gr.Column():
                    customer_id = gr.Textbox(
                        label="ID do Cliente",
                        placeholder="Ex: C00001",
                        value="C00001"
                    )
                    
                    order_date = gr.Textbox(
                        label="Data do Pedido",
                        placeholder="YYYY-MM-DD",
                        value=datetime.now().strftime("%Y-%m-%d")
                    )
                    
                    categoria = gr.Dropdown(
                        choices=list(config.CATEGORIES.keys()),
                        label="Categoria do Produto",
                        value="Eletrônicos"
                    )
                    
                    estado = gr.Dropdown(
                        choices=list(config.ESTADOS.keys()),
                        label="Estado",
                        value="SP"
                    )
                
                with gr.Column():
                    regiao = gr.Dropdown(
                        choices=["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"],
                        label="Região",
                        value="Sudeste"
                    )
                    
                    idade_cliente = gr.Slider(
                        minimum=18,
                        maximum=80,
                        value=35,
                        step=1,
                        label="Idade do Cliente"
                    )
                    
                    segmento_cliente = gr.Dropdown(
                        choices=["Bronze", "Prata", "Ouro", "Platina"],
                        label="Segmento do Cliente",
                        value="Ouro"
                    )
            
            with gr.Row():
                predict_btn = gr.Button("Prever Preço", variant="primary", size="lg")
                clear_btn = gr.ClearButton(
                    components=[customer_id, order_date, categoria, estado, 
                               regiao, idade_cliente, segmento_cliente]
                )
                example_btn = gr.Button("Carregar Exemplo")
            
            output_single = gr.Markdown(label="Resultado")
            
            # Ações dos botões
            predict_btn.click(
                fn=predict_price,
                inputs=[customer_id, order_date, categoria, estado, regiao, 
                       idade_cliente, segmento_cliente],
                outputs=output_single
            )
            
            example_btn.click(
                fn=lambda: get_example_data(),
                inputs=None,
                outputs=[customer_id, order_date, categoria, estado, regiao, 
                        idade_cliente, segmento_cliente]
            )
        
        # Aba 2: Predição em Lote
        with gr.TabItem("Predição em Lote"):
            gr.Markdown(
                """
                ### Upload de arquivo CSV para predições em massa
                
                **Formato do arquivo CSV:**
                - Colunas obrigatórias: `customer_id`, `order_date`, `categoria`, `estado`, 
                  `regiao`, `idade_cliente`, `segmento_cliente`
                - Data no formato: YYYY-MM-DD
                - Sem cabeçalho com caracteres especiais
                """
            )
            
            with gr.Row():
                file_input = gr.File(
                    label="Upload do arquivo CSV",
                    file_types=[".csv"]
                )
            
            batch_predict_btn = gr.Button("Processar Lote", variant="primary", size="lg")
            
            with gr.Row():
                output_batch = gr.Markdown(label="Resultado")
                download_file = gr.File(label="Download dos Resultados")
            
            batch_predict_btn.click(
                fn=predict_batch_from_file,
                inputs=file_input,
                outputs=[output_batch, download_file]
            )
            
            # Botão para baixar CSV de exemplo
            gr.Markdown("### Baixar CSV de Exemplo")
            
            def create_example_csv():
                """Cria um CSV de exemplo"""
                example_data = {
                    'customer_id': ['C00001', 'C00002', 'C00003'],
                    'order_date': ['2024-11-22', '2024-11-22', '2024-11-22'],
                    'categoria': ['Eletrônicos', 'Roupas', 'Livros'],
                    'estado': ['SP', 'RJ', 'RS'],
                    'regiao': ['Sudeste', 'Sudeste', 'Sul'],
                    'idade_cliente': [35, 28, 42],
                    'segmento_cliente': ['Ouro', 'Prata', 'Bronze']
                }
                df = pd.DataFrame(example_data)
                example_path = 'example_batch.csv'
                df.to_csv(example_path, index=False)
                return example_path
            
            example_csv_btn = gr.Button("Gerar CSV de Exemplo")
            example_csv_output = gr.File(label="Arquivo de Exemplo")
            
            example_csv_btn.click(
                fn=create_example_csv,
                inputs=None,
                outputs=example_csv_output
            )
        
        # Aba 3: Informações do Modelo
        with gr.TabItem("Sobre o Modelo"):
            gr.Markdown(
                f"""
                ## Informações do Modelo
                
                ### Tecnologia
                - **Algoritmo:** LightGBM (Gradient Boosting)
                - **Tipo:** Regressão
                - **Features:** 45+ variáveis preditoras
                
                ### Features Utilizadas
                
                #### 1. Históricas (RFM)
                - Número de pedidos anteriores
                - Valor médio de pedidos passados
                - Dias desde último pedido
                - Tendência de gastos
                - Estatísticas (máx, mín, desvio)
                
                #### 2. Temporais
                - Dia da semana
                - Mês e trimestre
                - Sazonalidade (Black Friday, Natal, etc.)
                - Features cíclicas (sin/cos)
                - Início/meio/fim de mês
                
                #### 3. Geográficas
                - Estado
                - Região
                - Preço médio por localização
                - Densidade populacional
                
                #### 4. Categoria
                - Tipo de produto
                - Preço médio da categoria
                - Popularidade da categoria
                - Desvio padrão de preços
                
                #### 5. Cliente
                - Idade
                - Segmento (Bronze, Prata, Ouro, Platina)
                - Interações entre variáveis
                
                ### Métricas de Performance
                - **R² Score:** 0.9978 (99.78% de variância explicada)
                - **RMSE:** R$ 15.40
                - **MAE:** R$ 4.91
                - **MAPE:** 3.75%
                
                ### Validação
                - **Método:** Validação cruzada 5-fold
                - **Dataset:** 10.000 pedidos
                - **Período:** 2022-2024
                
                ### Categorias Disponíveis
                {chr(10).join([f"- {cat}: R$ {info['base_price']:.2f} (base)" for cat, info in config.CATEGORIES.items()])}
                
                ### Estados Cobertos
                {chr(10).join([f"- {estado} ({info['regiao']})" for estado, info in config.ESTADOS.items()])}
                
                ### Como Usar
                
                1. **Predição Individual:** Preencha os campos e clique em "Prever Preço"
                2. **Predição em Lote:** Faça upload de um arquivo CSV com múltiplos pedidos
                3. **CSV de Exemplo:** Baixe um arquivo de exemplo para ver o formato correto
                
                ### Observações
                - Para clientes novos, o modelo usa valores padrão para features históricas
                - A predição leva em conta sazonalidade e eventos especiais
                - Preços variam significativamente por categoria e localização
                
                ---
                
                **Desenvolvido com:** Python, LightGBM, Pandas, Scikit-learn, Gradio
                """
            )
    
    gr.Markdown(
        """
        ---
        
        ### Como executar localmente
        
        ```bash
        # Instalar dependências
        pip install -r requirements.txt
        
        # Treinar modelo (se necessário)
        python src/train.py
        
        # Iniciar interface
        python app.py
        ```
        
        **GitHub:** [github.com/seu-usuario/preco-pedido-ml](https://github.com)
        """
    )


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*15 + "INICIANDO INTERFACE GRADIO")
    print("="*70)
    print("\nAcesse a interface no navegador através da URL que será exibida abaixo.")
    print("Pressione Ctrl+C para encerrar o servidor.\n")
    
    # Iniciar interface
    demo.launch(
        share=False,  # Mude para True se quiser compartilhar publicamente
        server_name="127.0.0.1",
        show_error=True,
        inbrowser=True  # Abre automaticamente no navegador
    )
