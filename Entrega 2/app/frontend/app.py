import gradio as gr
from datetime import date
import json
from client import APIClient

# configuracion de api backend
client = APIClient()

def predict_single(
    # customer inputs
    customer_id, product_id, fecha_prediccion,
    customer_type, coord_x, coord_y, num_deliveries,
    # product inputs
    brand, category, sub_category, segment, package, size,
    # historical inputs (opcionales)
    use_historical,
    total_ordenes, productos_unicos, veces_comprado, dias_ultima_compra
):
    """
    funcion principal de prediccion para interfaz simple
    """

    # construir request
    request_data = {
        "customer_id": customer_id,
        "product_id": product_id,
        "fecha_prediccion": fecha_prediccion,
        "customer": {
            "customer_type": customer_type,
            "X": coord_x,
            "Y": coord_y,
            "num_deliver_per_week": num_deliveries
        },
        "product": {
            "brand": brand,
            "category": category,
            "sub_category": sub_category,
            "segment": segment,
            "package": package,
            "size": size
        }
    }

    # agregar datos historicos si usuario los habilito
    if use_historical:
        request_data["historical"] = {
            "total_ordenes_global": total_ordenes if total_ordenes > 0 else None,
            "productos_unicos_global": productos_unicos if productos_unicos > 0 else None,
            "veces_comprado_global": veces_comprado if veces_comprado > 0 else None,
            "dias_desde_ultima_compra": dias_ultima_compra if dias_ultima_compra > 0 else None
        }

    # llamar api
    try:
        response = client.predict(request_data)

        # formatear resultado
        prediction_text = "COMPRARA" if response['prediction'] == 1 else "NO COMPRARA"
        probability_text = f"{response['probability']:.1%}"

        # crear visualizacion de features importantes
        features_md = "### features mas influyentes\n\n"
        for feat in response['top_features']:
            features_md += f"- **{feat['feature']}**: {feat['importance']:.3f}\n"

        # metadata del modelo
        metadata_md = f"""
### informacion del modelo

- version: {response['model_metadata']['model_version']}
- f1-score: {response['model_metadata']['f1_score']:.3f}
- roc-auc: {response['model_metadata']['roc_auc']:.3f}
"""

        return (
            prediction_text,
            probability_text,
            response['interpretation'],
            response['confidence_level'],
            response['recommendation'],
            features_md,
            metadata_md
        )

    except Exception as e:
        error_msg = f"error: {str(e)}"
        return error_msg, "", "", "", "", "", ""

def check_health():
    """verifica estado del backend"""
    try:
        health = client.health_check()
        if health['status'] == 'healthy':
            return f"✅ backend operativo\nmodelo cargado: {health['model_loaded']}"
        else:
            return f"❌ backend inoperativo"
    except Exception as e:
        return f"❌ error al conectar con backend: {str(e)}"

# crear interfaz principal
with gr.Blocks(title="sodai drinks - prediccion de compras", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🥤 prediccion de compras - sodai drinks

    sistema de prediccion de probabilidad de compra de productos basado en machine learning.

    ## como usar

    1. ingresa los datos del cliente y producto
    2. opcionalmente, activa y completa el historial de compras
    3. presiona "predecir" para obtener resultado

    el modelo predice si un cliente comprara un producto en la siguiente semana.
    """)

    # check de salud
    with gr.Row():
        health_button = gr.Button("🔍 verificar estado del sistema", size="sm")
        health_output = gr.Textbox(label="estado", interactive=False, lines=2)

    health_button.click(fn=check_health, outputs=health_output)

    gr.Markdown("---")

    # tabs para diferentes modos
    with gr.Tabs():

        # tab 1: prediccion simple
        with gr.Tab("📊 prediccion individual"):

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### datos de identificacion")
                    customer_id = gr.Textbox(label="id cliente", value="CUST_001")
                    product_id = gr.Textbox(label="id producto", value="PROD_001")
                    fecha_prediccion = gr.Textbox(
                        label="fecha de prediccion (YYYY-MM-DD)",
                        value=date.today().isoformat()
                    )

                    gr.Markdown("### datos del cliente")
                    customer_type = gr.Dropdown(
                        label="tipo de cliente",
                        choices=["REGULAR", "VIP", "NEW"],
                        value="REGULAR"
                    )
                    coord_x = gr.Number(label="coordenada x", value=0.0)
                    coord_y = gr.Number(label="coordenada y", value=0.0)
                    num_deliveries = gr.Slider(
                        label="entregas por semana",
                        minimum=0,
                        maximum=7,
                        step=1,
                        value=2
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### datos del producto")
                    brand = gr.Textbox(label="marca", value="coca_cola")
                    category = gr.Dropdown(
                        label="categoria",
                        choices=["refrescos", "agua", "jugos", "energeticas"],
                        value="refrescos"
                    )
                    sub_category = gr.Dropdown(
                        label="subcategoria",
                        choices=["gaseosas", "sin_gas", "naturales", "deportivas"],
                        value="gaseosas"
                    )
                    segment = gr.Dropdown(
                        label="segmento",
                        choices=["LOW", "MEDIUM", "HIGH", "PREMIUM"],
                        value="MEDIUM"
                    )
                    package = gr.Dropdown(
                        label="empaque",
                        choices=["botella", "lata", "tetra_pak", "garrafa"],
                        value="botella"
                    )
                    size = gr.Number(label="tamaño (litros)", value=1.5, minimum=0.1)

                with gr.Column(scale=1):
                    gr.Markdown("### historial de compras (opcional)")
                    use_historical = gr.Checkbox(
                        label="usar datos historicos",
                        value=False,
                        info="activa para clientes existentes con historial"
                    )

                    total_ordenes = gr.Number(
                        label="total de ordenes del cliente",
                        value=10,
                        minimum=0,
                        visible=False
                    )
                    productos_unicos = gr.Number(
                        label="productos unicos comprados",
                        value=5,
                        minimum=0,
                        visible=False
                    )
                    veces_comprado = gr.Number(
                        label="veces que compro este producto",
                        value=3,
                        minimum=0,
                        visible=False
                    )
                    dias_ultima_compra = gr.Number(
                        label="dias desde ultima compra",
                        value=7,
                        minimum=0,
                        visible=False
                    )

                    # toggle visibility de campos historicos
                    def toggle_historical(use_hist):
                        return [gr.update(visible=use_hist)] * 4

                    use_historical.change(
                        fn=toggle_historical,
                        inputs=[use_historical],
                        outputs=[total_ordenes, productos_unicos, veces_comprado, dias_ultima_compra]
                    )

            predict_button = gr.Button("🔮 predecir", variant="primary", size="lg")

            gr.Markdown("### resultado de la prediccion")

            with gr.Row():
                prediction_output = gr.Textbox(label="prediccion", scale=1)
                probability_output = gr.Textbox(label="probabilidad", scale=1)
                interpretation_output = gr.Textbox(label="interpretacion", scale=1)
                confidence_output = gr.Textbox(label="confianza", scale=1)

            recommendation_output = gr.Textbox(label="recomendacion", lines=2)

            with gr.Row():
                features_output = gr.Markdown(label="features importantes")
                metadata_output = gr.Markdown(label="metadata del modelo")

            # conectar boton con funcion
            predict_button.click(
                fn=predict_single,
                inputs=[
                    customer_id, product_id, fecha_prediccion,
                    customer_type, coord_x, coord_y, num_deliveries,
                    brand, category, sub_category, segment, package, size,
                    use_historical,
                    total_ordenes, productos_unicos, veces_comprado, dias_ultima_compra
                ],
                outputs=[
                    prediction_output,
                    probability_output,
                    interpretation_output,
                    confidence_output,
                    recommendation_output,
                    features_output,
                    metadata_output
                ]
            )

            # ejemplos precargados
            gr.Examples(
                examples=[
                    # cliente regular, coca cola 1.5L, sin historial
                    ["CUST_001", "PROD_001", "2024-01-15", "REGULAR", 0.0, 0.0, 2,
                     "coca_cola", "refrescos", "gaseosas", "MEDIUM", "botella", 1.5,
                     False, 0, 0, 0, 0],
                    # cliente vip, producto premium con historial
                    ["CUST_VIP_001", "PROD_PREM_001", "2024-12-20", "VIP", 1.5, 2.3, 5,
                     "perrier", "agua", "sin_gas", "PREMIUM", "botella", 0.75,
                     True, 50, 15, 10, 3],
                    # cliente nuevo, producto popular
                    ["CUST_NEW_001", "PROD_POP_001", "2024-06-15", "NEW", -0.5, 0.8, 1,
                     "sprite", "refrescos", "gaseosas", "MEDIUM", "lata", 0.355,
                     False, 0, 0, 0, 0],
                ],
                inputs=[
                    customer_id, product_id, fecha_prediccion,
                    customer_type, coord_x, coord_y, num_deliveries,
                    brand, category, sub_category, segment, package, size,
                    use_historical,
                    total_ordenes, productos_unicos, veces_comprado, dias_ultima_compra
                ],
                label="ejemplos"
            )

        # tab 2: informacion del modelo
        with gr.Tab("ℹ️ informacion del modelo"):
            gr.Markdown("""
            ## modelo de prediccion

            este sistema utiliza un modelo xgboost classifier entrenado con datos historicos
            de compras de sodai drinks.

            ### metricas del modelo

            - **accuracy**: 72%
            - **precision**: 62%
            - **recall**: 86%
            - **f1-score**: 72%
            - **roc-auc**: 84%

            ### features utilizadas (52 total)

            el modelo utiliza 52 features divididas en:
            - **27 numericas**: rfm del cliente, popularidad del producto, metricas temporales
            - **10 categoricas**: marca, categoria, segmento, tipo de cliente
            - **5 binarias**: flags de temporada, dia de semana, historial previo

            ### interpretacion de resultados

            - **probabilidad < 30%**: baja probabilidad de compra
            - **probabilidad 30-50%**: probabilidad media
            - **probabilidad 50-70%**: alta probabilidad
            - **probabilidad > 70%**: muy alta probabilidad

            ### como mejorar predicciones

            - proporcionar datos historicos reales del cliente
            - usar fechas recientes para predicciones
            - validar que los datos del producto sean correctos
            """)

            model_info_button = gr.Button("📄 cargar informacion detallada del modelo")
            model_info_output = gr.JSON(label="metricas y configuracion")

            def get_model_info():
                try:
                    return client.get_model_info()
                except Exception as e:
                    return {"error": str(e)}

            model_info_button.click(fn=get_model_info, outputs=model_info_output)

# lanzar aplicacion
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
