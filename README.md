
# 🛒 Predicción de Demanda en Retail

En este proyecto desarrollaré un **pipeline de Machine Learning** para entrenar y evaluar modelos de predicción de demanda en un escenario de retail.  

La base del proyecto es el reto [Rohlik Sales Forecasting Challenge](https://www.kaggle.com/competitions/rohlik-sales-forecasting-challenge-v2/overview), cuyo objetivo consiste en **predecir las ventas diarias de productos seleccionados para un horizonte de 14 días**, utilizando datos históricos de ventas, precios y calendario.

El proyecto sigue la metodología [**CRISP-DM**](https://www.ibm.com/docs/es/spss-modeler/saas?topic=dm-crisp-help-overview), abordando cada etapa:  
1. **Definición del problema** y justificación de impacto en el negocio.  
2. **Análisis exploratorio y preprocesamiento** de datos históricos.  
3. **Modelado y evaluación** con métricas adecuadas (WMAE como principal).  
4. **Implementación de un pipeline reproducible** con orquestación y registro de modelos.  
5. **Documentación y comunicación de resultados**, incluyendo análisis de errores y explicabilidad.  

La finalidad no es únicamente construir un modelo predictivo, sino también mostrar un flujo completo de *machine learning en producción local* usando **Prefect**, **MLflow** y **Evidently** como herramientas clave.


# 📌 Definición del Problema y Objetivos

## Contexto
Rohlik es una empresa del sector *e-grocery* (supermercado en línea) que administra múltiples almacenes para surtir pedidos diarios. En este tipo de negocio, el reto de **predecir la demanda futura de productos** es crucial para:
- Optimizar inventarios.  
- Reducir desperdicios por caducidad.  
- Mejorar el nivel de servicio (evitar faltantes de stock).  

## Problema a resolver
Actualmente, la planificación de inventarios se apoya en métodos heurísticos y reglas de negocio (por ejemplo, promedios históricos). Estos enfoques no capturan la complejidad de los patrones reales:
- Estacionalidad (alta demanda en fines de semana o feriados).  
- Promociones y descuentos.  
- Diferencias de comportamiento entre categorías de productos.  
- Variabilidad entre almacenes.  

El problema central es entonces:  
**¿Cómo construir un modelo de machine learning que pueda predecir de manera precisa las ventas diarias por SKU y almacén, minimizando errores en productos clave?**

## Relevancia e impacto
- **Compras**: prever cantidades exactas y negociar con proveedores.  
- **Logística**: planear rutas de entrega y personal de surtido.  
- **Dirección comercial**: evaluar impacto de promociones y ajustar estrategia.  

El impacto potencial es una **reducción de costos operativos** y una **mejor experiencia del cliente final**.

## Objetivos del proyecto
1. **Generar un modelo predictivo** capaz de estimar ventas futuras por SKU y almacén.  
2. **Evaluar el modelo** con la métrica oficial del reto (Weighted Mean Absolute Error, WMAE) y con al menos una métrica complementaria (MAE/RMSE).  
3. **Diseñar un pipeline reproducible** que abarque desde la ingesta de datos hasta el registro de modelos entrenados.  
4. **Explicar los resultados** con análisis de importancia de variables y análisis de errores, para ofrecer transparencia a los stakeholders.  

## Acciones sin ML (baseline de negocio)
Antes de recurrir a machine learning, se podrían aplicar:
- Promedios móviles.  
- Valor de ventas del día anterior como predicción (modelo ingenuo).  
- Ajustes manuales de inventario en fechas especiales (ej. Navidad).  

Estos métodos servirán como *baseline* y permitirán demostrar la mejora que ofrece el modelo de ML.
