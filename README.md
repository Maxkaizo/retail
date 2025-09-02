
# 🛒 Predicción de Demanda en Retail

En este proyecto desarrollaré un **pipeline de Machine Learning** para entrenar y evaluar modelos de predicción de demanda en un escenario de retail.  

La base del proyecto es el reto [Rohlik Sales Forecasting Challenge](https://www.kaggle.com/competitions/rohlik-sales-forecasting-challenge-v2/overview), cuyo objetivo consiste en **predecir las ventas diarias de productos seleccionados para un horizonte de 14 días**, utilizando datos históricos de ventas, precios y calendario.

El proyecto sigue la metodología [**CRISP-DM**](https://www.ibm.com/docs/es/spss-modeler/saas?topic=dm-crisp-help-overview), abordando cada etapa:  
1. **Definición del problema** y justificación de impacto en el negocio.  
2. **Análisis exploratorio y preprocesamiento** de datos históricos.  
3. **Modelado y evaluación** con métricas adecuadas (WMAE como principal).  
4. **Implementación de un pipeline reproducible** con orquestación y registro de modelos.  
5. **Documentación y comunicación de resultados**, incluyendo análisis de errores y explicabilidad.  

La finalidad no es únicamente construir un modelo predictivo, sino también mostrar un flujo completo de *machine learning en producción local* usando herramientas de código abierto, con buenas prácticas que permitan escalar de manera estable a entornos productivos y **mostrar cómo esta práctica puede convertirse en una herramienta que genere valor tangible para el negocio**.  

# 📌 Definición del Problema y Objetivos

## Contexto
Rohlik es una empresa del sector *e-grocery* (supermercado en línea) que administra múltiples almacenes para surtir pedidos diarios.  
En este tipo de negocio, el reto de **predecir la demanda futura de productos** es crucial para:
- Optimizar inventarios.  
- Reducir desperdicios por caducidad.  
- Mejorar el nivel de servicio (evitar faltantes de stock).  

## Problema a resolver
Actualmente, la planificación de inventarios se apoya en métodos heurísticos y reglas de negocio (por ejemplo, promedios históricos).  
Estos enfoques no capturan la complejidad de los patrones reales:
- Estacionalidad (alta demanda en fines de semana o feriados).  
- Promociones y descuentos.  
- Diferencias de comportamiento entre categorías de productos.  
- Variabilidad entre almacenes.  

El problema central es entonces:  
**¿Cómo construir un modelo de machine learning que pueda predecir de manera precisa las ventas diarias por SKU y almacén, minimizando errores en productos clave?**


## Objetivos del proyecto
De manera específica, se busca:  

1. **Mejorar la planificación de inventarios** mediante predicciones de ventas más precisas, lo que reduce faltantes y desperdicio de producto.  
2. **Optimizar la operación logística y de compras**, al proveer estimaciones confiables que faciliten la toma de decisiones en abastecimiento y distribución.  
3. **Evaluar el impacto del uso de machine learning frente a enfoques tradicionales**, demostrando con métricas claras la mejora alcanzada en la predicción de la demanda.  
4. **Implementar un pipeline reproducible** que sirva como punto de partida para incorporar prácticas de ML en el negocio, permitiendo escalar y mejorar con el tiempo.  
5. **Facilitar la interpretación de resultados** a través de análisis de errores y explicabilidad, para que los hallazgos puedan integrarse en procesos de negocio y no queden como una “caja negra”.  


## Relevancia e impacto
- **Compras**: prever cantidades exactas y negociar con proveedores.  
- **Logística**: planear rutas de entrega y personal de surtido.  
- **Dirección comercial**: evaluar impacto de promociones y ajustar estrategia.  

El impacto potencial es una **reducción de costos operativos** y una **mejor experiencia del cliente final**.

Citando la descripción de la competencia:
"Por qué es importante?
Las predicciones precisas de ventas son cruciales para los procesos de planeación, la cadena de suministro, la logística de entregas y la gestión de inventarios. Al optimizar los pronósticos, podemos minimizar el desperdicio y hacer más eficientes las operaciones, logrando que nuestros servicios de comercio electrónico de comestibles sean más sostenibles y eficaces."

## Acciones sin ML (baseline de negocio)

Antes de implementar un modelo de machine learning, existen diversas acciones que la empresa podría realizar para mejorar la planeación de demanda:

- **Métodos estadísticos simples**: aplicar promedios móviles o suavización exponencial para capturar tendencias básicas.  
- **Referencias a días similares previos**: usar como predicción los valores de ventas del **día anterior**, la **semana anterior** o incluso el **mes anterior** (modelo ingenuo).  
- **Reglas de negocio**: ajustar inventarios en fechas especiales (ej. Navidad, vacaciones escolares, eventos locales).  
- **Análisis de históricos**: revisar manualmente patrones de estacionalidad y detectar productos con mayor volatilidad o estacionalidad marcada.  
- **Segmentación de productos**: identificar categorías críticas (ej. lácteos, panadería, bebidas) y darles un tratamiento especial en las órdenes de compra.  
- **Buffers de seguridad**: establecer inventario de seguridad para productos de alta rotación o alta variabilidad en la demanda.  
- **Experiencia del personal**: muchas empresas recurren a la sensibilidad y conocimiento acumulado de compradores y planificadores de inventario; aunque no es un enfoque sistemático, puede complementar los métodos anteriores.  

Estas acciones son relevantes porque forman un **baseline operativo**, contra el cual se podrá comparar el valor añadido de un modelo de machine learning más avanzado. 

## 📊 Descripción del dataset

El dataset contiene información histórica de ventas de Rohlik para distintos productos e inventarios, junto con datos de calendario y contexto operacional.  
Las columnas principales son:  

### 📂 Archivos

- **sales_train.csv** – 
- **sales_test.csv** – conjunto de prueba completo.  
- **inventory.csv** –  
- **calendar.csv** – 


### **sales_train.csv / sales_test.csv**
Conjuntos de entrenamiento y pruebas que contiene los datos históricos de ventas para cada fecha e inventario, con las características descritas abajo.  
- **unique_id**: identificador único del inventario (producto en un almacén específico).  
- **date**: fecha del registro.  
- **warehouse**: nombre del almacén.  
- **total_orders**: número total de pedidos procesados por el almacén en esa fecha (variable de contexto compartida por todos los SKUs del almacén).  
- **sales**: volumen de ventas (en piezas o kg), ajustado por disponibilidad.  
- **sell_price_main**: precio de venta.  
- **availability**: proporción del día en que el inventario estuvo disponible (1 = todo el día).  
- **type_0_discount, type_1_discount, …**: descuentos aplicados, expresados como porcentaje sobre el precio original.  

### **inventory.csv**
Información adicional sobre el inventario, como el producto (los mismos productos en diferentes almacenes comparten el mismo **product_unique_id** y nombre, pero tienen distinto **unique_id**). 
- **unique_id**: identificador único del inventario.  
- **product_unique_id**: identificador del producto (compartido entre almacenes).  
- **name**: nombre del producto.  
- **L1_category_name, L2_category_name, …**: categorías internas (más granularidad a mayor número).  
- **warehouse**: nombre del almacén.  

### **calendar.csv**
Calendario con datos sobre feriados o eventos específicos de los almacenes. Algunas columnas ya están incluidas en el conjunto de entrenamiento, pero este archivo incluye filas adicionales para fechas donde ciertos almacenes pudieron haber estado cerrados por feriado o domingo (y por tanto no aparecen en el train set).  
- **warehouse**: nombre del almacén.  
- **date**: fecha.  
- **holiday_name**: nombre del feriado (si aplica).  
- **holiday**: indicador binario de feriado.  
- **shops_closed**: indicador de cierre de tiendas.  
- **winter_school_holidays**: vacaciones de invierno escolares.  
- **school_holidays**: vacaciones escolares.  

### **test_weights.csv**
- **unique_id**: identificador único del inventario.  
- **weight**: peso utilizado para el cálculo de la métrica oficial (WMAE).  

---

## 🚀 Estrategia de despliegue propuesta

El despliegue se realizará de manera **local, reproducible y en batch**, lo que significa que:  

- **Entrenamiento**: se ejecutará bajo demanda utilizando datos históricos, generando y registrando un modelo con sus métricas.  
- **Inferencia**: el modelo entrenado producirá predicciones para el siguiente día, simulando la llegada de datos nuevos.  
- **Evaluación**: conforme se dispongan los valores reales, se compararán contra las predicciones para calcular métricas de desempeño.  

Este enfoque asegura un pipeline **ejecutable de principio a fin** y fácilmente repetible, cumpliendo los requisitos de la rúbrica sin necesidad de desplegar un servicio de inferencia en línea.

## ⚙️ Arquitectura de la Solución

El proyecto se soporta en una **arquitectura local con contenedores**, que permite reproducir experimentos de manera confiable y almacenar resultados en la nube:

![alt text](image.png)

- **MLflow**: servidor de tracking y registry, configurado para guardar artefactos y modelos en un bucket de **Amazon S3**.  
- **Prefect 2.0**: servidor de orquestación para gestionar la ejecución de flujos de entrenamiento y validación.  
- **Postgres**: base de datos para almacenar metadatos de MLflow y Prefect.  
- **Docker Compose**: define los servicios y redes internas, asegurando que todos los componentes se levanten de forma consistente.  
- **S3 bucket (`ml-codigo-facilito-2025`)**: almacena de manera centralizada los modelos y artefactos generados en cada run.  

Con esta configuración:  
- Los experimentos se lanzan desde el entorno local (conda env `forecasting`).  
- MLflow recibe métricas, parámetros y artefactos.  
- Prefect registra y monitorea la ejecución de los flujos.  
- Los modelos quedan versionados y disponibles para ser promovidos en el registry.  

La infraestructura ya está **funcional y accesible vía UI** tanto para MLflow (`http://localhost:5000`) como para Prefect (`http://localhost:4200`).  

---

## 🧪 Estrategia de Entrenamiento

Se desarrolló una **libreta de entrenamiento inicial**, conectada al pipeline de tracking, que implementa la siguiente estrategia:

1. **Carga y filtrado de datos**  
   - Se utiliza el archivo enriquecido `sales_train_enriched.csv`.  
   - Para acelerar el desarrollo, se filtran únicamente registros del año **2023**.  
   - Se respeta la naturaleza temporal de los datos: los splits se hacen **cronológicamente** (80% entrenamiento, 20% validación), evitando fuga de información del futuro.  

2. **Selección de features y target**  
   - Target inicial: `sales_log`.  
   - Features seleccionadas: transformaciones logarítmicas (`orders_log`, `price_log`) y variables derivadas como `max_discount`.  
   - Esto asegura estabilidad en la distribución y reduce el sesgo en las métricas.  

3. **Modelos considerados**  
   - **Baseline** con RandomForest, usando hiperparámetros básicos.  
   - **LightGBM, XGBoost y CatBoost**, optimizados con **Hyperopt** sobre un espacio de hiperparámetros reducido (10–20 evaluaciones por modelo).  

4. **Registro en MLflow**  
   - Cada run se registra con parámetros, métricas (RMSE) y artefactos (modelo serializado).  
   - Los modelos quedan almacenados en S3 y versionados en el registry de MLflow, listos para ser comparados o promovidos.  

---

## ✅ Estado Actual

- La infraestructura (MLflow + Prefect + Postgres + S3) está **levantada y validada**.  
- Se han corrido los primeros experimentos de prueba exitosamente, confirmando el flujo de tracking y almacenamiento en la nube.  
- La libreta de entrenamiento ya permite entrenar modelos base y optimizados, dejando listos los componentes para la siguiente etapa:  
  - Definir la lista final de features relevantes.  
  - Completar la comparativa de modelos.  
  - Documentar los resultados en el README y realizar un primer *submit* del proyecto.  
