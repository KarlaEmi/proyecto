**PONTIFICIA UNIVERSIDAD CATÓLICA DEL PERÚ**  
**FACULTAD DE CIENCIAS E INGENIERÍA**  
**INTELIGENCIA ARTIFICIAL APLICADA (1INF62)**

**ENTREGABLE PARCIAL**   
**Tarea Académica (grupal)** 

**2026-2**

Deadline: domingo 11 de octubre 11:55 pm

**I. Integrantes que participaron en la elaboración del avance:**

|  | Nombre integrante | Descripción de su contribución |
| :---- | :---- | :---- |
| **1** | **Erich Fabian Leon Florez** | Revisión y análisis crítico de la literatura asignada (*paper* base). Definición operacional de la cohorte e identificación del *target* (complicaciones). Diseño y ejecución del Análisis Exploratorio de Datos (EDA) inicial. *Feature engineering* y preprocesamiento: tratamiento de valores nulos, registros anómalos, estrategia ante el desbalance de datos, etc. |
| **2** | **Karla Emily García Rojas** | Revisión y análisis crítico de la literatura asignada (*paper* base). Adquisición, consolidación y limpieza de las 5 tablas relacionales del *dataset* (SIS). *Feature engineering*: Extracción de variables geográficas (m s. n. m. / UBIGEO) y fisiológicas. Diseño, implementación y validación del *baseline.*  Revisión, refinamiento y validación final del EDA. |
| **3** | **Rasec Avyel Gonzales Leyva** | Subió su paper. |
| **4** | **Emilio** | Se retiró |

**II. Enlace al repositorio GitHub\*:**  
[https\://github.com/KarlaEmi/proyecto.git](https://github.com/KarlaEmi/proyecto.git) 

**III. Commit hash del último commit considerado para la entrega (OBLIGATORIO):**

**IV. Checklist de avance (✓/✗)**

| README completo | ✓ |
| :---- | :---- |
| Resumen de papers (mínimo 1 por integrante) | **✓** |
| Dataset | **✓** |
| Descripción del dataset (origen, características,...) | **✓** |
| Notebooks con EDA | **✓** |
| Propuesta de modelos | **✓** |
| Notebooks con baseline implementado | **✓** |
| Breve descripción de lo realizado (ENTREGABLE\_PARCIAL.md) | **✓** |


# Resumen del proyecto: predicción de complicaciones en pacientes hipertensos (SIS)

**Objetivo:** predecir si un paciente hipertenso asegurado en el SIS tendrá una **complicación grave en 2024**: ACV/AIT, infarto, insuficiencia cardíaca, enfermedad renal o fallecimiento. La predicción usa solo su información de 2022–2023.

**Notebooks:**

| Notebook | Etapa |
|---|---|
| `01_eda.ipynb` | Exploración de los datos crudos |
| `preprocesamiento.ipynb` | Construcción del dataset por paciente |
| `03_comparacion_modelos.ipynb` | Modelado y comparación de modelos |

---

## 1. Datos crudos

Son 5 archivos CSV del SIS (Plataforma Nacional de Datos Abiertos), con ≈170 millones de filas y atenciones de 2022 a 2024. Las tablas se unen con `ID_REGISTRO_REL`, el identificador de la atención.

| Archivo | Una fila es… | Filas |
|---|---|---|
| FUAS | una atención (paciente, fecha, edad, ubicación, presión arterial, hospitalización, destino) | 24.3 M |
| Diagnósticos | un diagnóstico CIE-10 de una atención | 38.1 M |
| Medicamentos | una entrega de medicamento | 39.4 M |
| Procedimientos | un procedimiento | 53.0 M |
| Insumos | una entrega de insumo | 15.8 M |

## 2. Hallazgos del EDA

| Hallazgo | Implicancia |
|---|---|
| 24.3 M atenciones de solo 1.03 M pacientes (mediana de 15 atenciones por paciente) | Hay que **resumir por paciente**. Si se modela por atención, el mismo paciente puede caer en train y en test |
| `ID_REGISTRO_REL` es único en FUAS y no hay IDs huérfanos | La unión de las 5 tablas a través de FUAS es segura |
| `DIAS_HOSP` vacío en el 98.5% de las atenciones | El vacío significa que no hubo hospitalización, así que se toma como 0 |
| Presión arterial vacía en el 49.8% de las atenciones | El vacío significa que no se midió. No se imputa por atención |
| Valores imposibles (edad 119, sistólica 400, `DIAS_HOSP` 1,467, cantidades de millones) | Se corrigen con rangos fisiológicos o se recortan por percentil |
| Duplicados exactos en Diagnósticos (4,489) y Procedimientos (7,836) | Se eliminan porque inflarían los conteos |
| `FECATE_POST_FECFED` no indica fallecimiento (20.2 M "SI" frente a 31 K fallecidos) | Se elimina. El fallecimiento se toma de `DESTINO_ASEGURADO = 'Fallecido'` |
| Miles de códigos distintos (hasta 16,271 en insumos), aunque el top 100–500 cubre del 80% al 99% | No se crea una columna por código. Se usan agregados (conteos y gasto) |
| 144,968 pacientes se atendieron en más de un departamento | La ubicación del paciente se resume con la **moda** de sus atenciones |

## 3. Construcción del dataset por paciente (`preprocesamiento.ipynb`)

**Ventanas de tiempo sin superposición:** las variables X salen de 2022–2023 y el objetivo y sale de 2024. *Justificación:* así no se usan datos del futuro para predecir el futuro.

**Selección de la cohorte:**

| Paso | Pacientes |
|---|---|
| Pacientes en los datos | 1,026,473 |
| Con hipertensión (CIE-10 I10–I15) en 2022–2023 | 632,115 |
| Sin complicación previa (ACV, IAM, IC o ERC en 2022–2023) | 552,723 |
| Con atenciones en 2024 (**cohorte final**) | **423,314** |

- Se excluye a quienes ya tenían una complicación porque el objetivo es predecir complicaciones **nuevas**.
- Se exige al menos una atención en 2024 porque, sin ella, no se sabe si el paciente tuvo la complicación.

**Variable objetivo `y`:** un diagnóstico (definitivo o repetido) de ACV/AIT (I60–I64, G45), infarto (I21–I23), insuficiencia cardíaca (I50) o enfermedad renal (N17–N19), o un fallecimiento, en 2024. Resultado: **22,389 positivos (5.29%)**.

- El 15.52% calculado en el EDA contaba diabetes, dislipidemia y obesidad como complicaciones. Aquí son **antecedentes**, porque son enfermedades crónicas previas y no eventos agudos.

**Limpieza:**
- Solo se usan diagnósticos `DEFINITIVO` y `REPETIDO` (los presuntivos tienen más ruido).
- Se eliminan los duplicados exactos.
- Edades fuera de 14–110 y presiones fuera de rango fisiológico (o con sistólica ≤ diastólica) pasan a NaN. `DIAS_HOSP` vacío pasa a 0, con un máximo de 365.
- `VALOR_NETO` se recorta al percentil 99.9 para quitar errores de digitación.

**Variables (30 en el dataset, 28 tras quitar dos redundantes en el modelado):**

| Grupo | Variables |
|---|---|
| Demográficas | `edad` (última atención), `masculino` |
| Ubicación | `es_lima_frecuente`: 1 si la **moda** del departamento del paciente es Lima. Usa la moda y no el último registro porque es más robusta a atenciones esporádicas en otro lugar |
| | `altitud_msnm`: altitud del distrito (ubigeo) más frecuente del paciente. Se toma de la tabla INEI de la librería *ubigeos-peru* y se cruza por **código**, porque su función por nombre asigna la misma altitud a distritos homónimos |
| Uso de servicios | `n_atenciones`, `n_emergencias`, `n_hospitalizaciones`, `dias_hosp`* |
| Presión arterial | `n_mediciones_pa`, `pas_media`, `pas_max`, `pad_media`*, `pad_max`, `prop_pa_alta` (proporción de mediciones ≥140/90) |
| | **`presion_pulso`** (media y máx.) = sistólica − diastólica. En hipertensos mayores de 50 años indica rigidez arterial y predice infarto y ACV |
| Antecedentes | `diabetes`, `dislipidemia`, `obesidad`, `fib_auricular`, `cardiopatia`, `n_diagnosticos` |
| Consumos | `n_` y `gasto_` de medicamentos, procedimientos e insumos, y `polifarmacia` (5 o más medicamentos distintos) |
| Indicador | `sin_medicion_pa`: 1 si al paciente nunca se le midió la presión |

\* Eliminadas después de analizar la correlación (sección 4).

**Decisiones de diseño:**
- **No se escala ni se imputa** en el dataset consolidado; todo queda en unidades originales. *Justificación:* el escalado y la imputación se ajustan en el modelado **solo con train**, así se evita la fuga de información.
- La ubicación (`UBIGEO`, `DEPARTAMENTO`) corresponde al **establecimiento** de atención, no al domicilio. Es una limitación.
- Se leen solo las columnas necesarias, por bloques y con tipos `category`, para procesar 1 M de pacientes sin agotar la memoria.

## 4. Preprocesamiento para el modelado (`03_comparacion_modelos.ipynb`)

| Paso | Justificación |
|---|---|
| División train/test 80/20 estratificada (`SEED = 42`) | Mantiene el 5.29% de positivos en ambos conjuntos |
| **Imputación por mediana**, ajustada solo con train | Solo faltan las variables de presión (3.9%, pacientes sin medición) y 3 edades. Se usa la mediana porque es robusta a valores extremos. No se borran filas porque estos pacientes tienen más complicaciones (8.4% frente a 5.2%). `sin_medicion_pa` conserva la información de que el dato no existía |
| **Matriz de correlación** (Pearson y Spearman, solo con train) → se eliminan `dias_hosp` y `pad_media` | `dias_hosp` ≈ `n_hospitalizaciones` (Spearman 0.997). `pad_media` es redundante porque `presion_pulso = pas_media − pad_media`, y es la de menor relación con `y`. Los demás pares (< 0.9) se mantienen |
| **Estandarización** (`StandardScaler`, ajustado con train), solo para KNN y Regresión Logística | KNN usa distancias y la RL se optimiza por gradiente. Los árboles no la necesitan |
| **`class_weight='balanced'`** (Árbol y RL); test **sin balancear** | Con 5% de positivos, el modelo no puede ganar diciendo siempre "sin complicación". No se usa SMOTE para no crear pacientes sintéticos |
| Hiperparámetros elegidos por **ROC-AUC** en validación | Con este desbalance, el accuracy engaña: predecir siempre 0 da 94.7% |

## 5. Modelos y resultados

**Hiperparámetros elegidos (en validación):**
- **KNN, k = 101:** el mayor de la grilla. Con k pequeño memoriza el train (brecha train–val de 0.25 con k=15).
- **Árbol, profundidad = 7:** empata con la profundidad 5 (+0.0002). Desde profundidad 10 hay sobreajuste claro.
- **Regresión Logística, lbfgs, C = 0.01:** las 12 combinaciones de solver y C dan el mismo AUC, porque con 270 K pacientes la regularización casi no influye.

**Resultados en test (84,663 pacientes):**

| Modelo | Accuracy | Precision | Recall | F1 | **ROC-AUC test** | ROC-AUC CV (5 folds) |
|---|---|---|---|---|---|---|
| KNN (k=101) | 0.947 | 0.000 | 0.000 | 0.000 | 0.669 | 0.652 ± 0.011 |
| Árbol (prof. 7) | 0.630 | 0.087 | 0.634 | 0.153 | 0.683 | 0.650 ± 0.010 |
| **Regresión Logística** | 0.654 | 0.092 | 0.628 | 0.161 | **0.695** | **0.693 ± 0.009** |

**Conclusiones:**
1. **La Regresión Logística es el mejor modelo:** tiene el mayor AUC en test y en validación cruzada, es la más estable y no sobreajusta.
2. **El accuracy engaña.** KNN tiene el mayor accuracy porque no detecta ninguna complicación. Con `balanced`, el Árbol y la RL detectan ≈63% de los pacientes que se complicarán, con muchos falsos positivos (precision ≈ 9%).
3. **KNN es el modelo más débil para datos tabulares médicos y administrativos.** Su mejor k siempre fue el máximo de la grilla, por tres razones:
   - La **alta dimensionalidad**: las distancias pierden sentido y todas las variables pesan igual.
   - El **desbalance**: entre los vecinos casi no hay positivos, y no admite `class_weight`.
   - El **costo de inferencia**: calcula la distancia a unos 340 K pacientes por cada predicción.
4. **La capacidad predictiva es moderada (AUC ≈ 0.67–0.69).** Es esperable sin datos de laboratorio (creatinina, colesterol, HbA1c). Ningún AUC pasa de 0.85, así que no hay señales de fuga de información.
5. **Eliminar las columnas redundantes no cambió el AUC de la RL** (0.6947), lo que confirma que lo eran. El modelo quedó más simple.

**Limitaciones:**
- Solo se observa a los pacientes que vuelven a atenderse en 2024.
- La ubicación es la del establecimiento.
- No hay datos de laboratorio ni un catálogo para identificar los antihipertensivos.
- El gasto (`VALOR_NETO`) es 0 en muchas entregas.

**Siguientes pasos:**
- Ajustar el umbral de decisión según el costo clínico de no detectar a un paciente.
- Aplicar `log1p` a gastos y conteos.
- Incorporar laboratorio y grupos terapéuticos.
