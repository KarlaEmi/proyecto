# Predicción de complicaciones en pacientes hipertensos asegurados en el SIS (Perú)

Proyecto del curso **Inteligencia Artificial Aplicada (1INF62)**, PUCP, semestre 2026-2.
Equipo: Karla García Rojas, Erich León Flórez y Rasec Gonzales Leyva.

> **Pregunta:** a partir de las atenciones de 2022–2023 de un paciente hipertenso del Seguro Integral de Salud (SIS), ¿se puede predecir si tendrá en **2024** una complicación grave (ACV/AIT, infarto, insuficiencia cardíaca, enfermedad renal o fallecimiento)?

---

## Índice

1. [Problemática](#1-problemática)
2. [Objetivo y definición del problema](#2-objetivo-y-definición-del-problema)
3. [Datos](#3-datos)
4. [Estructura del repositorio](#4-estructura-del-repositorio)
5. [Dependencias e instalación](#5-dependencias-e-instalación)
6. [Cómo reproducir el flujo](#6-cómo-reproducir-el-flujo)
7. [Resultados](#7-resultados)
8. [Limitaciones](#8-limitaciones)
9. [Referencias](#9-referencias)

---

## 1. Problemática

### 1.1 Una enfermedad frecuente, silenciosa y mal controlada

La hipertensión arterial (HTA) se define como una presión arterial de **140/90 mmHg o más**. Para diagnosticarla, la sistólica debe ser ≥ 140 mmHg o la diastólica ≥ 90 mmHg en dos días distintos [1]. Casi nunca da síntomas, por eso la única forma de detectarla es medir la presión.

Según la Organización Mundial de la Salud (OMS) [1, 2]:

- En 2024 tenían hipertensión unos **1,400 millones de adultos de 30 a 79 años**. Es el **33%** de las personas de esa edad.
- Dos tercios de ellos viven en países de ingresos bajos y medios, como el Perú.
- El **44%** no sabe que tiene hipertensión. Solo el **23%** la tiene controlada.

El número de personas con hipertensión **se duplicó entre 1990 y 2019**: pasó de unos 650 millones a unos 1,280 millones de adultos de 30 a 79 años. Gran parte de este aumento se debe al crecimiento y al envejecimiento de la población [3].

### 1.2 Por qué importan las complicaciones

Si la hipertensión no se controla, daña varios órganos [1]:

| Órgano | Complicación | Códigos CIE-10 usados en este proyecto |
|---|---|---|
| Cerebro | Accidente cerebrovascular (ACV) y accidente isquémico transitorio (AIT) | I60–I64, G45 |
| Corazón | Infarto agudo de miocardio (IAM) | I21–I23 |
| Corazón | Insuficiencia cardíaca (IC) | I50, I11.0, I13.0, I13.2 |
| Riñón | Enfermedad renal, que puede llegar a insuficiencia renal | N17–N19, I12, I13.1, I13.2 |
| — | Fallecimiento | `DESTINO_ASEGURADO = 'Fallecido'` |

El estudio Global Burden of Disease (GBD 2019) comparó **87 factores de riesgo** en 204 países. La **presión sistólica elevada** fue el principal factor de riesgo de muerte en el mundo. Se le atribuyen **10.8 millones de muertes** y **235 millones de años de vida ajustados por discapacidad (AVAD)** en 2019, más que al tabaco o a la glucosa elevada [4].

Las complicaciones también se pueden **prevenir**. Un metaanálisis de 123 ensayos con 613,815 participantes encontró que bajar la sistólica en 10 mmHg reduce [5]:

- 20% los eventos cardiovasculares mayores,
- 27% los ACV,
- 28% la insuficiencia cardíaca,
- 13% la mortalidad total.

Por eso, **identificar a tiempo a los pacientes con más riesgo** permite priorizar su seguimiento y su tratamiento.

### 1.3 El contexto peruano

- Con datos de la ENDES (2014–2022), la prevalencia de hipertensión en adultos peruanos es del **23%** [6].
- El subdiagnóstico es alto. Un estudio poblacional peruano estimó que el **69.5%** de los hipertensos no tenía diagnóstico [7].
- En la cohorte PERU MIGRANT, los hipertensos tuvieron **5.1 veces** más riesgo de morir en 10 años que las personas con presión normal [8].

El **SIS** asegura principalmente a la población sin otro seguro de salud. Por eso atiende a una población con menos recursos, donde el control de la presión suele ser más difícil.

### 1.4 Por qué usar aprendizaje automático con datos administrativos

En el SIS no hay datos de laboratorio ni monitoreo continuo de la presión. Sí hay **registros administrativos** de cada atención: diagnósticos CIE-10, medicamentos, procedimientos, insumos y algunas mediciones de presión.

La literatura muestra que esto puede ser suficiente para predecir el riesgo:

- Feng et al. usaron datos administrativos de 259,873 hipertensos de Alberta (Canadá). Sus modelos de *machine learning* predijeron hospitalizaciones por IC, IAM y ACV con un índice de concordancia de hasta **0.81** [9].
- Alimbayev et al. predijeron la mortalidad a un año de pacientes con diabetes en Kazajistán, también con datos administrativos. Los modelos de *gradient boosting* alcanzaron un AUC de **0.78–0.80**. La hipertensión fue una de las variables más importantes [10].

Este proyecto aplica la misma idea a los datos abiertos del SIS.

---

## 2. Objetivo y definición del problema

**Objetivo:** construir un modelo de clasificación binaria que estime, para cada paciente hipertenso, la probabilidad de una complicación grave en 2024.

```
|-------- 2022 y 2023 --------|---- 2024 ----|
  Variables (X): historial       Objetivo (y): ¿tuvo complicación?
```

- **Unidad de análisis:** el paciente (una fila por paciente). Así un mismo paciente no puede quedar a la vez en train y en test.
- **Cohorte:** pacientes con HTA (I10–I15) en 2022–2023 que:
  - no tuvieron una complicación previa,
  - no fallecieron antes de 2024,
  - tienen al menos una atención en 2024.
- **Objetivo `y`:** 1 si en 2024 aparece alguno de los eventos de la tabla 1.2. Solo se usan diagnósticos `DEFINITIVO` o `REPETIDO`.
- **Desbalance:** el 5.6% de los pacientes tiene `y = 1`. Por eso:
  - se evalúa con ROC-AUC, PR-AUC, Brier y la captura del 10% de mayor riesgo, y no con accuracy;
  - el baseline usa `class_weight='balanced'`, pero los modelos finales van sin pesos o recalibrados, porque los pesos inflan las probabilidades (ver la sección 7);
  - no se usa SMOTE como estrategia principal, porque no mejora el AUC y descalibra las probabilidades [11].

---

## 3. Datos

**Fuente:** *Prestaciones de salud asociadas a los asegurados con Hipertensión Arterial*, publicado por el SIS en la Plataforma Nacional de Datos Abiertos del Perú. Los datos están anonimizados.

Son 5 tablas, con unos **170 millones de filas** y **1.03 millones de pacientes**. Cubren atenciones de 2022 a 2024. Las tablas se unen por `ID_REGISTRO_REL`, el identificador de la atención.

| Archivo | Una fila es… | Filas aprox. |
|---|---|---|
| `1. HTA_FUAS` | una atención (paciente, fecha, edad, ubicación, presión arterial, hospitalización, destino) | 24.3 M |
| `2. HTA_Diagnosticos` | un diagnóstico CIE-10 de una atención | 38.1 M |
| `3. HTA_Medicamentos` | una entrega de medicamento | 39.4 M |
| `4. HTA_Procedimientos` | un procedimiento | 53.0 M |
| `5. HTA_Insumos` | una entrega de insumo | 15.8 M |

El diccionario de variables está en `data/raw/Metadatos - Prestaciones de salud asociadas a los asegurados con Hipertensión Arterial.docx`.

**Tabla auxiliar:** la altitud de cada distrito sale de `ubigeo_inei_2025.csv`, del repositorio [ubigeos_peru](https://github.com/MichaelSuarez0/ubigeos_peru). Se cruza por **código** de ubigeo, no por nombre, porque hay distritos con el mismo nombre en distintas regiones.

---

## 4. Estructura del repositorio

```
proyecto/
├── README.md                    ← este archivo
├── ENTREGABLE_PARCIAL.md        ← informe del avance parcial (cohorte, variables, baseline)
├── requirements.txt             ← dependencias de Python
├── presentation.pdf             ← presentación del proyecto (15 diapositivas)
├── .gitattributes               ← reglas de Git LFS
├── data/
│   ├── raw/                     ← datos originales del SIS (comprimidos, con Git LFS)
│   │   ├── 1. HTA_FUAS.zip
│   │   ├── 2. HTA_Diagnosticos.zip
│   │   ├── 3. HTA_Medicamentos.zip
│   │   ├── 4. HTA_Procedimientos.zip
│   │   ├── 5. HTA_Insumos.zip
│   │   └── Metadatos - ... .docx  ← diccionario de datos
│   └── processed/
│       └── dataset_consolidado.csv  ← una fila por paciente (salida del preprocesamiento)
├── notebooks/
│   ├── 01_eda.ipynb             ← análisis exploratorio de las 5 tablas
│   └── 02_modelos.ipynb         ← Partes 2 a 9: modelos, validación y comparación final
├── src/
│   ├── preprocessing.ipynb      ← limpieza, cohorte, objetivo y variables (genera el dataset consolidado)
│   ├── preprocessing.py         ← la misma lógica como módulo, con ventanas de tiempo y objetivos alternativos
│   ├── models.py                ← todos los modelos con los hiperparámetros elegidos
│   └── evaluation.py            ← métricas y generación de results/
├── results/
│   ├── metrics.csv              ← tabla comparativa de todos los modelos sobre el mismo test
│   ├── model_comparison.txt     ← análisis redactado de la comparación de modelos
│   └── plots/                   ← gráficos de resultados (00–18)
│       └── README.md            ← qué muestra cada gráfico
└── papers/
    ├── paper_integrante1.pdf    ← papers base revisados por el equipo
    ├── paper_integrante2.pdf
    ├── paper_integrante3.pdf
    └── resumen_papers.md        ← resumen y aporte de cada paper al proyecto
```

### Qué hace cada cuaderno y script

| Orden | Archivo | Entrada | Salida | Contenido |
|---|---|---|---|---|
| 1 | `notebooks/01_eda.ipynb` | 5 CSV crudos | gráficos y tablas | Tamaño y calidad de cada tabla, valores faltantes e imposibles, duplicados, distribución de diagnósticos, presión arterial y gasto. Verifica la cohorte y los códigos del objetivo |
| 2 | `src/preprocessing.ipynb` | 5 CSV crudos + `ubigeo_inei_2025.csv` | `dataset_consolidado.csv` | Lectura por bloques, limpieza con rangos fisiológicos, selección de la cohorte, construcción de `y` y agregación de variables por paciente (demografía, uso de servicios, presión arterial, antecedentes, consumos) |
| 3 | `notebooks/02_modelos.ipynb` | `dataset_consolidado.csv` (las Partes 4 y 6 también leen los CSV crudos) | predicciones e hiperparámetros en `data/processed/`, gráficos | Parte 2: KNN, árbol y regresión logística (baseline). Parte 3: Random Forest. Partes 4 y 5: secuencias mensuales y red recurrente (GRU). Parte 6: robustez, objetivos alternativos y validación temporal. Parte 7: gradient boosting y ensambles. Parte 8: perceptrón multicapa (MLP). Parte 9: comparación final |
| 4 | `src/evaluation.py` | predicciones de test guardadas por `02_modelos.ipynb` | `results/metrics.csv`, `results/plots/` | Calcula las métricas de todos los modelos sobre los mismos pacientes de test. Se ejecuta con `python src/evaluation.py` |

`src/preprocessing.py` y `src/models.py` son módulos que usa `02_modelos.ipynb`: el primero arma cohortes con otras ventanas de tiempo y el segundo reúne los modelos con sus hiperparámetros.

### Variables del dataset consolidado

`dataset_consolidado.csv` tiene **419,226 pacientes** y **36 variables** más el objetivo `y`. Todas se calculan con las atenciones de 2022–2023.

| Grupo | Variables |
|---|---|
| Demográficas | `edad`, `masculino` |
| Ubicación | `es_lima_frecuente`, `altitud_msnm` |
| Uso de servicios | `n_atenciones`, `n_emergencias`, `n_hospitalizaciones`, `dias_hosp`, `dias_desde_ultima` (días entre la última atención y el 31/12/2023), `nivel_eess_max` (nivel más alto de establecimiento visitado), `prop_capita` (proporción de atenciones pagadas por cápita) |
| Presión arterial | `n_mediciones_pa`, `pas_media`, `pas_max`, `pas_ultima`, `pad_media`, `pad_max`, `presion_pulso`, `presion_pulso_max`, `prop_pa_alta`, `sin_medicion_pa` |
| Antecedentes | `diabetes`, `dislipidemia`, `obesidad`, `fib_auricular`, `cardiopatia`, `hta_cardiaca` (I11.9), `n_comorbilidades`, `n_diagnosticos` |
| Consumos | `n_medicamentos`, `gasto_medicamentos`, `n_procedimientos`, `gasto_procedimientos`, `n_insumos`, `gasto_insumos`, `polifarmacia` |
| Objetivo | `y` |

Los modelos usan 34 de las 36 variables: quitan `dias_hosp` y `pad_media` por estar muy correlacionadas con otras.

---

## 5. Dependencias e instalación

### Requisitos

- **Python 3.10 o superior** (los cuadernos funcionan con Python 3.13).
- **Git LFS**, para descargar los `.zip` de `data/raw/`.
- **Memoria:** el preprocesamiento lee ≈170 M filas. Lee por bloques y usa tipos `category`, pero conviene tener **16 GB de RAM o más**. Con menos memoria, usa una muestra con el parámetro `FRAC` (por ejemplo, `FRAC = 0.05`).
- **Disco:** unos 10 GB libres para los CSV descomprimidos.

### Librerías

| Librería | Uso en el proyecto |
|---|---|
| `pandas` | Lectura por bloques, limpieza y agregación por paciente |
| `numpy` | Cálculos numéricos y validación de rangos (en `float64`) |
| `scipy` | Correlación de Spearman entre modelos y búsqueda de umbrales |
| `matplotlib`, `seaborn` | Gráficos del EDA, del preprocesamiento y de los modelos |
| `scikit-learn` | `train_test_split`, `StratifiedKFold`, `Pipeline`, `SimpleImputer`, `StandardScaler`, KNN, árbol, regresión logística, Random Forest, HistGradientBoosting y métricas |
| `torch` | Redes neuronales (GRU y MLP) |
| `jupyter` | Ejecutar los cuadernos |
| `lightgbm` *(opcional)* | Modelo LightGBM; si no está instalado, se omite |
| `optuna` *(opcional)* | Búsqueda de hiperparámetros; si no está instalado, se usa una búsqueda aleatoria |

Las versiones mínimas están en `requirements.txt`.

### Instalación

```bash
# 1. Clonar el repositorio y descargar los archivos grandes
git lfs install
git clone https://github.com/KarlaEmi/proyecto.git
cd proyecto
git lfs pull

# 2. Crear un entorno virtual
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

# 3. Instalar las dependencias
pip install -r requirements.txt

# 4. Descomprimir los datos crudos
#    Los .zip quedan en data/raw/ y contienen los CSV "1. HTA_FUAS.csv", etc.
```

Además, descarga `ubigeo_inei_2025.csv` desde [ubigeos_peru](https://github.com/MichaelSuarez0/ubigeos_peru) y colócalo en la misma carpeta que los CSV.

---

## 6. Cómo reproducir el flujo

Los cuadernos buscan solos las carpetas `data/raw/` y `data/processed/` desde la carpeta en que se abren. Si tus datos están en otro lugar, escribe la ruta en `RUTA_MANUAL` (o en las variables `RUTA_MANUAL_*` de cada parte de `02_modelos.ipynb`).

1. **EDA:** abre `notebooks/01_eda.ipynb` y ejecuta todas las celdas.
2. **Preprocesamiento:** abre `src/preprocessing.ipynb` y revisa los parámetros de la primera celda:
   - `FRAC`: fracción de pacientes. `None` = todos; con una fracción, la salida se llama `dataset_consolidado_muestra.csv`.
   - `FIN_OBS = 20231231` y `INI_RES = 20240101`: separan el periodo de las variables del periodo del objetivo.
   - `CHUNK`: filas por bloque al leer.

   El cuaderno guarda `dataset_consolidado.csv` en `data/processed/`. Si se activa alguno de sus controles de calidad (por ejemplo, demasiadas presiones anuladas), la ejecución se detiene con un mensaje.
3. **Modelos:** abre `notebooks/02_modelos.ipynb` y ejecuta las partes en orden. Cada parte guarda sus predicciones de test e hiperparámetros en `data/processed/`, que usan las partes siguientes.
   - Las Partes 4 y 6 vuelven a leer los CSV crudos de `data/raw/`.
   - La Parte 5 (GRU) entrena con el 30% de train (`FRAC_TRAIN = 0.3`); con `None` usa todo el train.
   - La Parte 8 (MLP) es la más lenta (unos 100 minutos). `N_TRIALS` controla cuántas configuraciones prueba.
4. **Resultados:** desde la raíz del repositorio, ejecuta:

   ```bash
   python src/evaluation.py
   ```

   Lee las predicciones guardadas y escribe `results/metrics.csv`, los gráficos de `results/plots/` y `results/resumen_generado.txt`. No sobrescribe `results/model_comparison.txt`, que es el análisis redactado.

La semilla es `SEED = 42` en todos los cuadernos.

---

## 7. Resultados

Todos los modelos se evaluaron sobre los **mismos 83,846 pacientes de test** (20% de la cohorte, partición estratificada), que se usaron una sola vez. Los hiperparámetros se eligieron solo con train. El azar da un PR-AUC de 0.056 y una captura del 10%.

| Modelo | Parte | ROC-AUC | PR-AUC | Brier | Captura del 10% de mayor riesgo |
|---|---|---|---|---|---|
| **Ensamble RF + boosting + GRU** (promedio) | 7 | **0.712** | 0.133 | 0.051 | 27.6% |
| Gradient boosting | 7 | 0.709 | 0.133 | 0.051 | 27.1% |
| MLP | 8 | 0.709 | 0.129 | 0.051 | 26.9% |
| Random Forest | 3 | 0.705 | 0.124 | 0.051 | 26.3% |
| GRU (secuencias mensuales) | 5 | 0.703 | 0.125 | 0.051 | 26.6% |
| Regresión logística (baseline, `class_weight='balanced'`) | 2 | 0.693 | 0.116 | 0.221 | 24.6% |
| KNN (k = 301) | 2 | 0.681 | 0.112 | 0.052 | 23.8% |
| Árbol de decisión (profundidad 10) | 2 | 0.675 | 0.102 | 0.225 | 22.4% |

La tabla completa, con intervalos de confianza y los demás ensambles, está en `results/metrics.csv`. El análisis está en `results/model_comparison.txt`.

**Conclusiones principales:**
- **Modelo recomendado: gradient boosting.** Rinde igual que el MLP, pero se entrena en segundos y da probabilidades calibradas.
- El **ensamble con la GRU** es el mejor resultado (+0.007 de ROC-AUC sobre el Random Forest, diferencia significativa), porque la GRU aprovecha el orden temporal de las atenciones.
- El 10% de pacientes de mayor riesgo concentra el **27% de las complicaciones**, 2.7 veces lo esperado al azar.
- `class_weight='balanced'` no mejora el ROC-AUC y multiplica el Brier por cuatro: las probabilidades salen infladas. Los modelos finales van sin pesos o recalibrados.
- Ocho algoritmos distintos quedan entre 0.675 y 0.709 por separado. El techo lo pone la información disponible (sin laboratorio, tabaquismo ni adherencia), no el algoritmo.

---

## 8. Limitaciones

- **Sesgo de detección:** solo se observan las complicaciones de los pacientes que vuelven a atenderse en el SIS. Un evento atendido en otro sistema de salud no aparece.
- **Ubicación:** `UBIGEO` y `DEPARTAMENTO` son los del establecimiento, no los del domicilio del paciente.
- **Gasto y cápita:** en las atenciones pagadas por cápita, el gasto registrado es siempre 0. Por eso las variables `gasto_*` reflejan más la forma de financiamiento que el consumo real. Se prefieren los conteos `n_*`.
- **Presión arterial:** falta en casi la mitad de las atenciones y se registra con preferencia por números redondos.
- **Sin laboratorio:** no hay datos de laboratorio ni un catálogo para identificar los antihipertensivos.

---

## 9. Referencias

**Organización Mundial de la Salud**

1. World Health Organization. *Hypertension* (nota descriptiva). 25 de septiembre de 2025. https://www.who.int/news-room/fact-sheets/detail/hypertension
2. World Health Organization. *Global report on hypertension 2025: high stakes – turning evidence into action*. Ginebra: OMS; 2025. https://www.who.int/publications/i/item/9789240115569

**Artículos indexados en PubMed**

3. NCD Risk Factor Collaboration (NCD-RisC). Worldwide trends in hypertension prevalence and progress in treatment and control from 1990 to 2019: a pooled analysis of 1201 population-representative studies with 104 million participants. *Lancet*. 2021;398(10304):957–980. doi:[10.1016/S0140-6736(21)01330-1](https://doi.org/10.1016/S0140-6736(21)01330-1). PMID: [34450083](https://pubmed.ncbi.nlm.nih.gov/34450083/)
4. GBD 2019 Risk Factors Collaborators. Global burden of 87 risk factors in 204 countries and territories, 1990–2019: a systematic analysis for the Global Burden of Disease Study 2019. *Lancet*. 2020;396(10258):1223–1249. doi:[10.1016/S0140-6736(20)30752-2](https://doi.org/10.1016/S0140-6736(20)30752-2). PMID: [33069327](https://pubmed.ncbi.nlm.nih.gov/33069327/)
5. Ettehad D, Emdin CA, Kiran A, et al. Blood pressure lowering for prevention of cardiovascular disease and death: a systematic review and meta-analysis. *Lancet*. 2016;387(10022):957–967. doi:[10.1016/S0140-6736(15)01225-8](https://doi.org/10.1016/S0140-6736(15)01225-8). PMID: [26724178](https://pubmed.ncbi.nlm.nih.gov/26724178/)
6. Prevalence, trends, and factors associated with hypertensive crisis among Peruvian adults. *Cadernos de Saúde Pública*. 2024;40(2):e00155123. https://scielosp.org/article/csp/2024.v40n2/e00155123
7. Factores asociados a hipertensión arterial no diagnosticada en el Perú. *Archivos de Cardiología de México*. 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12148522
8. Hidalgo-Benites A, Senosain-Leon V, Carrillo-Larco RM, et al. Blood pressure and 10-year all-cause mortality: Findings from the PERU MIGRANT Study. *F1000Research*. 2023;10:1134. doi:[10.12688/f1000research.73900.4](https://doi.org/10.12688/f1000research.73900.4). PMID: [38046986](https://pubmed.ncbi.nlm.nih.gov/38046986/)
9. Feng Y, Leung AA, Lu X, Liang Z, Quan H, Walker RL. Personalized prediction of incident hospitalization for cardiovascular disease in patients with hypertension using machine learning. *BMC Medical Research Methodology*. 2022;22(1):325. doi:[10.1186/s12874-022-01814-3](https://doi.org/10.1186/s12874-022-01814-3). PMID: [36528631](https://pubmed.ncbi.nlm.nih.gov/36528631/)
10. Alimbayev A, Zhakhina G, Gusmanov A, et al. Predicting 1-year mortality of patients with diabetes mellitus in Kazakhstan based on administrative health data using machine learning. *Scientific Reports*. 2023;13(1):8412. doi:[10.1038/s41598-023-35551-4](https://doi.org/10.1038/s41598-023-35551-4). PMID: [37225754](https://pubmed.ncbi.nlm.nih.gov/37225754/)
11. van den Goorbergh R, van Smeden M, Timmerman D, Van Calster B. The harm of class imbalance corrections for risk prediction models: illustration and simulation using logistic regression. *Journal of the American Medical Informatics Association*. 2022;29(9):1525–1534. doi:[10.1093/jamia/ocac093](https://doi.org/10.1093/jamia/ocac093). PMID: [35686364](https://pubmed.ncbi.nlm.nih.gov/35686364/)

**Datos**

12. Seguro Integral de Salud (SIS). *Prestaciones de salud asociadas a los asegurados con Hipertensión Arterial*. Plataforma Nacional de Datos Abiertos del Perú.
