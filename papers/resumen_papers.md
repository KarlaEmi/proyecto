**Paper 1: Personalized prediction of incident hospitalization for cardiovascular disease in patients with hypertension using machine learning**

**Características:** 

- Las variables que mostraron mayor impacto predictivo en los modelos **incluyeron la edad,** **antecedentes cardiovasculares previos, la cantidad y tipo de medicamentos prescritos, la frecuencia de uso de servicios de salud (consultas y emergencias)** y ciertos parámetros de laboratorio.

- El aporte principal del artículo de Feng es demostrar que no se requieren variables clínicas directas de consultorio para construir un modelo predictivo de alta precisión ($Cindex=0.82$).

- El incluir el estado de comorbilidad, datos demográficos, resultados de pruebas de laboratorio y medicación mejoraría la evaluación del pronóstico y orientará las decisiones de tratamiento para pacientes con hipertensión. 

- Los árboles de decisión alimentados con variables agregadas de las 5 tablas del SIS (conteo de atenciones, comorbilidades CIE-10 y dispensación de medicamentos) replican la misma estructura de información que Feng utilizó para predecir quiénes terminarán hospitalizados por infarto, accidente cerebrovascular o insuficiencia cardíaca.

**Importante para el trabajo:**

- Hace hincapié en la importancia de definir correctamente nuestro target, el paciente con hipertensión. Define casos recientes de hipertensión, descarga a personas con ECV diagnosticados previamente. 

- Por otro lado, menciona la importancia del random forest y la baja correlación lineal existente entre los datos  y con ello llega a que mejor son los árboles. **(Superioridad de enfoques no lineales)**

- El artículo de Feng aporta el mismo desafío que tenemos respecto a nuestro dataset: cómo predecir el riesgo de complicaciones cardiovasculares en hipertensos utilizando únicamente registros administrativos de salud (sin depender de datos clínicos en tiempo real como tomas continuas de presión) **(Aprovechamiento de fuentes administrativas).** 

**Paper 2: EN Events Prediction Ability in Patients with Hypertension using Artificial Neural Network Analysis of Ambulatory Blood Pressure Monitoring Compared to Clinical Risk Stratification**

**Características:**

- Las variables que mostraron el mayor impacto predictivo en el modelo de red neuronal incluyeron la presión arterial sistólica nocturna (PASmNOCHE / night mSBP), la cual obtuvo el 100% de importancia normalizada independiente. Le siguieron en relevancia la PAS promedio de 24 horas, el índice de masa corporal (IMC), la variabilidad de la presión arterial media de 24 horas, la PAM nocturna y la presión de pulso de 24 horas.

- Problemática resuelta: El aporte principal del artículo de Di Gennaro et al. es demostrar que integrar las 27 variables hemodinámicas continuas y circadianas del Monitoreo Ambulatorio de Presión Arterial (MAPA) mediante Redes Neuronales Artificiales (RNA) supera de forma estadísticamente significativa a la Estratificación de Riesgo Clínica (EC) convencional recomendada por las guías clínicas. El modelo de RNA incrementó la capacidad discriminativa para predecir eventos graves de un AUC-ROC \= 0,67 (estratificación clínica) a un AUC-ROC \= 0,81 (RNA Perceptrón Multicapa 27/7/2; p \< 0.001).

- Importancia de la aplicación: Incorporar la variabilidad hemodinámica de 24 horas, las cargas hipertensivas y el patrón circadiano nocturno (non-dipping) permite superar los sesgos e imprecisiones inherentes a las tomas aisladas de presión en consultorio. La implementación de estas arquitecturas neuronales en software (por ejemplo, mediante algoritmos en Python conectados a historias clínicas electrónicas o a los reportes de MAPA) permite calcular automáticamente la probabilidad individual de sufrir eventos cardiovasculares graves, orientando decisiones terapéuticas preventivas más oportunas.

**Importante para el trabajo:**

- Definición precisa del target: Establece como variable dependiente un punto final combinado de eventos graves (Serious Events), compuesto por la primera ocurrencia de muerte, accidente cerebrovascular (ACV/AIT), infarto agudo de miocardio (IAM), insuficiencia cardíaca e insuficiencia renal en un seguimiento mediano de 6,6 años.

- Superioridad de enfoques no lineales: Resalta la capacidad superior de los algoritmos de redes neuronales (Perceptrón Multicapa con funciones de activación tangente hiperbólica y softmax) para capturar interacciones fisiológicas no lineales entre múltiples parámetros hemodinámicos que la regresión logística tradicional no puede detectar.

- Aprovechamiento de datos hemodinámicos subutilizados: Enfrenta el desafío de transformar registros de monitoreo funcional (MAPA) en un predictor continuo de riesgo, destacando que el comportamiento hemodinámico durante el sueño es la señal fisiológica más crítica para prevenir complicaciones graves