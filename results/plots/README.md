# results/plots/

| Archivo | Qué muestra | Parte |
|---|---|---|
| `00_comparacion_roc_auc.png` | ROC-AUC de todos los modelos y diferencia frente al Random Forest (IC 95%) | Resumen |
| `00_comparacion_metricas.png` | PR-AUC, Brier y captura del 10% de mayor riesgo | Resumen |
| `01_curvas_roc_modelos_curso.png` | Curvas ROC de KNN, árbol y regresión logística | 2 |
| `02_calibracion_modelos_curso.png` | Calibración de los modelos del curso (efecto de `class_weight`) | 2 |
| `03_captura_por_deciles_modelos_curso.png` | Complicaciones capturadas por decil de riesgo | 2 |
| `04_validacion_cruzada_modelos_curso.png` | Variabilidad en validación cruzada | 2 |
| `05_random_forest_vs_regresion_logistica.png` | Bootstrap pareado RF − regresión logística | 3 |
| `06_random_forest_calibracion.png` | Calibración del RF frente a la regresión logística | 3 |
| `07_random_forest_importancia_variables.png` | Importancia por permutación y MDI | 3 |
| `08_gru_importancia_por_trimestre.png` | Qué meses usa la GRU | 5 |
| `09_gru_vs_modelos_tabulares.png` | Curvas ROC de la GRU frente al RF y la regresión logística | 5 |
| `10_modelos_de_referencia.png` | Edad y sexo, clínico mínimo, regresión logística y RF | 6 |
| `11_curva_de_decision.png` | Beneficio neto según el umbral de riesgo | 6 |
| `12_objetivos_alternativos.png` | AUC con el evento grave, renal, cardiovascular y fallecimiento | 6 |
| `13_validacion_temporal.png` | Desempeño y calibración en el año siguiente | 6 |
| `14_boosting_importancia_variables.png` | Importancia por permutación del gradient boosting | 7 |
| `15_correlacion_entre_modelos.png` | Correlación entre las predicciones de los modelos | 7 |
| `16_ensambles_vs_random_forest.png` | ROC-AUC de modelos y ensambles con IC 95% | 7 |
| `17_calibracion_y_beneficio_neto_ensamble.png` | Calibración y curva de decisión del mejor ensamble | 7 |
| `18_mlp_roc_calibracion.png` | Curva ROC, calibración y matriz de confusión del MLP | 8 |
