"""
Métricas y evaluación del proyecto.

Funciones reutilizables (las mismas definiciones que usa notebooks/02_modelos.ipynb):
    metricas(y, prob)                     ROC-AUC, PR-AUC, Brier, captura y precisión en el top 10%
    bootstrap_auc(y, probs, referencia)   diferencias de ROC-AUC con IC 95% por bootstrap pareado
    beneficio_neto(y, prob, umbrales)     curva de decisión (Vickers y Elkin, 2006)
    captura_por_decil(y, prob)            % de complicaciones en cada decil de riesgo

Generación de los resultados oficiales del repositorio (results/):
    python src/evaluation.py

    Lee las predicciones de test que guarda cada parte de notebooks/02_modelos.ipynb en data/processed/ (todas sobre los mismos 83,846
    pacientes) y escribe:
        results/metrics.csv              tabla comparativa de todos los modelos
        results/plots/*.png              curvas ROC, comparación con IC, calibración, curva de decisión, deciles
        results/resumen_generado.txt     resumen automático (results/model_comparison.txt es el análisis
                                         redactado, y no se sobrescribe)
"""
import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve

SEED = 42

# Archivo de predicciones -> (columna, nombre del modelo, tipo, parte de notebooks/02_modelos.ipynb)
FUENTES = [
    ('predicciones_test_02.csv', 'prob_knn', 'KNN', 'Modelo', 'Parte 2'),
    ('predicciones_test_02.csv', 'prob_arbol', 'Árbol de decisión', 'Modelo', 'Parte 2'),
    ('predicciones_test_02.csv', 'prob_rl', 'Regresión logística (balanced)', 'Baseline', 'Parte 2'),
    ('predicciones_test_rf_rl.csv', 'prob_rl_sin_pesos', 'Regresión logística', 'Modelo', 'Parte 3'),
    ('predicciones_test_rf_rl.csv', 'prob_rf', 'Random Forest', 'Modelo', 'Parte 3'),
    ('predicciones_test_gru.csv', 'prob_gru', 'GRU', 'Modelo', 'Parte 5'),
    ('predicciones_test_gru.csv', 'prob_lstm', 'LSTM', 'Modelo', 'Parte 5'),
    ('predicciones_test_gru.csv', 'prob_mlp_control', 'MLP de control (secuencias)', 'Modelo', 'Parte 5'),
    ('predicciones_test_ensambles.csv', 'prob_gradient_boosting', 'Gradient boosting', 'Modelo', 'Parte 7'),
    ('predicciones_test_ensambles.csv', 'prob_lightgbm', 'LightGBM', 'Modelo', 'Parte 7'),
    ('predicciones_test_mlp.csv', 'prob_mlp', 'MLP', 'Modelo', 'Parte 8'),
]
# Ensambles por promedio simple que se recalculan aquí (partes 7 y 8)
ENSAMBLES = {
    'Ensamble RF + GB + GRU': ['Random Forest', 'Gradient boosting', 'GRU'],
    'Ensamble RF + GB + GRU + MLP': ['Random Forest', 'Gradient boosting', 'GRU', 'MLP'],
}
BASELINE_PREFERIDO = ['Regresión logística (balanced)', 'Regresión logística']


# ------------------------------------------------------------------------------------------ métricas
def metricas(y, prob, top=0.10):
    """Métricas de discriminación (ROC-AUC, PR-AUC), calibración (Brier) y utilidad (top 10%)."""
    y, prob = np.asarray(y), np.asarray(prob)
    k = int(len(prob) * top)
    sel = np.argsort(-prob)[:k]
    return {'ROC-AUC': roc_auc_score(y, prob), 'PR-AUC': average_precision_score(y, prob),
            'Brier': brier_score_loss(y, prob),
            'Captura top 10% (%)': y[sel].sum() / y.sum() * 100,
            'Precisión top 10% (%)': y[sel].mean() * 100}


def bootstrap_auc(y, probs, referencia, n=500, seed=SEED):
    """Δ ROC-AUC de cada modelo frente a `referencia`, con IC 95% por bootstrap PAREADO (mismos pacientes)."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    filas = []
    for _ in range(n):
        b = rng.integers(0, len(y), len(y))
        if y[b].min() == y[b].max():
            continue
        ref = roc_auc_score(y[b], probs[referencia][b])
        filas.append({k: roc_auc_score(y[b], p[b]) - ref for k, p in probs.items()})
    d = pd.DataFrame(filas)
    inf, sup = d.quantile(0.025), d.quantile(0.975)
    return pd.DataFrame({'Δ ROC-AUC': d.mean(), 'IC 95% inf': inf, 'IC 95% sup': sup,
                         'Significativa': np.where((inf > 0) | (sup < 0), 'Sí', 'No')})


def intervalo_auc(y, prob, n=500, seed=SEED):
    """IC 95% bootstrap del ROC-AUC de un modelo."""
    y, prob = np.asarray(y), np.asarray(prob)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        b = rng.integers(0, len(y), len(y))
        if y[b].min() != y[b].max():
            aucs.append(roc_auc_score(y[b], prob[b]))
    return np.quantile(aucs, [0.025, 0.975])


def beneficio_neto(y, prob, umbrales):
    """Beneficio neto de seguir a quienes superan cada umbral: VP/n − FP/n · t/(1−t)."""
    y, prob = np.asarray(y), np.asarray(prob)
    n = len(y)
    out = []
    for t in umbrales:
        marca = prob >= t
        out.append(((marca & (y == 1)).sum() - (marca & (y == 0)).sum() * t / (1 - t)) / n)
    return np.array(out)


def captura_por_decil(y, prob):
    """% del total de complicaciones que cae en cada decil de riesgo (1 = 10% de mayor riesgo)."""
    y, prob = np.asarray(y), np.asarray(prob)
    orden = np.argsort(-prob)
    deciles = np.array_split(orden, 10)
    return np.array([y[d].sum() / y.sum() * 100 for d in deciles])


# ------------------------------------------------------------------------------- resultados oficiales
def buscar_raiz(inicio=None):
    inicio = Path(inicio) if inicio else Path.cwd()
    for p in [inicio, *inicio.parents]:
        if (p / 'data' / 'processed').is_dir():
            return p
    raise FileNotFoundError("No se encontró la carpeta data/processed/ (ejecuta desde el repositorio).")


def recolectar_predicciones(ruta_proc):
    """Une en una tabla (una fila por paciente de test) las probabilidades guardadas por cada parte."""
    ruta_proc = Path(ruta_proc)
    tabla, info, y_ref = None, {}, None
    for archivo, col, nombre, tipo, cuaderno in FUENTES:
        f = ruta_proc / archivo
        if not f.exists():
            continue
        d = pd.read_csv(f)
        if col not in d.columns:
            continue
        d['paciente'] = d['paciente'].astype(str)
        d = d.set_index('paciente')
        if tabla is None:
            tabla = d[['y']].copy()
        if set(d.index) != set(tabla.index):
            print(f"  Aviso: {archivo} tiene otros pacientes de test; se omite {nombre}")
            continue
        d = d.loc[tabla.index]
        if not (d['y'].to_numpy() == tabla['y'].to_numpy()).all():
            print(f"  Aviso: el y de {archivo} no coincide; se omite {nombre}")
            continue
        tabla[nombre] = d[col].to_numpy()
        info[nombre] = (tipo, cuaderno)
    if tabla is None:
        raise FileNotFoundError(f"No hay archivos de predicciones en {ruta_proc}: ejecuta notebooks/02_modelos.ipynb.")
    for nombre, comp in ENSAMBLES.items():
        if all(c in tabla.columns for c in comp):
            tabla[nombre] = tabla[comp].mean(axis=1)
            info[nombre] = ('Ensamble', 'Parte 7' if 'MLP' not in comp else 'Parte 8')
    return tabla, info


def tabla_comparativa(tabla, info, n_boot=500):
    y = tabla['y'].to_numpy()
    modelos = [c for c in tabla.columns if c != 'y']
    probs = {m: tabla[m].to_numpy() for m in modelos}
    filas = []
    for m in modelos:
        inf, sup = intervalo_auc(y, probs[m], n=n_boot)
        filas.append({'Modelo': m, 'Tipo': info[m][0], 'Parte': info[m][1], **metricas(y, probs[m]),
                      'ROC-AUC IC 95% inf': inf, 'ROC-AUC IC 95% sup': sup})
    res = pd.DataFrame(filas).set_index('Modelo')
    baseline = next((b for b in BASELINE_PREFERIDO if b in probs), modelos[0])
    d_base = bootstrap_auc(y, probs, baseline, n=n_boot)
    res['Δ ROC-AUC vs baseline'] = d_base['Δ ROC-AUC']
    res['Significativa vs baseline'] = d_base['Significativa']
    if 'Random Forest' in probs:
        d_rf = bootstrap_auc(y, probs, 'Random Forest', n=n_boot)
        res['Δ ROC-AUC vs RF'] = d_rf['Δ ROC-AUC']
        res['IC 95% inf vs RF'] = d_rf['IC 95% inf']
        res['IC 95% sup vs RF'] = d_rf['IC 95% sup']
        res['Significativa vs RF'] = d_rf['Significativa']
    res.loc[baseline, ['Significativa vs baseline']] = '—'
    return res.sort_values('ROC-AUC', ascending=False), baseline


def graficos(tabla, res, carpeta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    AZUL, NARANJA, AQUA, MORADO, GRIS, TEXTO, TEXTO_2 = '#2a78d6', '#eb6834', '#1baf7a', '#7e5bd6', '#a3a29b', '#0b0b0b', '#52514e'
    plt.rcParams.update({
        'figure.dpi': 130, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#b5b4ad', 'axes.titleweight': 'bold', 'axes.titlesize': 12, 'axes.titlecolor': TEXTO,
        'axes.labelcolor': TEXTO_2, 'xtick.color': TEXTO_2, 'ytick.color': TEXTO_2, 'axes.grid': True,
        'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'legend.frameon': False})
    color_tipo = {'Baseline': GRIS, 'Modelo': AZUL, 'Ensamble': NARANJA}
    y = tabla['y'].to_numpy()
    archivos = []

    # 1. ROC-AUC con IC 95%
    r = res.sort_values('ROC-AUC')
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(r) + 1.4))
    ax.errorbar(r['ROC-AUC'], np.arange(len(r)), xerr=[r['ROC-AUC'] - r['ROC-AUC IC 95% inf'],
                r['ROC-AUC IC 95% sup'] - r['ROC-AUC']], fmt='none', ecolor=GRIS, elinewidth=2, capsize=3)
    ax.scatter(r['ROC-AUC'], np.arange(len(r)), c=[color_tipo[t] for t in r['Tipo']], s=55, zorder=3)
    for i, v in enumerate(r['ROC-AUC']):
        ax.text(r['ROC-AUC IC 95% sup'].iloc[i] + 0.001, i, f'{v:.3f}', va='center', fontsize=9, color=TEXTO_2)
    ax.set_yticks(np.arange(len(r)), r.index)
    ax.set(title='ROC-AUC en test con IC 95% (gris: baseline · azul: modelo · naranja: ensamble)', xlabel='ROC-AUC')
    plt.tight_layout()
    archivos.append(carpeta / 'comparacion_roc_auc.png')
    fig.savefig(archivos[-1], bbox_inches='tight')
    plt.close(fig)

    # 2. Curvas ROC de los modelos individuales principales
    principales = [m for m in ['Random Forest', 'Gradient boosting', 'GRU', 'MLP', 'Regresión logística', 'KNN',
                               'Árbol de decisión', 'Ensamble RF + GB + GRU'] if m in tabla.columns]
    paleta = [AZUL, AQUA, NARANJA, MORADO, GRIS, '#c9a227', '#8a8a85', TEXTO]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for m, c in zip(principales, paleta):
        fpr, tpr, _ = roc_curve(y, tabla[m])
        ax.plot(fpr, tpr, color=c, linewidth=2.2 if 'Ensamble' in m else 1.5, label=f"{m} ({res.loc[m, 'ROC-AUC']:.3f})")
    ax.plot([0, 1], [0, 1], color=GRIS, linestyle='--', linewidth=1)
    ax.set(title='Curvas ROC en test', xlabel='Tasa de falsos positivos', ylabel='Sensibilidad (recall)')
    ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    archivos.append(carpeta / 'curvas_roc.png')
    fig.savefig(archivos[-1], bbox_inches='tight')
    plt.close(fig)

    # 3. Calibración
    calib = [m for m in ['Regresión logística (balanced)', 'Regresión logística', 'Random Forest', 'Gradient boosting',
                         'Ensamble RF + GB + GRU'] if m in tabla.columns]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for m, c in zip(calib, [GRIS, TEXTO_2, AZUL, AQUA, NARANJA]):
        real, pred = calibration_curve(y, tabla[m], n_bins=10, strategy='quantile')
        ax = axes[0] if 'balanced' in m else axes[1]
        ax.plot(pred, real, marker='o', color=c, linewidth=2, label=f"{m} (Brier {res.loc[m, 'Brier']:.4f})")
    for ax in axes:
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], color=GRIS, linestyle='--', linewidth=1, label='Calibración perfecta')
        ax.set(xlabel='Probabilidad predicha', ylabel='Proporción real')
        ax.legend(fontsize=8)
    axes[0].set_title('Con pesos de clase (Parte 2): probabilidades infladas')
    axes[1].set_title('Sin pesos de clase (partes 3-8): calibrados')
    plt.tight_layout()
    archivos.append(carpeta / 'calibracion.png')
    fig.savefig(archivos[-1], bbox_inches='tight')
    plt.close(fig)

    # 4. Curva de decisión
    umbrales = np.linspace(0.01, 0.30, 59)
    prev = y.mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, c in zip([m for m in ['Random Forest', 'Gradient boosting', 'Regresión logística', 'Ensamble RF + GB + GRU']
                     if m in tabla.columns], [AZUL, AQUA, GRIS, NARANJA]):
        ax.plot(umbrales, beneficio_neto(y, tabla[m], umbrales), color=c, linewidth=2, label=m)
    ax.plot(umbrales, prev - (1 - prev) * umbrales / (1 - umbrales), color=GRIS, linestyle='--', label='Seguir a todos')
    ax.axhline(0, color=TEXTO, linewidth=1, label='No seguir a nadie')
    ax.axvspan(0.03, 0.15, color='#f2f1ec', zorder=0)
    ax.set_ylim(-0.005, prev * 1.05)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set(title='Curva de decisión (zona sombreada: umbrales razonables)', xlabel='Umbral de riesgo',
           ylabel='Beneficio neto')
    ax.legend(fontsize=8)
    plt.tight_layout()
    archivos.append(carpeta / 'curva_decision.png')
    fig.savefig(archivos[-1], bbox_inches='tight')
    plt.close(fig)

    # 5. Captura por decil del mejor modelo frente al azar
    mejor = res[res['Tipo'] != 'Ensamble'].index[0]
    cap = captura_por_decil(y, tabla[mejor])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(np.arange(1, 11), cap, color=AZUL, width=0.7)
    ax.axhline(10, color=NARANJA, linestyle='--', linewidth=1.5, label='Azar (10% por decil)')
    for i, v in enumerate(cap, 1):
        ax.text(i, v + 0.4, f'{v:.1f}%', ha='center', fontsize=8, color=TEXTO_2)
    ax.set_xticks(np.arange(1, 11))
    ax.set(title=f'{mejor}: % de las complicaciones en cada decil de riesgo', xlabel='Decil (1 = mayor riesgo)',
           ylabel='% del total de complicaciones')
    ax.legend()
    plt.tight_layout()
    archivos.append(carpeta / 'captura_deciles.png')
    fig.savefig(archivos[-1], bbox_inches='tight')
    plt.close(fig)
    return archivos


def resumen_texto(tabla, res, baseline):
    """results/resumen_generado.txt: tabla, mejor modelo y análisis (automático)."""
    y = tabla['y'].to_numpy()
    ind = res[res['Tipo'] != 'Ensamble']
    mejor, mejor_ens = ind.index[0], (res[res['Tipo'] == 'Ensamble'].index[0] if (res['Tipo'] == 'Ensamble').any() else None)
    f = lambda m, c='ROC-AUC': res.loc[m, c]
    lineas = [
        "COMPARACIÓN DE MODELOS - Complicaciones graves en pacientes hipertensos del SIS (Perú)",
        f"Generado por src/evaluation.py el {date.today():%d-%m-%Y}",
        "=" * 100, "",
        "DATOS DE EVALUACIÓN",
        f"  Test: {len(y):,} pacientes (20% estratificado, random_state=42), {y.mean():.2%} con complicación en 2024.",
        "  Todos los modelos se evalúan sobre exactamente los mismos pacientes. Los hiperparámetros se eligieron",
        "  solo con train (validación interna, OOB o validación cruzada); test se usó una sola vez.", "",
        "TABLA COMPARATIVA (ordenada por ROC-AUC; IC 95% por bootstrap)",
    ]
    cols = ['Tipo', 'ROC-AUC', 'ROC-AUC IC 95% inf', 'ROC-AUC IC 95% sup', 'PR-AUC', 'Brier', 'Captura top 10% (%)']
    t = res[cols].copy()
    t.columns = ['Tipo', 'ROC-AUC', 'IC inf', 'IC sup', 'PR-AUC', 'Brier', 'Top 10% (%)']
    lineas += ['  ' + l for l in t.round(4).to_string().split('\n')]
    lineas += ["", f"  Baseline: {baseline}. Δ frente al baseline y frente al Random Forest: ver results/metrics.csv.", ""]

    lineas += ["MEJOR MODELO", f"  Individual: {mejor} — ROC-AUC {f(mejor):.4f} [{f(mejor, 'ROC-AUC IC 95% inf'):.4f}; "
               f"{f(mejor, 'ROC-AUC IC 95% sup'):.4f}], PR-AUC {f(mejor, 'PR-AUC'):.4f}, Brier {f(mejor, 'Brier'):.4f}."]
    if 'Δ ROC-AUC vs RF' in res and mejor != 'Random Forest':
        lineas.append(f"  Frente al Random Forest: {f(mejor, 'Δ ROC-AUC vs RF'):+.4f} [{f(mejor, 'IC 95% inf vs RF'):+.4f}; "
                      f"{f(mejor, 'IC 95% sup vs RF'):+.4f}] (significativa: {f(mejor, 'Significativa vs RF')}).")
    lineas.append(f"  Frente al baseline: {f(mejor, 'Δ ROC-AUC vs baseline'):+.4f} (significativa: {f(mejor, 'Significativa vs baseline')}).")
    if mejor_ens:
        lineas.append(f"  Mejor ensamble: {mejor_ens} — ROC-AUC {f(mejor_ens):.4f}, PR-AUC {f(mejor_ens, 'PR-AUC'):.4f}, "
                      f"captura del top 10% {f(mejor_ens, 'Captura top 10% (%)'):.1f}%.")
    rango = ind['ROC-AUC']
    lineas += [f"  Los {len(ind)} modelos individuales quedan entre {rango.min():.3f} y {rango.max():.3f} de ROC-AUC.", ""]

    lineas += [
        "ANÁLISIS: ¿POR QUÉ FUNCIONÓ MEJOR?",
        "  1. Los ensambles de árboles (Random Forest, gradient boosting) superan a KNN, al árbol único y a la",
        "     regresión logística porque captan relaciones no lineales e interacciones entre edad, comorbilidades y",
        "     uso del sistema, sin que haya que especificarlas. El boosting gana con árboles pequeños (15 hojas) y",
        "     el MLP no mejora con más capas: la señal está en interacciones de pocas variables.",
        "  2. La GRU (Parte 5) aprovecha la evolución mensual: el último trimestre de 2023 concentra más de la mitad de su",
        "     importancia. Ordena a los pacientes de forma distinta a los modelos tabulares (correlación ~0.88 con el",
        "     RF), por eso combinarla con ellos es lo que más mejora el ensamble.",
        "  3. KNN y el árbol único rinden menos: KNN sufre con 34 variables de escalas y naturalezas mezcladas, y un",
        "     solo árbol es inestable; ambos necesitan mucha regularización (k grande, hojas grandes).",
        "  4. Los pesos de clase (class_weight='balanced') no mejoran el ROC-AUC y descalibran las probabilidades",
        "     (Brier ~0.22 frente a ~0.05 sin pesos). Por eso los modelos finales de las partes 3-8 van sin pesos.", "",
        "LIMITACIONES",
        "  - Techo de ~0.71 de ROC-AUC con todos los algoritmos: el límite lo pone la información disponible",
        "    (datos administrativos, sin laboratorio, tabaquismo ni adherencia), no el modelo.",
        "  - Sesgo de detección: parte de la señal es uso del sistema (prop_capita, nivel_eess_max, n_atenciones).",
        "    En la Parte 6, con el objetivo 'grave' el AUC sube a 0.731 y prop_capita pierde dos tercios de su peso:",
        "    el sesgo existe, pero la señal clínica sobrevive.",
        "  - ACV (0.68) e infarto (0.67) son los eventos menos predecibles.",
        "  - El fallecimiento solo se conoce si ocurre en una atención del SIS.",
        "  - Una sola validación temporal (2022->2023 frente a 2023->2024): el RF pierde 0.011 de AUC y sobrestima",
        "    el riesgo en ~10%, corregible recalibrando el intercepto.", "",
        "INSIGHTS",
        f"  - Utilidad práctica: con {mejor}, el 10% de pacientes de mayor riesgo concentra el "
        f"{f(mejor, 'Captura top 10% (%)'):.1f}% de las",
        f"    complicaciones ({f(mejor, 'Captura top 10% (%)') / 10:.1f} veces el azar). Entre umbrales de 3% y 15% "
        "el modelo da más beneficio",
        "    neto que seguir a todos o a nadie (Parte 6).",
        "  - Las variables administrativas aportan tanto como las comorbilidades: edad y sexo dan 0.594; un modelo",
        "    clínico mínimo, 0.649; el Random Forest con las 34 variables, 0.704 (Parte 6).",
        "  - La continuidad del seguimiento aparece como protectora frente a los eventos graves.",
        "  - El siguiente avance debe venir de variables nuevas (recencia, variabilidad de la presión, síntomas",
        "    previos), no de algoritmos más complejos.",
    ]
    return '\n'.join(lineas) + '\n'


def generar_resultados(raiz=None, n_boot=500):
    raiz = buscar_raiz(raiz)
    ruta_proc, ruta_res = raiz / 'data' / 'processed', raiz / 'results'
    ruta_res.mkdir(exist_ok=True)
    print("Leyendo predicciones de", ruta_proc)
    tabla, info = recolectar_predicciones(ruta_proc)
    print(f"  {len(tabla):,} pacientes de test | modelos: {', '.join(c for c in tabla.columns if c != 'y')}")
    res, baseline = tabla_comparativa(tabla, info, n_boot=n_boot)
    res.round(5).to_csv(ruta_res / 'metrics.csv', encoding='utf-8-sig')
    archivos = graficos(tabla, res, ruta_res / 'plots')
    (ruta_res / 'resumen_generado.txt').write_text(resumen_texto(tabla, res, baseline), encoding='utf-8')
    print("Escrito:")
    for a in [ruta_res / 'metrics.csv', ruta_res / 'resumen_generado.txt', *archivos]:
        print("  -", a.relative_to(raiz))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Genera results/metrics.csv, results/plots/ y results/resumen_generado.txt")
    ap.add_argument('--raiz', default=None, help="carpeta raíz del repositorio (por defecto se busca desde aquí)")
    ap.add_argument('--n-boot', type=int, default=500, help="remuestreos bootstrap para los IC (por defecto 500)")
    a = ap.parse_args()
    generar_resultados(Path(a.raiz) if a.raiz else Path(__file__).resolve().parent, n_boot=a.n_boot)
