"""
Modelos del proyecto: predicción de complicaciones graves en pacientes hipertensos del SIS (Perú).

Reúne en un solo módulo todos los modelos implementados, con los hiperparámetros elegidos en los cuadernos:

| Modelo                       | Dónde se ajusta (notebooks/02_modelos.ipynb) | Función                         |
|------------------------------|----------------------------------------------|---------------------------------|
| KNN                          | Parte 2                                      | crear_knn()                     |
| Árbol de decisión            | Parte 2                                      | crear_arbol()                   |
| Regresión logística          | Parte 2 (baseline)                           | crear_regresion_logistica()     |
| Random Forest                | Parte 3                                      | crear_random_forest()           |
| GRU / LSTM y MLP de control  | Parte 5                                      | crear_gru(), crear_mlp_control()|
| Gradient boosting            | Parte 7                                      | crear_gradient_boosting()       |
| Ensamble (promedio)          | Parte 7                                      | promediar()                     |
| MLP (perceptrón multicapa)   | Parte 8                                      | construir_mlp(), entrenar_mlp() |

Los modelos de scikit-learn se devuelven como Pipeline (imputación -> estandarización -> modelo), de modo que
`fit(X_train)` aprende medianas y escalas solo con train. Los modelos de PyTorch (MLP, GRU) importan torch solo
cuando se usan, para que el resto del módulo funcione sin PyTorch instalado.

Uso típico:
    from models import cargar_dataset, particion, crear_random_forest
    df = cargar_dataset()
    X_train, X_test, y_train, y_test = particion(df)
    rf = crear_random_forest().fit(X_train, y_train)
    prob = rf.predict_proba(X_test)[:, 1]
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

SEED = 42
COLS_ELIMINAR = ['dias_hosp', 'pad_media']   # redundantes según la matriz de correlación (Parte 2)

# Hiperparámetros finales. Cada parte de notebooks/02_modelos.ipynb guarda los que eligió (con validación interna, nunca mirando test) en
# data/processed/, y este módulo los lee de ahí:
#   hiperparametros_02.json          -> KNN, árbol y regresión logística (Parte 2)
#   mejores_hiperparametros_rf.csv   -> Random Forest (Parte 3)
#   mejores_hiperparametros_gb.csv   -> Gradient boosting (Parte 7)
# Los valores de abajo son los elegidos en la ejecución del 17/20-nov-2026 (RF y GB) y se usan si falta el archivo.
HP_RF = {'n_estimators': 500, 'min_samples_leaf': 50, 'max_samples': 0.3, 'max_features': 0.5,
         'max_depth': 16, 'class_weight': None}
HP_GB = {'learning_rate': 0.05, 'max_leaf_nodes': 15, 'min_samples_leaf': 100,
         'l2_regularization': 0.0, 'max_features': 0.7}


def _leer_hiperparametros():
    """Lee los hiperparámetros guardados por los cuadernos (si existen)."""
    import json
    hp = {}
    try:
        carpeta = buscar_dataset().parent
    except FileNotFoundError:
        return hp
    f02 = carpeta / 'hiperparametros_02.json'
    if f02.exists():
        hp.update(json.loads(f02.read_text(encoding='utf-8')))
    for clave, archivo in [('rf', 'mejores_hiperparametros_rf.csv'), ('gb', 'mejores_hiperparametros_gb.csv')]:
        if (carpeta / archivo).exists():
            serie = pd.read_csv(carpeta / archivo, index_col=0).iloc[:, 0]
            hp[clave] = {k: _convertir(k, v) for k, v in serie.items()}
    return hp


_ENTEROS = {'n_estimators', 'min_samples_leaf', 'min_samples_split', 'max_depth', 'max_leaf_nodes', 'n_neighbors'}


def _convertir(clave, v):
    """Convierte un valor leído de CSV ('None', '', '20.0', '0.5'...) al tipo que espera scikit-learn."""
    if isinstance(v, str):
        if v.strip() in ('None', 'nan', ''):
            return None
        try:
            v = float(v)
        except ValueError:
            return v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if clave in _ENTEROS:
        return int(v)
    return float(v) if isinstance(v, (int, float, np.number)) else v


def _hp(modelo):
    """Hiperparámetros de un modelo de 02 ('knn', 'arbol', 'rl'); error claro si 02 aún no los guardó."""
    hp = _leer_hiperparametros()
    if modelo not in hp:
        raise ValueError(f"No se encontraron los hiperparámetros de '{modelo}': ejecuta la Parte 2 de "
                         "notebooks/02_modelos.ipynb (guarda data/processed/hiperparametros_02.json) o pásalos como argumentos.")
    return hp[modelo]


# --------------------------------------------------------------------------------------------- datos
def buscar_dataset(nombre='dataset_consolidado.csv', inicio=None):
    """Busca data/processed/<nombre> desde la carpeta actual hacia arriba (sirve desde notebooks/ o src/)."""
    inicio = Path(inicio) if inicio else Path.cwd()
    for p in [inicio, *inicio.parents]:
        for d in (p, p / 'data' / 'processed'):
            if (d / nombre).exists():
                return d / nombre
    raise FileNotFoundError(f"No se encontró {nombre}. Ejecuta primero src/preprocessing.ipynb.")


def cargar_dataset(ruta=None):
    """Lee dataset_consolidado.csv (una fila por paciente, objetivo en la columna 'y')."""
    return pd.read_csv(ruta or buscar_dataset(), index_col='paciente')


def particion(df, test_size=0.2, seed=SEED):
    """Partición 80/20 estratificada usada en TODOS los cuadernos: los mismos pacientes de test para cada modelo."""
    y = df['y']
    X = df.drop(columns=['y'] + [c for c in COLS_ELIMINAR if c in df.columns])
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def particion_validacion(X_train, y_train, test_size=0.2, seed=SEED):
    """Validación interna dentro de train (elección de hiperparámetros, early stopping, umbral)."""
    return train_test_split(X_train.index, test_size=test_size, random_state=seed, stratify=y_train)


# --------------------------------------------------------------------------------------- scikit-learn
def _pipeline(modelo, escalar=True):
    pasos = [('imputer', SimpleImputer(strategy='median'))]
    if escalar:
        pasos.append(('scaler', StandardScaler()))
    return Pipeline(pasos + [('model', modelo)])


def crear_knn(n_neighbors=None):
    """KNN con imputación y estandarización (las distancias exigen variables en la misma escala)."""
    n_neighbors = n_neighbors or _hp('knn')['n_neighbors']
    return _pipeline(KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1))


def crear_arbol(max_depth=None, min_samples_leaf=None, min_samples_split=20, class_weight='balanced', seed=SEED):
    """Árbol de decisión (no necesita estandarización)."""
    if max_depth is None or min_samples_leaf is None:
        hp = _hp('arbol')
        max_depth = max_depth or hp['max_depth']
        min_samples_leaf = min_samples_leaf or hp['min_samples_leaf']
    return _pipeline(DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                            min_samples_split=min_samples_split, class_weight=class_weight,
                                            random_state=seed), escalar=False)


def crear_regresion_logistica(C=None, solver=None, class_weight='balanced', seed=SEED):
    """Regresión logística. class_weight='balanced' (Parte 2) o None (versión calibrada usada en las partes 3-8)."""
    if C is None or solver is None:
        hp = _hp('rl')
        C = C if C is not None else hp['C']
        solver = solver or hp['solver']
    return _pipeline(LogisticRegression(C=C, solver=solver, class_weight=class_weight, max_iter=1000,
                                        random_state=seed))


def crear_random_forest(seed=SEED, n_jobs=-1, **hp):
    """Random Forest con los hiperparámetros de la Parte 3 (sin pesos de clase: probabilidades calibradas)."""
    params = {**HP_RF, **_leer_hiperparametros().get('rf', {}), **hp}
    params = {k: v for k, v in params.items() if k in HP_RF}
    return _pipeline(RandomForestClassifier(random_state=seed, n_jobs=n_jobs, oob_score=False, **params),
                     escalar=False)


def crear_gradient_boosting(seed=SEED, **hp):
    """Gradient boosting por histogramas (Parte 7). Maneja los faltantes de forma nativa: no se imputa."""
    params = {**HP_GB, **_leer_hiperparametros().get('gb', {}), **hp}
    params = {k: v for k, v in params.items() if k in HP_GB}
    import inspect
    if 'max_features' not in inspect.signature(HistGradientBoostingClassifier).parameters:   # scikit-learn < 1.4
        params.pop('max_features', None)
    return HistGradientBoostingClassifier(max_iter=2000, early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=30, random_state=seed, **params)


def modelos_tabulares():
    """Diccionario nombre -> modelo sin entrenar, para recorrer todos los modelos de scikit-learn."""
    return {
        'KNN': crear_knn(),
        'Árbol de decisión': crear_arbol(),
        'Regresión logística': crear_regresion_logistica(class_weight=None),
        'Random Forest': crear_random_forest(),
        'Gradient boosting': crear_gradient_boosting(),
    }


# -------------------------------------------------------------------------------------------- PyTorch
def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:
        raise ImportError("Este modelo necesita PyTorch: pip install torch") from e
    return torch, nn


def construir_mlp(n_in, p):
    """MLP de la Parte 8: [Linear -> (BatchNorm) -> activación -> (Dropout)] x n_capas -> Linear(1) (logit).

    p: dict con n_capas, ancho, forma ('constante' | 'piramide'), activacion ('relu' | 'silu' | 'gelu'),
       dropout, batchnorm.
    """
    torch, nn = _torch()
    anchos = [max(8, p['ancho'] // (2 ** i)) if p['forma'] == 'piramide' else p['ancho'] for i in range(p['n_capas'])]
    act = {'relu': nn.ReLU, 'silu': nn.SiLU, 'gelu': nn.GELU}[p['activacion']]
    capas, d = [], n_in
    for h in anchos:
        capas.append(nn.Linear(d, h))
        if p['batchnorm']:
            capas.append(nn.BatchNorm1d(h))
        capas.append(act())
        if p['dropout'] > 0:
            capas.append(nn.Dropout(p['dropout']))
        d = h
    capas.append(nn.Linear(d, 1))
    return nn.Sequential(*capas)


def predecir_mlp(modelo, A, lote=65536):
    """Probabilidad de complicación para cada fila de A (numpy float32 ya preprocesado)."""
    torch, _ = _torch()
    dispositivo = next(modelo.parameters()).device
    modelo.eval()
    salida = []
    with torch.no_grad():
        for i in range(0, len(A), lote):
            x = torch.as_tensor(A[i:i + lote], device=dispositivo)
            salida.append(torch.sigmoid(modelo(x).squeeze(1)).cpu().numpy())
    return np.concatenate(salida)


def entrenar_mlp(p, A_tr, y_tr, A_val, y_val, seed=SEED, max_epochs=40, paciencia=6, alpha=None):
    """Entrena un MLP con AdamW y early stopping por ROC-AUC de validación. Devuelve (modelo, mejor AUC, época).

    alpha: peso de la clase positiva = (negativos/positivos) ** alpha. 0 = sin pesos (probabilidades calibradas).
    """
    from sklearn.metrics import roc_auc_score
    torch, nn = _torch()
    dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)
    alpha = p.get('alpha', 0.0) if alpha is None else alpha
    ratio = (1 - y_tr.mean()) / y_tr.mean()
    modelo = construir_mlp(A_tr.shape[1], p).to(dispositivo)
    perdida = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(ratio ** alpha), device=dispositivo))
    opt = torch.optim.AdamW(modelo.parameters(), lr=p['lr'], weight_decay=p['wd'])
    Xt = torch.as_tensor(A_tr, device=dispositivo)
    yt = torch.as_tensor(np.asarray(y_tr, dtype='float32'), device=dispositivo)
    mejor_auc, mejor_ep, mejor_estado, sin_mejora = -1.0, 0, None, 0
    for ep in range(max_epochs):
        modelo.train()
        perm = torch.randperm(len(Xt), device=dispositivo)
        for i in range(0, len(Xt), p['batch']):
            idx = perm[i:i + p['batch']]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            perdida(modelo(Xt[idx]).squeeze(1), yt[idx]).backward()
            opt.step()
        auc = roc_auc_score(y_val, predecir_mlp(modelo, A_val))
        if auc > mejor_auc + 1e-5:
            mejor_auc, mejor_ep, sin_mejora = auc, ep + 1, 0
            mejor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
        if sin_mejora >= paciencia:
            break
    modelo.load_state_dict(mejor_estado)
    return modelo, mejor_auc, mejor_ep


def _clases_recurrentes():
    torch, nn = _torch()

    class RedRecurrente(nn.Module):
        """GRU/LSTM de la Parte 5: 24 meses x 21 variables -> capa recurrente -> último estado + estáticas -> logit."""
        def __init__(self, n_var, n_est, hidden=64, capas=1, dropout=0.2, tipo='gru'):
            super().__init__()
            self.entrada = nn.Sequential(nn.Linear(n_var, 32), nn.ReLU())
            celda = nn.GRU if tipo == 'gru' else nn.LSTM
            self.rnn = celda(32, hidden, num_layers=capas, batch_first=True, dropout=dropout if capas > 1 else 0.0)
            self.salida = nn.Sequential(nn.Linear(hidden + n_est, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

        def forward(self, x, e):
            h, _ = self.rnn(self.entrada(x))
            return self.salida(torch.cat([h[:, -1, :], e], dim=1)).squeeze(1)

    class MLPControl(nn.Module):
        """Control de la Parte 5: misma información sin orden temporal (media y máximo de los 24 meses)."""
        def __init__(self, n_var, n_est, dropout=0.2):
            super().__init__()
            self.red = nn.Sequential(nn.Linear(2 * n_var + n_est, 64), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

        def forward(self, x, e):
            return self.red(torch.cat([x.mean(dim=1), x.amax(dim=1), e], dim=1)).squeeze(1)

    return RedRecurrente, MLPControl


def crear_gru(n_var, n_est, **kw):
    """Red recurrente (tipo='gru' o 'lstm') de la Parte 5."""
    RedRecurrente, _ = _clases_recurrentes()
    return RedRecurrente(n_var, n_est, **kw)


def crear_mlp_control(n_var, n_est, **kw):
    """MLP de control de la Parte 5."""
    _, MLPControl = _clases_recurrentes()
    return MLPControl(n_var, n_est, **kw)


# ------------------------------------------------------------------------------------------- ensamble
def promediar(*probabilidades):
    """Ensamble por promedio simple de probabilidades (Parte 7): sin parámetros, conserva la calibración."""
    return np.mean(np.column_stack(probabilidades), axis=1)
