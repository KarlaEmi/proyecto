"""Construcción de cohortes del proyecto HTA-SIS con ventanas de tiempo parametrizables.

Reproduce la lógica de ``src/preprocessing.ipynb`` (misma limpieza, mismos grupos CIE-10, mismas 36 variables),
pero permite elegir los años de observación y el año del resultado. Así se pueden construir:

* la cohorte principal: observación 2022-2023 -> resultado 2024 (la de ``dataset_consolidado.csv``);
* las cohortes de validación temporal: 2022 -> 2023 (desarrollo) y 2023 -> 2024 (validación).

Además de ``y`` (el compuesto original), cada cohorte trae objetivos alternativos para análisis de sensibilidad:

=================  ===============================================================================
``y_grave``        evento registrado en una atención de emergencia u hospitalización, o fallecimiento
``y_sin_n17``      compuesto sin la lesión renal aguda (N17)
``y_cardiovascular``  ACV/AIT, infarto o insuficiencia cardíaca
``y_erc``          enfermedad renal (N17-N19, I12, I13.1, I13.2)
``y_fallecido``    fallecimiento registrado
``y_acv`` ...      ACV/AIT, infarto e insuficiencia cardíaca por separado
=================  ===============================================================================

Diseño para ahorrar memoria: los diagnósticos y los consumos se resumen en tablas **paciente-año**, y cada
cohorte se arma combinando los años que necesita. Solo FUAS se conserva fila por fila.

Uso típico::

    from preprocessing import cargar_tablas, construir_cohorte
    t = cargar_tablas(ruta_raw, ruta_altitudes)
    df, flujo = construir_cohorte(t, anios_obs=[2022, 2023], anio_res=2024)
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals

# --------------------------------------------------------------------------------------------------------------
# Definiciones (idénticas a src/preprocessing.ipynb)
# --------------------------------------------------------------------------------------------------------------
GRUPOS = {
    'hta':           ('I10', 'I11', 'I12', 'I13', 'I14', 'I15'),
    'acv':           ('I60', 'I61', 'I62', 'I63', 'I64', 'G45'),
    'iam':           ('I21', 'I22', 'I23'),
    'ic':            ('I50', 'I110', 'I130', 'I132'),
    'erc':           ('N17', 'N18', 'N19', 'I12', 'I131', 'I132'),
    'diabetes':      ('E10', 'E11', 'E12', 'E13', 'E14'),
    'dislipidemia':  ('E78',),
    'obesidad':      ('E66',),
    'fib_auricular': ('I48',),
    'cardiopatia':   ('I20', 'I24', 'I25'),
    'hta_cardiaca':  ('I119',),
    # Solo para el análisis de sensibilidad: enfermedad renal sin la lesión renal aguda (N17)
    'erc_sin_n17':   ('N18', 'N19', 'I12', 'I131', 'I132'),
}
EVENTOS = ['acv', 'iam', 'ic', 'erc']
COMORBILIDADES = ['diabetes', 'dislipidemia', 'obesidad', 'fib_auricular', 'cardiopatia']
OBJETIVOS = ['y', 'y_grave', 'y_sin_n17', 'y_cardiovascular', 'y_erc', 'y_fallecido', 'y_acv', 'y_iam', 'y_ic']

COLS_FUAS = ['CODIGO_ANONIMIZADO', 'SEXO', 'ID_REGISTRO_REL', 'FECHA_ATENCION', 'EDAD', 'UBIGEO',
             'DEPARTAMENTO', 'NIVEL_EESS', 'SERVICIO_PRESTACIONAL', 'DIAS_HOSP',
             'PRES_ART_SISTOLICA', 'PRES_ART_DIASTOLICA', 'ES_CAPITA', 'DESTINO_ASEGURADO']
TIPOS_FUAS = {'CODIGO_ANONIMIZADO': str, 'SEXO': str, 'ID_REGISTRO_REL': 'int64', 'FECHA_ATENCION': 'int32',
              'EDAD': 'float32', 'UBIGEO': str, 'DEPARTAMENTO': str, 'NIVEL_EESS': 'int8', 'SERVICIO_PRESTACIONAL': str,
              'DIAS_HOSP': 'float32', 'PRES_ART_SISTOLICA': 'float32', 'PRES_ART_DIASTOLICA': 'float32',
              'ES_CAPITA': str, 'DESTINO_ASEGURADO': str}
CAT_FUAS = ['CODIGO_ANONIMIZADO', 'SEXO', 'UBIGEO', 'DEPARTAMENTO', 'SERVICIO_PRESTACIONAL', 'ES_CAPITA', 'DESTINO_ASEGURADO']

CONSUMOS = {  # archivo, columna del código, nombre de la variable, ¿contar códigos distintos?
    'medicamentos':   ('3. HTA_Medicamentos',   'COD_MEDICAMENTO',    True),
    'procedimientos': ('4. HTA_Procedimientos', 'COD_PROCEDIMIIENTO', False),
    'insumos':        ('5. HTA_Insumos',        'COD_INSUMO',         False),
}


def _log(msg, t0=None):
    extra = f" ({(time.time() - t0) / 60:.1f} min)" if t0 is not None else ''
    print(msg + extra, flush=True)


# --------------------------------------------------------------------------------------------------------------
# Lectura
# --------------------------------------------------------------------------------------------------------------
def buscar_datos(ruta_manual=None, archivo_prueba='1. HTA_FUAS') -> Path:
    """Carpeta que contiene '1. HTA_FUAS' (.csv o .zip): RUTA_MANUAL, la carpeta actual, data/raw/ o las superiores."""
    candidatas = [Path(ruta_manual)] if ruta_manual else \
        [c for p in [Path.cwd(), *Path.cwd().parents] for c in (p, p / 'data' / 'raw')]
    for carpeta in candidatas:
        for ext in ('.csv', '.zip'):
            archivo = carpeta / f'{archivo_prueba}{ext}'
            if archivo.exists():
                if ext == '.zip' and archivo.stat().st_size < 10_000:
                    raise FileNotFoundError(f"{archivo} es un puntero de Git LFS, no el archivo real. "
                                            "Ejecuta `git lfs pull` o usa los CSV descomprimidos.")
                return carpeta
    raise FileNotFoundError(f"No se encontró '{archivo_prueba}.csv' ni '.zip' desde {Path.cwd()}. Usa RUTA_MANUAL.")


def _ruta(ruta_raw, nombre):
    csv = Path(ruta_raw) / f'{nombre}.csv'
    return csv if csv.exists() else Path(ruta_raw) / f'{nombre}.zip'


def leer_csv(archivo, usecols, dtype, categorias=(), filtro=None, chunk=2_000_000):
    """Lee un CSV por bloques, filtra cada bloque y guarda los textos como 'category' (igual que el cuaderno)."""
    partes = []
    for bloque in pd.read_csv(archivo, usecols=usecols, dtype=dtype, chunksize=chunk):
        if filtro is not None:
            bloque = filtro(bloque)
        for c in categorias:
            bloque[c] = bloque[c].astype('category')
        partes.append(bloque)
    for c in categorias:
        todas = union_categoricals([p[c] for p in partes]).categories
        for p in partes:
            p[c] = p[c].cat.set_categories(todas)
    tabla = pd.concat(partes, ignore_index=True)
    del partes
    gc.collect()
    return tabla


# --------------------------------------------------------------------------------------------------------------
# Carga y resumen paciente-año
# --------------------------------------------------------------------------------------------------------------
def cargar_fuas(ruta_raw, frac=None, seed=42, chunk=2_000_000):
    """FUAS limpio, una fila por atención, con las columnas de apoyo del cuaderno (emergencia, pa_alta, pp...)."""
    archivo = _ruta(ruta_raw, '1. HTA_FUAS')
    filtro = None
    if frac is not None:
        unicos = pd.concat(b['CODIGO_ANONIMIZADO'].drop_duplicates()
                           for b in pd.read_csv(archivo, usecols=['CODIGO_ANONIMIZADO'], chunksize=chunk))
        muestra = set(unicos.drop_duplicates().sample(frac=frac, random_state=seed))
        filtro = lambda b: b[b['CODIGO_ANONIMIZADO'].isin(muestra)]
    fuas = leer_csv(archivo, COLS_FUAS, TIPOS_FUAS, CAT_FUAS, filtro, chunk)
    fuas['UBIGEO'] = fuas['UBIGEO'].cat.rename_categories(lambda u: str(u).zfill(6))
    codigos_paciente = fuas['CODIGO_ANONIMIZADO'].cat.categories
    fuas['paciente'] = fuas['CODIGO_ANONIMIZADO'].cat.codes.astype('int32')
    fuas = fuas.drop(columns='CODIGO_ANONIMIZADO')

    # Limpieza fila por fila (rangos fisiológicos fijos; ver el cuaderno)
    fuas.loc[(fuas['EDAD'] < 14) | (fuas['EDAD'] > 110), 'EDAD'] = np.nan
    pas = fuas['PRES_ART_SISTOLICA'].to_numpy(dtype='float64', na_value=np.nan)
    pad = fuas['PRES_ART_DIASTOLICA'].to_numpy(dtype='float64', na_value=np.nan)
    medida = ~np.isnan(pas) & ~np.isnan(pad)
    valida = medida & (pas >= 60) & (pas <= 260) & (pad >= 30) & (pad <= 150) & (pas > pad)
    invalida = medida & ~valida
    assert invalida.sum() / max(medida.sum(), 1) < 0.05, "Se están anulando demasiadas presiones: revisar los tipos de dato"
    fuas.loc[invalida, ['PRES_ART_SISTOLICA', 'PRES_ART_DIASTOLICA']] = np.nan
    del pas, pad, medida, valida, invalida
    fuas['DIAS_HOSP'] = fuas['DIAS_HOSP'].fillna(0).clip(upper=365)

    pas, pad = fuas['PRES_ART_SISTOLICA'], fuas['PRES_ART_DIASTOLICA']
    fuas['masculino'] = (fuas['SEXO'] == 'MASCULINO').astype('int8')
    fuas['hospitalizado'] = (fuas['DIAS_HOSP'] > 0).astype('int8')
    fuas['emergencia'] = fuas['SERVICIO_PRESTACIONAL'].str.upper().str.contains('EMERGENCIA').astype('int8')
    fuas['fallecido'] = fuas['DESTINO_ASEGURADO'].str.upper().str.contains('FALLEC').astype('int8')
    fuas['pa_alta'] = ((pas >= 140) | (pad >= 90)).astype('float32').where(pas.notna())
    fuas['pp'] = pas - pad
    fuas['capita'] = (fuas['ES_CAPITA'] == 'S').astype('int8')
    fuas['anio'] = (fuas['FECHA_ATENCION'] // 10000).astype('int16')
    fuas = fuas.drop(columns=['SEXO', 'SERVICIO_PRESTACIONAL', 'ES_CAPITA', 'DESTINO_ASEGURADO'])
    fuas = fuas.sort_values('FECHA_ATENCION', kind='stable')
    return fuas, codigos_paciente


def resumir_diagnosticos(ruta_raw, fuas, chunk=2_000_000):
    """Diagnósticos DEFINITIVO/REPETIDO resumidos por paciente-año.

    Devuelve ``dx_anual`` (un indicador por grupo CIE-10 y, para cada evento, si ocurrió en una atención de
    emergencia u hospitalización) y ``codigos_anual`` (códigos distintos por paciente-año, para n_diagnosticos).
    """
    ids = set(fuas['ID_REGISTRO_REL'])
    dx = leer_csv(_ruta(ruta_raw, '2. HTA_Diagnosticos'),
                  ['FECHA_CORTE', 'ULTIMO_MES_CONSUMOS', 'ID_REGISTRO_REL', 'CODDIA', 'TIPO_DIAGNOSTICO'],
                  {'FECHA_CORTE': 'int32', 'ULTIMO_MES_CONSUMOS': 'int32', 'ID_REGISTRO_REL': 'int64',
                   'CODDIA': str, 'TIPO_DIAGNOSTICO': str},
                  ['CODDIA', 'TIPO_DIAGNOSTICO'], lambda b: b[b['ID_REGISTRO_REL'].isin(ids)], chunk)
    dx = dx.drop_duplicates()
    tipo = dx['TIPO_DIAGNOSTICO'].str.strip()
    dx = dx.loc[tipo.isin(['DEFINITIVO', 'REPETIDO']).values, ['ID_REGISTRO_REL', 'CODDIA']]
    del tipo
    limpios = dx['CODDIA'].cat.categories.str.strip().str.upper()
    if limpios.is_unique:
        dx['CODDIA'] = dx['CODDIA'].cat.rename_categories(limpios)
    else:
        dx['CODDIA'] = dx['CODDIA'].astype(str).str.strip().str.upper().astype('category')
    dx = dx.merge(fuas[['ID_REGISTRO_REL', 'paciente', 'anio', 'emergencia', 'hospitalizado']], on='ID_REGISTRO_REL')
    dx = dx.drop(columns='ID_REGISTRO_REL')
    gc.collect()

    # Indicadores por grupo: se calculan sobre las categorías (pocos miles) y se mapean a las filas
    cats = dx['CODDIA'].cat.categories
    codigos = dx['CODDIA'].cat.codes.to_numpy()
    for nombre, prefijos in GRUPOS.items():
        en_grupo = np.asarray(cats.str.startswith(prefijos), dtype=bool)
        dx[nombre] = en_grupo[codigos] & (codigos >= 0)
    grave = ((dx['emergencia'] == 1) | (dx['hospitalizado'] == 1)).to_numpy()
    for e in EVENTOS:
        dx[f'{e}_grave'] = dx[e].to_numpy() & grave
    dx = dx.drop(columns=['emergencia', 'hospitalizado'])

    columnas = list(GRUPOS) + [f'{e}_grave' for e in EVENTOS]
    dx_anual = dx.groupby(['paciente', 'anio'], observed=True)[columnas].max()
    codigos_anual = dx[['paciente', 'anio', 'CODDIA']].drop_duplicates()
    del dx
    gc.collect()
    return dx_anual, codigos_anual


def resumir_consumos(ruta_raw, fuas, pacientes, anios, chunk=2_000_000):
    """Medicamentos, procedimientos e insumos de las atenciones de ``pacientes`` en ``anios``, por paciente-año.

    Igual que el cuaderno: duplicados exactos eliminados y VALOR_NETO recortado al percentil 99.9.
    Devuelve {nombre: (resumen_anual[n, gasto], codigos_anual o None)}.
    """
    atenciones = fuas.loc[fuas['paciente'].isin(pacientes) & fuas['anio'].isin(anios), ['ID_REGISTRO_REL', 'paciente', 'anio']]
    ids = set(atenciones['ID_REGISTRO_REL'])
    salida = {}
    for nombre, (archivo, col, distintos) in CONSUMOS.items():
        t0 = time.time()
        t = leer_csv(_ruta(ruta_raw, archivo),
                     ['FECHA_CORTE', 'ULTIMO_MES_CONSUMOS', 'ID_REGISTRO_REL', 'CODDIA', col, 'CANTIDAD_ENTREGADA', 'VALOR_NETO'],
                     {'FECHA_CORTE': 'int32', 'ULTIMO_MES_CONSUMOS': 'int32', 'ID_REGISTRO_REL': 'int64',
                      'CODDIA': str, col: str, 'CANTIDAD_ENTREGADA': 'int64', 'VALOR_NETO': 'float64'},
                     ['CODDIA', col], lambda b: b[b['ID_REGISTRO_REL'].isin(ids)], chunk)
        t = t.drop_duplicates()
        t = t[['ID_REGISTRO_REL', col, 'VALOR_NETO']].merge(atenciones, on='ID_REGISTRO_REL')
        t['VALOR_NETO'] = t['VALOR_NETO'].clip(upper=t['VALOR_NETO'].quantile(0.999))
        anual = t.groupby(['paciente', 'anio']).agg(n=('VALOR_NETO', 'size'), gasto=('VALOR_NETO', 'sum'))
        cods = t[['paciente', 'anio', col]].drop_duplicates() if distintos else None
        salida[nombre] = (anual, cods)
        del t
        gc.collect()
        _log(f"  {nombre}: listo", t0)
    return salida


def cargar_tablas(ruta_raw, ruta_altitudes, anios_consumo=(2022, 2023), frac=None, seed=42, chunk=2_000_000):
    """Lee las 5 tablas una sola vez y deja todo listo para construir cualquier cohorte.

    ``anios_consumo``: años de los que se resumen medicamentos/procedimientos/insumos (los de observación).
    """
    t0 = time.time()
    fuas, codigos_paciente = cargar_fuas(ruta_raw, frac, seed, chunk)
    _log(f"FUAS: {len(fuas):,} atenciones de {fuas['paciente'].nunique():,} pacientes", t0)
    dx_anual, codigos_anual = resumir_diagnosticos(ruta_raw, fuas, chunk)
    _log(f"Diagnósticos resumidos: {len(dx_anual):,} filas paciente-año", t0)
    # Candidatos: hipertensos en algún año de observación (las cohortes son subconjuntos de ellos)
    hta = dx_anual['hta'] & dx_anual.index.get_level_values('anio').isin(anios_consumo)
    candidatos = dx_anual.index.get_level_values('paciente')[hta.to_numpy()].unique()
    codigos_anual = codigos_anual[codigos_anual['paciente'].isin(candidatos)]
    consumos = resumir_consumos(ruta_raw, fuas, candidatos, list(anios_consumo), chunk)
    geo = pd.read_csv(ruta_altitudes, sep=';', dtype={'ubigeo': str})
    _log("Tablas listas", t0)
    return {'fuas': fuas, 'codigos_paciente': codigos_paciente, 'dx_anual': dx_anual, 'codigos_anual': codigos_anual,
            'consumos': consumos, 'altitud': geo.set_index('ubigeo')['altitud'], 'anios_consumo': tuple(anios_consumo)}


# --------------------------------------------------------------------------------------------------------------
# Cohorte
# --------------------------------------------------------------------------------------------------------------
def _moda_por_paciente(tabla, columna):
    conteo = (tabla.groupby(['paciente', columna], observed=True)
                   .agg(n=('FECHA_ATENCION', 'size'), ultima=('FECHA_ATENCION', 'max'))
                   .reset_index()
                   .sort_values(['paciente', 'n', 'ultima'], ascending=[True, False, False]))
    return conteo.drop_duplicates('paciente').set_index('paciente')[columna]


def construir_cohorte(t, anios_obs, anio_res, anios_exclusion=None, codigos_como_indice=True):
    """Arma la cohorte y su dataset (36 variables + objetivos) para una ventana de observación y un año de resultado.

    Parámetros
    ----------
    anios_obs : años de los que salen las variables (X) y en los que se exige un diagnóstico de hipertensión.
    anio_res : año del resultado (y).
    anios_exclusion : años en que se busca un evento grave o fallecimiento previo para excluir al paciente.
        Por defecto, los mismos de ``anios_obs`` (así es la cohorte principal).
    """
    anios_obs = sorted(anios_obs)
    anios_exclusion = sorted(anios_exclusion or anios_obs)
    faltan = set(anios_obs) - set(t['anios_consumo'])
    assert not faltan, f"No se resumieron los consumos de {faltan}: agrégalos a anios_consumo en cargar_tablas"
    fin_obs = anios_obs[-1] * 10000 + 1231
    fuas, dx_anual = t['fuas'], t['dx_anual']
    anio_dx = dx_anual.index.get_level_values('anio')
    pac_dx = dx_anual.index.get_level_values('paciente')

    # --- Selección (mismos pasos y orden que el cuaderno) ---
    n_total = fuas['paciente'].nunique()
    con_hta = pd.Index(pac_dx[(dx_anual['hta'] & anio_dx.isin(anios_obs)).to_numpy()].unique())
    previo = dx_anual[EVENTOS].any(axis=1) & anio_dx.isin(anios_exclusion)
    con_previo = pac_dx[previo.to_numpy()].unique()
    cohorte = con_hta[~con_hta.isin(con_previo)]
    n_sin_previo = len(cohorte)
    fallecidos = fuas.loc[(fuas['fallecido'] == 1) & fuas['anio'].isin(anios_exclusion), 'paciente'].unique()
    cohorte = cohorte[~cohorte.isin(fallecidos)]
    n_vivos = len(cohorte)
    con_res = fuas.loc[fuas['anio'] == anio_res, 'paciente'].unique()
    cohorte = cohorte[cohorte.isin(con_res)]
    obs_txt = '-'.join(map(str, [anios_obs[0], anios_obs[-1]])) if len(anios_obs) > 1 else str(anios_obs[0])
    exc_txt = '-'.join(map(str, [anios_exclusion[0], anios_exclusion[-1]])) if len(anios_exclusion) > 1 else str(anios_exclusion[0])
    flujo = pd.DataFrame({
        'Paso': ['Pacientes en los datos', f'Con hipertensión ({obs_txt})', f'Sin complicación previa ({exc_txt})',
                 f'Sin fallecimiento registrado ({exc_txt})', f'Con atenciones en {anio_res} (cohorte final)'],
        'Pacientes': [n_total, len(con_hta), n_sin_previo, n_vivos, len(cohorte)]})

    # --- Objetivos ---
    res = dx_anual[(anio_dx == anio_res)].droplevel('anio')
    res = res.reindex(cohorte, fill_value=False).astype(bool)
    muerte = fuas.loc[fuas['anio'] == anio_res].groupby('paciente')['fallecido'].max().reindex(cohorte).fillna(0).astype(bool)
    obj = pd.DataFrame(index=cohorte)
    for e in EVENTOS:
        obj[f'y_{e}'] = res[e]
    obj['y_fallecido'] = muerte
    obj['y'] = obj[[f'y_{e}' for e in EVENTOS] + ['y_fallecido']].any(axis=1)
    obj['y_grave'] = res[[f'{e}_grave' for e in EVENTOS]].any(axis=1) | muerte
    obj['y_sin_n17'] = res[['acv', 'iam', 'ic', 'erc_sin_n17']].any(axis=1) | muerte
    obj['y_cardiovascular'] = res[['acv', 'iam', 'ic']].any(axis=1)
    obj = obj[OBJETIVOS].astype(int)

    # --- Variables desde FUAS ---
    f = fuas[fuas['paciente'].isin(cohorte) & fuas['anio'].isin(anios_obs)]
    X1 = f.groupby('paciente', observed=True).agg(
        edad=('EDAD', 'last'), masculino=('masculino', 'first'),
        n_atenciones=('ID_REGISTRO_REL', 'count'), n_emergencias=('emergencia', 'sum'),
        n_hospitalizaciones=('hospitalizado', 'sum'), dias_hosp=('DIAS_HOSP', 'sum'),
        n_mediciones_pa=('PRES_ART_SISTOLICA', 'count'),
        pas_media=('PRES_ART_SISTOLICA', 'mean'), pas_max=('PRES_ART_SISTOLICA', 'max'),
        pad_media=('PRES_ART_DIASTOLICA', 'mean'), pad_max=('PRES_ART_DIASTOLICA', 'max'),
        presion_pulso=('pp', 'mean'), presion_pulso_max=('pp', 'max'), prop_pa_alta=('pa_alta', 'mean'),
        pas_ultima=('PRES_ART_SISTOLICA', 'last'), ultima_atencion=('FECHA_ATENCION', 'max'),
        prop_capita=('capita', 'mean'), nivel_eess_max=('NIVEL_EESS', 'max'),
    )
    ultima = pd.to_datetime(X1.pop('ultima_atencion').astype(str), format='%Y%m%d')
    X1['dias_desde_ultima'] = (pd.Timestamp(str(fin_obs)) - ultima).dt.days
    dep = _moda_por_paciente(f, 'DEPARTAMENTO')
    ubi = _moda_por_paciente(f, 'UBIGEO')
    X1.insert(2, 'es_lima_frecuente', (dep.reindex(X1.index).astype(str) == 'LIMA').astype(int))
    X1.insert(3, 'altitud_msnm', ubi.reindex(X1.index).astype(str).map(t['altitud']))

    # --- Variables desde Diagnósticos ---
    d = dx_anual[pac_dx.isin(cohorte) & anio_dx.isin(anios_obs)]
    X2 = d.groupby(level='paciente')[COMORBILIDADES + ['hta_cardiaca']].max().astype(int)
    X2['n_comorbilidades'] = X2[COMORBILIDADES].sum(axis=1)
    ca = t['codigos_anual']
    ca = ca[ca['paciente'].isin(cohorte) & ca['anio'].isin(anios_obs)]
    X2['n_diagnosticos'] = ca.drop_duplicates(['paciente', 'CODDIA']).groupby('paciente', observed=True).size()

    # --- Variables desde Medicamentos, Procedimientos e Insumos ---
    partes = []
    for nombre, (anual, cods) in t['consumos'].items():
        a = anual[anual.index.get_level_values('paciente').isin(cohorte) & anual.index.get_level_values('anio').isin(anios_obs)]
        a = a.groupby(level='paciente').sum()
        if cods is not None:
            c = cods[cods['paciente'].isin(cohorte) & cods['anio'].isin(anios_obs)]
            n = c.drop_duplicates(['paciente', c.columns[2]]).groupby('paciente').size()
        else:
            n = a['n']
        partes.append(pd.DataFrame({f'n_{nombre}': n, f'gasto_{nombre}': a['gasto']}))

    df = X1.join(X2)
    for p in partes:
        df = df.join(p)
    cols_cero = [c for c in df.columns if c not in X1.columns]
    df[cols_cero] = df[cols_cero].fillna(0)
    df[[c for c in cols_cero if not c.startswith('gasto_')]] = df[[c for c in cols_cero if not c.startswith('gasto_')]].astype(int)
    df['polifarmacia'] = (df['n_medicamentos'] >= 5).astype(int)
    df['sin_medicion_pa'] = (df['n_mediciones_pa'] == 0).astype(int)
    df = df.join(obj)
    if codigos_como_indice:
        df.index = t['codigos_paciente'][df.index]
        df.index.name = 'paciente'
    return df, flujo
