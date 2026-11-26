"""Evalúa BTCUSDT Spot a una hora con un solo corte o con folds mensuales expansivos."""

import argparse
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import pandas as pd

from btc_forecast.data.dataset import construir_muestras
from btc_forecast.data.ingest import leer_rango_local
from btc_forecast.data.splits import corte_con_validacion_interna, corte_cronologico, folds_mensuales_expansivos
from btc_forecast.evaluation.metrics import brier_eventos, cobertura_intervalo, mae_niveles
from btc_forecast.features.indicators import MACDIncremental, RSIIncremental
from btc_forecast.features.labels import EVENTOS
from btc_forecast.features.transforms import aplicar_escalado
from btc_forecast.models import baselines


CARPETA_MENSUAL = Path("data/raw/binance_spot/BTCUSDT/1h")
INICIO_HISTORIA = pd.Timestamp("2025-01-01", tz="UTC")
INICIO_RESERVA = pd.Timestamp("2025-10-01", tz="UTC")
MESES_HISTORIA = 3
NUM_FOLDS = 6


def parametros_student_t(ajuste, interna, evaluacion, semilla, arquitectura, mu_cero):
    from btc_forecast.training.trainer import entrenar

    red, escalado, _, mejor = entrenar(ajuste, interna, semilla=semilla, arquitectura=arquitectura, mu_cero=mu_cero)
    print(f"{arquitectura} CPU: ajuste={len(ajuste)}, interna={len(interna)}, mejor época={mejor}")
    return inferir_parametros(red, escalado, evaluacion)


def lotes_inferencia(red, escalado, evaluacion):
    import torch

    red.eval()
    for inicio in range(0, len(evaluacion), 16):
        lote = evaluacion[inicio:inicio + 16]
        ventanas = [aplicar_escalado(m.X, escalado) for m in lote]
        x = torch.stack([torch.from_numpy(v.to_numpy(dtype="float32", copy=True)) for v in ventanas])
        with torch.inference_mode():
            salida = red(x)
        yield lote, salida


def inferir_parametros(red, escalado, evaluacion):
    parametros = {}
    for lote, (mu, s, nu) in lotes_inferencia(red, escalado, evaluacion):
        for m, centro, escala in zip(lote, mu.tolist(), s.tolist(), strict=True):
            parametros[m.fecha_prediccion] = (centro, escala, nu.item())
    return parametros


def comprobar_fechas(resultado, fechas, arquitectura):
    niveles = ["persistencia", "cierre_constante"]
    eventos = ["frecuencia_historica", "ewma_gaussiana", "ewma_empirica"]
    if arquitectura:
        niveles.append(arquitectura)
        eventos.append(arquitectura)
    claves = ["fecha_prediccion", "modelo", "objetivo"]
    esperadas = pd.MultiIndex.from_product([fechas, niveles, ["rsi", "macd"]], names=claves).append(
        pd.MultiIndex.from_product([fechas, eventos, EVENTOS], names=claves)
    )
    presentes = pd.MultiIndex.from_frame(resultado[claves])
    problemas = {
        "faltantes": esperadas.difference(presentes),
        "duplicadas": presentes[presentes.duplicated()].unique(),
        "inesperadas": presentes.difference(esperadas),
    }
    print("Control de fechas por modelo y objetivo:", ", ".join(f"{k}={len(v)}" for k, v in problemas.items()))
    if any(len(v) for v in problemas.values()):
        raise ValueError({k: v.tolist() for k, v in problemas.items() if len(v)})
    indefinidas = resultado.loc[resultado["prediccion"].isna(), claves]
    if not indefinidas.empty:
        print("Predicciones indefinidas; se excluyen esas fechas para todos los modelos del objetivo:")
        print(indefinidas.to_string(index=False))


def evaluar_corte(velas, corte, incluir_transformer=False, semilla=42, *, arquitectura=None, mu_cero=False):
    muestras = construir_muestras(velas)
    if incluir_transformer or arquitectura:
        entrenamiento, interna, validacion = corte_con_validacion_interna(muestras, corte)
    else:
        entrenamiento, validacion = corte_cronologico(muestras, corte)
        interna = []
    resultado = evaluar_tramos(velas, corte, entrenamiento, interna, validacion,
                               fold=1, arquitectura=arquitectura or ("transformer" if incluir_transformer else None),
                               semilla=semilla, mu_cero=mu_cero)
    return resultado, len(entrenamiento), len(validacion)


def evaluar_tramos(velas, corte, ajuste, interna, validacion, *, fold, arquitectura, semilla, mu_cero):
    """Ajusta las referencias en ajuste y evalúa todos los modelos sobre los mismos orígenes."""
    if not ajuste or not validacion:
        raise ValueError("El corte debe dejar muestras de entrenamiento y validación.")
    if arquitectura:
        from btc_forecast.models.forecast import pronosticar

        parametros = parametros_student_t(ajuste, interna, validacion, semilla, arquitectura, mu_cero)
    frecuencias = baselines.ajustar_frecuencias(ajuste)
    residuos = baselines.ajustar_residuos_ewma(ajuste)
    filas = []
    for m, rsi, macd, actual in estados_evaluacion(velas, validacion):
        fecha = m.fecha_prediccion
        niveles = {
            "persistencia": baselines.persistencia(actual),
            "cierre_constante": baselines.cierre_constante(rsi, macd),
        }
        probabilidades = {
            "frecuencia_historica": baselines.pronosticar_eventos(m, frecuencias),
            "ewma_gaussiana": baselines.ewma_gaussiana(m, rsi, macd)["Probabilidad"],
            "ewma_empirica": baselines.ewma_empirica(m, rsi, macd, residuos)["Empírica"],
        }
        elegibles = baselines.elegibilidad(m)
        if arquitectura:
            eventos_t, cuantiles = pronosticar(*parametros[fecha], m, rsi, macd)
            mediana = cuantiles.loc[0.5]
            niveles[arquitectura] = baselines.Niveles(mediana["RSI"], mediana["MACD"])
            probabilidades[arquitectura] = eventos_t["probabilidad"]
        # Evalúa t+1 en una copia del estado de t.
        observado = baselines.Niveles(deepcopy(rsi).actualizar(m.cierre_t1),
                                      deepcopy(macd).actualizar(m.cierre_t1)[2])
        for modelo, pronostico in niveles.items():
            for indicador, valor in pronostico._asdict().items():
                filas.append((fecha, m.fecha_objetivo, modelo, "nivel", indicador, valor,
                              getattr(observado, indicador), None, None, None, None, None))
        for modelo, probabilidades_modelo in probabilidades.items():
            for evento, p in probabilidades_modelo.items():
                detalle = (None, None, None, None)
                if modelo == arquitectura:
                    detalle = (*parametros[fecha], eventos_t.at[evento, "cierre_critico"])
                filas.append((fecha, m.fecha_objetivo, modelo, "evento", evento, p,
                              m.etiquetas[evento], elegibles[evento], *detalle))

    resultado = pd.DataFrame(filas, columns=["fecha_prediccion", "fin_etiqueta", "modelo", "tipo",
                                             "objetivo", "prediccion", "observado", "elegible",
                                             "mu", "s", "nu", "cierre_critico"])
    resultado["elegible"] = resultado["elegible"].astype("boolean")
    resultado = resultado.assign(mercado="BTCUSDT Spot", horizonte_horas=1,
                                 fold=fold, corte=corte, semilla=semilla)
    es_evento = resultado.tipo.eq("evento")
    resultado["evento"] = resultado.objetivo.where(es_evento)
    resultado["probabilidad"] = resultado.prediccion.where(es_evento)
    resultado["etiqueta"] = resultado.observado.where(es_evento).astype("Int8")
    comprobar_fechas(resultado, [m.fecha_prediccion for m in validacion], arquitectura)
    if arquitectura:
        resultado["modo_mu"] = pd.Series("cero" if mu_cero else "aprendida", index=resultado.index).where(
            resultado.modelo.eq(arquitectura))
        elegibles = resultado.loc[resultado.modelo.eq(arquitectura) & resultado.evento.eq("macd_alcista")
                                  & resultado.elegible]
        print("Primera fecha elegible para MACD alcista (UTC)")
        columnas = ["fecha_prediccion", "modelo", "modo_mu", "cierre_critico", "mu", "s", "nu",
                    "probabilidad", "etiqueta"]
        print(elegibles.head(1)[columnas].to_string(index=False, float_format=lambda v: f"{v:.9f}"))
    return resultado


def resumir(predicciones):
    """Por objetivo, compara los modelos en las fechas con predicciones comunes."""
    niveles, eventos = [], []
    for (tipo, objetivo), grupo in predicciones.groupby(["tipo", "objetivo"], sort=False):
        p = grupo.pivot(index="fecha_prediccion", columns="modelo", values="prediccion")
        datos = grupo.drop_duplicates("fecha_prediccion").set_index("fecha_prediccion")
        y = datos["observado"].reindex(p.index)
        comun = p.notna().all(axis=1) & y.notna()
        observaciones = pd.DataFrame({modelo: y for modelo in p})
        if tipo == "nivel":
            metricas = mae_niveles(p.loc[comun], observaciones.loc[comun])
            niveles.append({"indicador": objetivo, "n": int(comun.sum()), **metricas["MAE"]})
        else:
            mascaras = {"todas": comun, "elegibles": comun & datos["elegible"]}
            for poblacion, mascara in mascaras.items():
                metricas = brier_eventos(p.loc[mascara], observaciones.loc[mascara])
                eventos.append({"evento": objetivo, "velas": poblacion, "n": int(mascara.sum()),
                                "positivos": int(y.loc[mascara].sum()), **metricas["Brier"]})
    return pd.DataFrame(niveles), pd.DataFrame(eventos)


def escribir_parquet(predicciones, destino):
    with NamedTemporaryFile(dir=destino.parent, delete=False) as archivo:
        temporal = Path(archivo.name)
    try:
        predicciones.to_parquet(temporal, index=False)
        temporal.replace(destino)
    finally:
        temporal.unlink(missing_ok=True)


def guardar_resultados(predicciones, corrida, nombre):
    mae, brier = resumir(predicciones)
    destino = corrida / f"predicciones_{nombre}.parquet"
    escribir_parquet(predicciones.assign(corrida=corrida.name), destino)
    print("\nMAE por indicador")
    print(mae.to_string(index=False))
    print("\nBrier por evento y población")
    print(brier.to_string(index=False))
    print(f"\n{len(predicciones)} predicciones guardadas en {destino}")


def ejecutar_folds_mensuales(seleccion, carpeta_datos, corrida, arquitectura, semilla, mu_cero):
    """Lee hasta el final del último mes solicitado y evalúa cada fold seleccionado desde cero."""
    todos = seleccion == "todos"
    ultimo = NUM_FOLDS if todos else int(seleccion)
    fin = INICIO_HISTORIA + pd.DateOffset(months=MESES_HISTORIA + ultimo)
    velas = leer_rango_local(INICIO_HISTORIA, fin, carpeta_datos)
    folds = folds_mensuales_expansivos(
        construir_muestras(velas), INICIO_HISTORIA, meses_historia=MESES_HISTORIA,
        meses_evaluacion=ultimo, inicio_reserva=INICIO_RESERVA,
    )
    total = 0
    for numero, (corte, ajuste, interna, evaluacion) in enumerate(folds, start=1):
        if not todos and numero != ultimo:
            continue
        print(f"Fold {numero} | {corte:%Y-%m} UTC | Ajuste: {len(ajuste)} | "
              f"Interna: {len(interna)} | Evaluación: {len(evaluacion)}", flush=True)
        resultado = evaluar_tramos(velas, corte, ajuste, interna, evaluacion, fold=numero,
                                   arquitectura=arquitectura, semilla=semilla, mu_cero=mu_cero)
        guardar_resultados(resultado, corrida, f"fold{numero:02d}_{corte:%Y-%m}")
        total += len(resultado)
    return total


def estados_evaluacion(velas, evaluacion):
    """Avanza los indicadores hasta cada origen sin perder su historial previo."""
    origenes = {m.fecha_prediccion: m for m in evaluacion}
    fechas = pd.to_datetime(velas["close_time"], unit="ms", utc=True)
    cierres = velas.set_index(fechas)["close"].sort_index().loc[:evaluacion[-1].fecha_prediccion]
    anterior = None
    for fecha, cierre in cierres.items():
        if anterior is None or fecha - anterior != pd.Timedelta(hours=1):
            rsi, macd = RSIIncremental(), MACDIncremental()
        anterior = fecha
        nivel_rsi = rsi.actualizar(float(cierre))
        valores_macd = macd.actualizar(float(cierre))
        if fecha in origenes:
            yield origenes[fecha], rsi, macd, baselines.Niveles(nivel_rsi, valores_macd[2])


def emparejar_niveles(actual, anterior):
    """Exige exactamente un registro por origen y la misma observación en ambas tablas."""
    for nombre, tabla in (("regresión directa", actual), ("Student-t", anterior)):
        fechas = tabla.index
        invalidas = fechas[fechas.isna() | fechas.duplicated(keep=False)]
        if len(invalidas):
            raise ValueError(f"Orígenes nulos o duplicados en {nombre}: {invalidas.unique().tolist()}")
    faltantes = anterior.index.difference(actual.index)
    adicionales = actual.index.difference(anterior.index)
    if len(faltantes) or len(adicionales):
        raise ValueError(f"Orígenes ausentes de la corrida directa: {faltantes.tolist()}; "
                         f"orígenes adicionales: {adicionales.tolist()}")
    anterior = anterior.loc[actual.index]
    for campo in ("fin_etiqueta", "cierre_observado", "cierre_t", "rsi_observado", "macd_observado"):
        par = pd.DataFrame({"enero": actual[campo], "student_t": anterior[campo]})
        diferencias = par.loc[par.isna().any(axis=1) | par.enero.ne(par.student_t)]
        if not diferencias.empty:
            raise ValueError(f"{len(diferencias)} diferencias en {campo}:\n{diferencias.to_string()}")
    print(f"Emparejamiento verificado en {len(actual)} orígenes UTC únicos. "
          "Faltantes=0, adicionales=0, duplicados=0, diferencias de cierre y observación=0.")
    return actual.join(anterior[["rsi_p50", "macd_p50"]].rename(
        columns={"rsi_p50": "rsi_student_t", "macd_p50": "macd_student_t"}), validate="one_to_one")


def evaluar_niveles_directos(red, escalado, velas, evaluacion, ruta_student_t):
    from btc_forecast.models.forecast import reconstruir_regresion_directa

    anterior = pd.read_parquet(ruta_student_t).set_index("fecha_prediccion")
    if anterior["modelo"].unique().tolist() != ["transformer"] or anterior.checkpoint.nunique(dropna=False) != 1:
        raise ValueError("El Parquet Student-t debe contener un único checkpoint Transformer.")
    salidas = {m.fecha_prediccion: z for lote, salida in lotes_inferencia(red, escalado, evaluacion)
               for m, z in zip(lote, salida[:, 0, :].tolist(), strict=True)}
    filas = []
    for m, rsi, macd, actual in estados_evaluacion(velas, evaluacion):
        _, directa = reconstruir_regresion_directa(
            salidas[m.fecha_prediccion], red.escalado_objetivos,
            rsi_t=actual.rsi, macd_t=actual.macd, cierre_t=m.cierre_t)
        fila = dict(fecha_prediccion=m.fecha_prediccion, fin_etiqueta=m.fecha_objetivo,
                    cierre_t=m.cierre_t, cierre_observado=m.cierre_t1,
                    rsi_observado=deepcopy(rsi).actualizar(m.cierre_t1),
                    macd_observado=deepcopy(macd).actualizar(m.cierre_t1)[2])
        for nombre, niveles in (("persistencia", baselines.persistencia(actual)),
                                ("cierre_constante", baselines.cierre_constante(rsi, macd)),
                                ("directa", directa)):
            fila[f"rsi_{nombre}"], fila[f"macd_{nombre}"] = niveles
        filas.append(fila)
    tabla = emparejar_niveles(pd.DataFrame(filas).set_index("fecha_prediccion"), anterior)
    tabla["checkpoint_student_t"] = anterior.checkpoint.iloc[0]
    return tabla


def comparar_mae_niveles(tabla):
    """Calcula los cuatro MAE sobre el mismo panel completo de fechas."""
    metodos = ["persistencia", "cierre_constante", "student_t", "directa"]
    resultados = []
    for indicador, referencia in (("rsi", "persistencia"), ("macd", "cierre_constante")):
        predicciones = pd.DataFrame({m: tabla[f"{indicador}_{m}"] for m in metodos})
        observado = tabla[f"{indicador}_observado"]
        metricas = mae_niveles(predicciones, pd.DataFrame({m: observado for m in metodos}))
        if not metricas["n"].eq(len(tabla)).all():
            raise ValueError(f"Predicciones u observaciones incompletas para {indicador}:\n{metricas}")
        base = metricas.at[referencia, "MAE"]
        metricas["mejora_pct"] = 100 * (1 - metricas.MAE / base) if base > 0 else float("nan")
        resultados.append(metricas.assign(indicador=indicador, referencia=referencia))
    return pd.concat(resultados).rename_axis("metodo").reset_index()


def guardar_tabla_checkpoint(tabla, ruta, nombre):
    instante = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ")
    destino = Path("artifacts/walkforward") / f"{instante}_{ruta.stem}" / nombre
    destino.parent.mkdir(parents=True, exist_ok=False)
    escribir_parquet(tabla.reset_index(), destino)
    print(f"{len(tabla)} velas guardadas en {destino}")


def evaluar_checkpoint(ruta, datos, predicciones_student_t=None):
    """Evalúa con los pesos, escalado y fronteras temporales del checkpoint."""
    from btc_forecast.models.forecast import pronosticar
    from btc_forecast.training.trainer import cargar_checkpoint

    red, escalado, meta = cargar_checkpoint(ruta)
    directa = meta["modelo"] == "directa"
    if directa != (predicciones_student_t is not None):
        raise ValueError("La comparación requiere un checkpoint directo y --predicciones-student-t juntos.")
    velas = pd.read_parquet(datos)
    muestras = construir_muestras(velas, longitud=meta["ventana"])
    cortes = meta["cortes"]
    previas, evaluacion = corte_cronologico(muestras, cortes["inicio_evaluacion"])
    _, interna = corte_cronologico(previas, cortes["inicio_validacion_interna"])
    if not interna or not evaluacion or interna[0].fecha_prediccion != pd.Timestamp(cortes["inicio_validacion_interna"]):
        raise ValueError("Los datos deben cubrir la validación interna del checkpoint y su evaluación posterior.")
    ultima, primera = interna[-1], evaluacion[0]
    detalle = "" if directa else f", y={ultima.y:.10f}"
    print(f"Última etiqueta interna: origen {ultima.fecha_prediccion}, fin {ultima.fecha_objetivo}{detalle}")
    print(f"Primera predicción evaluada: {primera.fecha_prediccion}, fin {primera.fecha_objetivo}")
    if directa:
        tabla = evaluar_niveles_directos(red, escalado, velas, evaluacion, predicciones_student_t)
        metricas = comparar_mae_niveles(tabla)
        print(metricas.to_string(index=False, float_format=lambda valor: f"{valor:.6f}"))
        guardar_tabla_checkpoint(tabla.assign(checkpoint_directo=ruta.name, semilla_directa=meta["semilla"]),
                                 ruta, "comparacion_niveles.parquet")
        return
    parametros = inferir_parametros(red, escalado, evaluacion)
    filas = []
    for m, rsi, macd, _ in estados_evaluacion(velas, evaluacion):
        mu, s, nu = parametros[m.fecha_prediccion]
        eventos, q = pronosticar(mu, s, nu, m, rsi, macd)
        fila = dict(fecha_prediccion=m.fecha_prediccion, fin_etiqueta=m.fecha_objetivo,
                    cierre_t=m.cierre_t, cierre_observado=m.cierre_t1, mu=mu, s=s, nu=nu,
                    cierre_p10=q.at[0.1, "cierre"], cierre_p50=q.at[0.5, "cierre"], cierre_p90=q.at[0.9, "cierre"],
                    rsi_p50=q.at[0.5, "RSI"], macd_p50=q.at[0.5, "MACD"],
                    rsi_observado=deepcopy(rsi).actualizar(m.cierre_t1),
                    macd_observado=deepcopy(macd).actualizar(m.cierre_t1)[2])
        for evento in EVENTOS:
            fila[f"p_{evento}"] = eventos.at[evento, "probabilidad"]
            fila[f"y_{evento}"] = m.etiquetas[evento]
            fila[f"elegible_{evento}"] = eventos.at[evento, "elegible"]
        filas.append(fila)
    tabla = pd.DataFrame(filas).set_index("fecha_prediccion")
    if not tabla.index.is_unique or tabla.index.tolist() != [m.fecha_prediccion for m in evaluacion]:
        raise ValueError("Las fechas pronosticadas deben coincidir una a una con las muestras de evaluación.")

    def columnas(nombres, patron):
        return pd.DataFrame({n: tabla[patron.format(n)] for n in nombres})

    niveles = ["rsi", "macd"]
    print(mae_niveles(columnas(niveles, "{}_p50"), columnas(niveles, "{}_observado")))
    print(cobertura_intervalo(columnas(["cierre"], "{}_p10"), columnas(["cierre"], "{}_p90"),
                              columnas(["cierre"], "{}_observado")))
    y = columnas(EVENTOS, "y_{}")
    print(brier_eventos(columnas(EVENTOS, "p_{}"), y).assign(positivos=y.sum()))
    guardar_tabla_checkpoint(tabla.assign(checkpoint=ruta.name, semilla=meta["semilla"],
                             modelo=meta["modelo"], epoca=meta["epoca"]), ruta, "predicciones_checkpoint.parquet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument("--checkpoint", type=Path, help="Evalúa el modelo guardado desde su corte de evaluación.")
    modo.add_argument("--corte", default="2025-01-20T11:59:59.999Z", help="Fecha ISO del corte único (UTC).")
    modo.add_argument("--fold", choices=[*(str(i) for i in range(1, NUM_FOLDS + 1)), "todos"],
                      help="Fold mensual de 2025 (1 = abril, 6 = septiembre) o todos.")
    parser.add_argument("--datos", type=Path, help="Parquet para corte único, o carpeta de meses para --fold.")
    parser.add_argument("--predicciones-student-t", type=Path, metavar="PARQUET",
                        help="Predicciones Student-t anteriores que se comparan con el checkpoint directo.")
    red = parser.add_mutually_exclusive_group()
    red.add_argument("--incluir-transformer", dest="arquitectura", action="store_const", const="transformer")
    red.add_argument("--arquitectura", choices=("lineal", "transformer"))
    parser.add_argument("--modo-mu", choices=("cero", "aprendida"), help="Por defecto, aprendida.")
    parser.add_argument("--semilla", type=int)
    args = parser.parse_args()
    if args.predicciones_student_t and not args.checkpoint:
        parser.error("--predicciones-student-t requiere --checkpoint.")
    if args.checkpoint:
        if args.arquitectura or args.modo_mu or args.semilla is not None:
            parser.error("Con --checkpoint, arquitectura, modo de mu y semilla se leen del archivo.")
        evaluar_checkpoint(args.checkpoint, args.datos or CARPETA_MENSUAL / "2025-01.parquet",
                           args.predicciones_student_t)
        return
    if args.semilla is None:
        args.semilla = 42
    if args.modo_mu and not args.arquitectura:
        parser.error("--modo-mu requiere una arquitectura.")
    mu_cero = args.modo_mu == "cero"

    inicio = perf_counter()
    modo_corrida = f"fold_{args.fold}" if args.fold else "corte"
    modelos = f"{args.arquitectura}_mu_{args.modo_mu or 'aprendida'}" if args.arquitectura else "referencias"
    instante = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ")
    corrida = Path("artifacts/walkforward") / f"{instante}_{modo_corrida}_{modelos}_s{args.semilla}"
    corrida.mkdir(parents=True, exist_ok=False)
    if args.fold:
        total = ejecutar_folds_mensuales(args.fold, args.datos or CARPETA_MENSUAL, corrida,
                                         args.arquitectura, args.semilla, mu_cero)
    else:
        corte = pd.to_datetime(args.corte, utc=True)
        velas = pd.read_parquet(args.datos or CARPETA_MENSUAL / "2025-01.parquet")
        predicciones, n_entrenamiento, n_validacion = evaluar_corte(
            velas, corte, semilla=args.semilla, arquitectura=args.arquitectura, mu_cero=mu_cero
        )
        print(f"Corte UTC: {corte} | Entrenamiento: {n_entrenamiento} | Validación: {n_validacion}")
        guardar_resultados(predicciones, corrida, f"corte_{corte.value}")
        total = len(predicciones)
    print(f"Total: {total} predicciones | Tiempo de corrida: {perf_counter() - inicio:.2f} s")


if __name__ == "__main__":
    main()
