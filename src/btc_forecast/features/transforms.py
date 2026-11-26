"""Variables horarias causales de BTCUSDT, indexadas por la apertura UTC."""

from collections import deque
from math import fsum, isfinite, log, sqrt
from pathlib import Path

import pandas as pd

from btc_forecast.features.indicators import MACDIncremental, RSIIncremental


VARIABLES = (
    "retorno_log", "rango_relativo", "volumen_relativo_20", "rsi_100",
    "macd_cierre", "senal_cierre", "histograma_cierre", "volatilidad_ewma",
)
HORA = pd.Timedelta(hours=1)


def calcular_variables(tabla, lambda_ewma=0.94, calentamiento=200):
    """Reinicia los indicadores tras cada hueco. La varianza EWMA empieza en r².

    La volatilidad tiene piso 1e-12. El calentamiento y los cocientes de volumen
    sin media válida quedan en NaN.
    """
    if not 0 < lambda_ewma < 1 or calentamiento < 0:
        raise ValueError("Se requiere 0 < lambda_ewma < 1 y calentamiento no negativo.")
    aperturas = pd.to_datetime(tabla["open_time"], unit="ms", utc=True)
    if aperturas.isna().any() or aperturas.duplicated().any():
        raise ValueError("Se requieren aperturas válidas y únicas.")
    datos = tabla[["high", "low", "close", "volume"]].set_index(aperturas).sort_index()
    filas = []
    anterior = None
    for fecha, alto, bajo, cierre, volumen in datos.astype(float).itertuples(name=None):
        if anterior is None or fecha - anterior != HORA:
            macd = MACDIncremental()
            rsi = RSIIncremental()
            volumenes = deque(maxlen=20)
            cierre_previo = varianza = None
            cantidad = 0
        if not all(map(isfinite, (alto, bajo, cierre, volumen))) or cierre <= 0 or volumen < 0:
            raise ValueError(f"Datos inválidos en {fecha}. Se requiere cierre positivo y volumen no negativo.")
        valores_macd = macd.actualizar(cierre)
        nivel = rsi.actualizar(cierre)
        volumenes.append(volumen)
        cantidad += 1
        retorno = None
        if cierre_previo is not None:
            retorno = log(cierre / cierre_previo)
            varianza = (retorno**2 if varianza is None else
                        lambda_ewma * varianza + (1 - lambda_ewma) * retorno**2)
        anterior, cierre_previo = fecha, cierre
        if cantidad <= calentamiento:
            filas.append((None,) * len(VARIABLES))
            continue

        media = fsum(volumenes) / 20 if len(volumenes) == 20 else None
        normalizados = (None, None, None) if valores_macd is None else tuple(
            valor / cierre for valor in valores_macd[2:]
        )
        filas.append((
            retorno, (alto - bajo) / cierre, volumen / media if media else None,
            nivel / 100 if nivel is not None else None, *normalizados,
            max(sqrt(varianza), 1e-12) if varianza is not None else None,
        ))
    return pd.DataFrame(filas, index=datos.index, columns=VARIABLES, dtype=float)


def ajustar_escalado(ventanas_ajuste):
    """Ajusta media y desviación poblacional de X con una fila por cierre UTC.

    Las columnas constantes usan escala 1. El objetivo conserva su log-retorno.
    """
    ventanas = list(ventanas_ajuste)
    if not ventanas:
        raise ValueError("Se necesita al menos una ventana de ajuste.")
    for x in ventanas:
        if tuple(x.columns) != VARIABLES or not isinstance(x.index, pd.DatetimeIndex) or x.index.hasnans:
            raise ValueError("Cada ventana debe conservar las variables y tener fechas válidas.")
    filas = pd.concat(ventanas)
    # Dos valores distintos para la misma fecha indican un conflicto.
    unicas = filas.rename_axis("fecha").reset_index().drop_duplicates().set_index("fecha").sort_index()
    if unicas.index.has_duplicates:
        raise ValueError("Hay valores distintos para la misma fecha de ajuste.")
    if unicas.empty or unicas.isin([float("inf"), -float("inf")]).any().any() or unicas.isna().any().any():
        raise ValueError("El ajuste requiere filas completas y finitas.")
    constantes = unicas.min().eq(unicas.max())
    centro = unicas.mean().mask(constantes, unicas.iloc[0])
    escala = unicas.std(ddof=0).mask(constantes, 1.0)
    return pd.DataFrame({"centro": centro, "escala": escala})


def aplicar_escalado(ventana, parametros):
    """Devuelve una nueva X en el orden original usando el ajuste ya calculado."""
    if tuple(ventana.columns) != VARIABLES or tuple(parametros.index) != VARIABLES:
        raise ValueError("Las variables y su orden deben coincidir con VARIABLES.")
    return (ventana - parametros["centro"]) / parametros["escala"]


def demostrar_escalado(tabla):
    from btc_forecast.data.dataset import construir_muestras
    from btc_forecast.data.splits import corte_con_validacion_interna

    corte = pd.Timestamp("2025-01-20T11:59:59.999Z")
    ajuste, interna, _ = corte_con_validacion_interna(construir_muestras(tabla), corte)
    parametros = ajustar_escalado(m.X for m in ajuste)
    muestra = interna[0]
    ventana = muestra.X.tail(4)
    escalada = aplicar_escalado(ventana, parametros)
    print(f"\nAjuste: {len(ajuste)} ventanas, {len(pd.concat([m.X for m in ajuste]).index.unique())} cierres únicos")
    print("Ventana de cuatro velas antes del escalado:")
    print(ventana.to_string(float_format=lambda x: f"{x:.6f}"))
    print("Después del escalado:")
    print(escalada.to_string(float_format=lambda x: f"{x:.6f}"))
    assert muestra.y == log(muestra.cierre_t1 / muestra.cierre_t)
    print(f"Objetivo en log-retorno: {muestra.y:.10f}")

    alterada = tabla.copy()
    fechas = pd.to_datetime(alterada["close_time"], unit="ms", utc=True)
    mascara = fechas.between(muestra.fecha_prediccion, interna[-1].fecha_prediccion)
    alterada["volume"] = alterada["volume"].astype(float)
    alterada.loc[mascara, "volume"] *= 3
    nuevo_ajuste, nueva_interna, _ = corte_con_validacion_interna(construir_muestras(alterada), corte)
    assert not muestra.X.equals(nueva_interna[0].X)
    pd.testing.assert_frame_equal(parametros, ajustar_escalado(m.X for m in nuevo_ajuste), check_exact=True)
    print("Verificado: triplicar el volumen en validación cambia X y conserva exactamente el ajuste.")


def main():
    ruta = Path("data/raw/binance_spot/BTCUSDT/1h/2025-01.parquet")
    tabla = pd.read_parquet(ruta).sort_values("open_time")
    fecha = pd.Timestamp("2025-01-09 08:00", tz="UTC")
    fin = fecha + HORA - pd.Timedelta(milliseconds=1)
    prefijo = tabla.loc[tabla["close_time"] <= fin.value // 1_000_000]
    fila = calcular_variables(tabla).loc[fecha]
    truncada = calcular_variables(prefijo).loc[fecha]
    if not fila.notna().all():
        raise ValueError("Faltan variables para la vela del ejemplo.")
    pd.testing.assert_series_equal(fila, truncada, check_exact=True)
    previo, cierre = prefijo["close"].tail(2).astype(float)
    volumenes = prefijo["volume"].tail(20).astype(float)
    print(f"Apertura UTC: {fecha} | Cierre UTC: {fin}")
    print(fila.to_string(float_format=lambda valor: f"{valor:.10f}"))
    print(f"Retorno = ln({cierre:.2f} / {previo:.2f}) = {fila.retorno_log:.10f}")
    print(f"Volumen relativo = {volumenes.iloc[-1]:.5f} / ({fsum(volumenes):.5f} / 20)"
          f" = {fila.volumen_relativo_20:.10f}")
    print("Las ocho variables coinciden exactamente al truncar en el cierre.")
    demostrar_escalado(tabla)


if __name__ == "__main__":
    main()
