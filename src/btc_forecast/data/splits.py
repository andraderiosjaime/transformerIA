"""Corte cronológico según el instante de predicción y de observación del objetivo."""

import argparse
from collections.abc import Iterable
from math import ceil
from pathlib import Path

import pandas as pd

from btc_forecast.data.dataset import Muestra, construir_muestras
from btc_forecast.data.ingest import leer_rango_local


def corte_cronologico(muestras: Iterable[Muestra], corte) -> tuple[list[Muestra], list[Muestra]]:
    """Entrena con etiquetas conocidas al corte y valida desde ese instante.

    Descarta los orígenes anteriores cuya etiqueta cruza el corte. Las fechas sin zona son UTC.
    """
    corte = pd.Timestamp(corte)
    if pd.isna(corte):
        raise ValueError("Se requiere un instante de corte válido.")
    if corte.tzinfo is None:
        corte = corte.tz_localize("UTC")
    entrenamiento = []
    validacion = []
    for muestra in sorted(muestras, key=lambda m: m.fecha_prediccion):
        prediccion = muestra.fecha_prediccion
        fin_etiqueta = muestra.fecha_objetivo
        if not prediccion < fin_etiqueta:
            raise ValueError("El objetivo debe conocerse después de la predicción.")
        if fin_etiqueta <= corte:
            entrenamiento.append(muestra)
        elif prediccion >= corte:
            validacion.append(muestra)
    return entrenamiento, validacion


def corte_con_validacion_interna(
    muestras: Iterable[Muestra], corte, proporcion_interna: float = 0.20
) -> tuple[list[Muestra], list[Muestra], list[Muestra]]:
    """Reserva para validación interna la última fracción de muestras, redondeada hacia arriba.

    En cada corte se admiten etiquetas conocidas justo en la frontera.
    """
    if not 0 < proporcion_interna < 1:
        raise ValueError("La proporción interna debe estar entre 0 y 1, sin incluirlos.")
    previas, evaluacion = corte_cronologico(muestras, corte)
    n_interna = ceil(len(previas) * proporcion_interna)
    if n_interna >= len(previas):
        raise ValueError("Faltan muestras anteriores al corte para formar ajuste y validación interna.")
    corte_interno = previas[-n_interna].fecha_prediccion
    ajuste, interna = corte_cronologico(previas, corte_interno)
    if not ajuste:
        raise ValueError("Ninguna etiqueta de ajuste se conoce al inicio de la validación interna.")
    return ajuste, interna, evaluacion


def folds_mensuales_expansivos(muestras: Iterable[Muestra], inicio_historia, *,
                              meses_historia=3, meses_evaluacion=6,
                              proporcion_interna=0.20, inicio_reserva=None):
    """Devuelve (inicio de mes UTC, ajuste, interna, evaluación) para cada fold.

    Origen y etiqueta deben caer en el mes evaluado. Por defecto, la reserva empieza el mes siguiente al desarrollo.
    """
    inicio = pd.to_datetime(inicio_historia, utc=True)
    if pd.isna(inicio) or inicio.day != 1 or inicio != inicio.normalize():
        raise ValueError("La historia debe empezar el primer día del mes a medianoche UTC.")
    if meses_historia < 1 or meses_evaluacion < 1:
        raise ValueError("Se necesita al menos un mes de historia y uno de evaluación.")
    primero = inicio + pd.DateOffset(months=meses_historia)
    limites = pd.date_range(primero, periods=meses_evaluacion + 1, freq="MS")
    reserva = limites[-1] if inicio_reserva is None else pd.to_datetime(inicio_reserva, utc=True)
    if pd.isna(reserva) or reserva < limites[-1]:
        raise ValueError("El periodo reservado debe empezar después de los meses de desarrollo.")
    desarrollo = [m for m in muestras if inicio <= m.fecha_prediccion < reserva
                  and m.fecha_objetivo < reserva]
    for corte, fin in zip(limites[:-1], limites[1:]):
        hasta_fin = (m for m in desarrollo if m.fecha_prediccion < fin and m.fecha_objetivo < fin)
        ajuste, interna, evaluacion = corte_con_validacion_interna(hasta_fin, corte, proporcion_interna)
        if not evaluacion:
            raise ValueError(f"Sin muestras de evaluación completas para {corte:%Y-%m}.")
        yield corte, ajuste, interna, evaluacion


def mostrar_folds_mensuales(proporcion_interna):
    inicio = pd.Timestamp("2025-01-01", tz="UTC")
    reserva = pd.Timestamp("2025-10-01", tz="UTC")
    muestras = construir_muestras(leer_rango_local(inicio, reserva))
    folds = folds_mensuales_expansivos(muestras, inicio, proporcion_interna=proporcion_interna,
                                      inicio_reserva=reserva)
    print(f"Inicio de reserva (UTC): {reserva.isoformat()}")
    print("Fold | Mes | Tramo | Primera predicción UTC | Última predicción UTC | Último fin de etiqueta UTC | n")
    for numero, (corte, ajuste, interna, evaluacion) in enumerate(folds, start=1):
        fin = corte + pd.offsets.MonthBegin(1)
        assert max(m.fecha_objetivo for m in ajuste) <= interna[0].fecha_prediccion
        assert max(m.fecha_objetivo for m in interna) <= corte <= evaluacion[0].fecha_prediccion
        assert max(m.fecha_objetivo for m in evaluacion) < fin <= reserva
        for nombre, tramo in [("Ajuste", ajuste), ("Interna", interna), ("Evaluación", evaluacion)]:
            print(numero, corte.strftime("%Y-%m"), nombre, tramo[0].fecha_prediccion,
                  tramo[-1].fecha_prediccion, max(m.fecha_objetivo for m in tramo), len(tramo), sep=" | ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds-mensuales", action="store_true",
                        help="Construye los seis folds de abril a septiembre de 2025.")
    parser.add_argument("--corte", default="2025-01-20T11:59:59.999Z",
                        help="Instante de corte ISO 8601 (UTC por defecto).")
    parser.add_argument("--proporcion-interna", type=float, default=0.20,
                        help="Fracción final del bloque anterior destinada a validación interna.")
    args = parser.parse_args()
    if args.folds_mensuales:
        mostrar_folds_mensuales(args.proporcion_interna)
        return
    ruta = Path("data/raw/binance_spot/BTCUSDT/1h/2025-01.parquet")
    muestras = list(construir_muestras(pd.read_parquet(ruta)))
    entrenamiento, validacion = corte_cronologico(muestras, args.corte)
    print(f"Corte: {args.corte}")
    print(f"Entrenamiento: {len(entrenamiento)} muestras | Validación: {len(validacion)} muestras")
    print("Grupo | Inicio de X (UTC) | Predicción (UTC) | Fin etiqueta (UTC) | y")
    for grupo, seleccion in [("Entrenamiento", entrenamiento[-2:]), ("Validación", validacion[:2])]:
        for m in seleccion:
            print(grupo, m.X.index[0].isoformat(), m.fecha_prediccion.isoformat(),
                  m.fecha_objetivo.isoformat(), f"{m.y:.8f}", sep=" | ")
    if entrenamiento and validacion:
        compartidas = entrenamiento[-1].X.index.intersection(validacion[0].X.index)
        print(f"Velas compartidas por las dos ventanas contiguas al corte: {len(compartidas)}")

    try:
        ajuste, interna, evaluacion = corte_con_validacion_interna(muestras, args.corte, args.proporcion_interna)
    except ValueError as error:
        parser.error(str(error))
    print(f"\nValidación interna solicitada: {args.proporcion_interna:.2%}")
    print("Tramo | Primera predicción UTC | Última predicción UTC | Último fin de etiqueta UTC | Muestras")
    for nombre, tramo in [("Ajuste", ajuste), ("Validación interna", interna), ("Evaluación", evaluacion)]:
        if tramo:
            print(nombre, tramo[0].fecha_prediccion, tramo[-1].fecha_prediccion,
                  max(m.fecha_objetivo for m in tramo), len(tramo), sep=" | ")
        else:
            print(nombre, "-", "-", "-", 0, sep=" | ")


if __name__ == "__main__":
    main()
