"""Compara pronósticos a una hora sobre las mismas velas elegibles de cada evento."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from btc_forecast.evaluation.metrics import brier_eventos, precision_promedio
from btc_forecast.features.labels import EVENTOS


MODELOS = ("transformer", "frecuencia_historica", "ewma_gaussiana", "ewma_empirica")
MESES = pd.date_range("2025-04-01", periods=6, freq="MS", tz="UTC")


def leer_folds(corrida, abril=None):
    """Comprueba mes, horizonte y semilla antes de reunir los seis Parquet."""
    partes = []
    semillas = set()
    for fold, inicio in enumerate(MESES, start=1):
        ruta = corrida / f"predicciones_fold{fold:02d}_{inicio:%Y-%m}.parquet"
        datos = pd.read_parquet(ruta)
        if fold == 1 and abril is not None:
            previo = pd.read_parquet(abril)
            claves = ["fecha_prediccion", "modelo", "objetivo"]
            def comparable(tabla):
                return tabla.drop(columns="corrida").set_index(claves).sort_index()

            pd.testing.assert_frame_equal(comparable(datos), comparable(previo), check_exact=True)
            ruta, datos = abril, previo
        print(f"Fold {fold}: {ruta}")
        fin = inicio + pd.offsets.MonthBegin()
        valida = (datos.fold.eq(fold) & datos.corte.eq(inicio) & datos.mercado.eq("BTCUSDT Spot")
                  & datos.horizonte_horas.eq(1) & datos.fecha_prediccion.ge(inicio)
                  & datos.fin_etiqueta.lt(fin)
                  & (datos.fin_etiqueta - datos.fecha_prediccion).eq(pd.Timedelta(hours=1)))
        if datos.empty or not valida.fillna(False).all() or datos.semilla.isna().any():
            raise ValueError(f"Metadatos incompatibles en {ruta}.")
        semillas.update(datos.semilla.unique())
        eventos = datos.loc[datos.tipo.eq("evento") & datos.modelo.isin(MODELOS)]
        if set(eventos.evento) != set(EVENTOS) or set(eventos.modelo) != set(MODELOS):
            raise ValueError(f"Faltan modelos o eventos esperados en {ruta}.")
        partes.append(eventos)
    if len(semillas) != 1:
        raise ValueError("Los folds deben compartir la semilla.")
    return pd.concat(partes, ignore_index=True)


def alinear_horas(filas, modelos):
    """Alinea los pronósticos por origen conservando también las horas inelegibles."""
    if filas.duplicated(["fecha_prediccion", "modelo"]).any():
        raise ValueError("Hay predicciones repetidas para la misma fecha, modelo y evento.")
    metadatos = filas.groupby("fecha_prediccion")[["etiqueta", "elegible"]]
    if metadatos.nunique(dropna=False).gt(1).any().any():
        raise ValueError("Los modelos discrepan en la etiqueta o en la elegibilidad.")
    p = filas.pivot(index="fecha_prediccion", columns="modelo", values="probabilidad")
    p = p.reindex(columns=list(modelos)).sort_index()
    meta = metadatos.first().reindex(p.index)
    elegibles = meta.elegible.eq(True).fillna(False)
    validas = (p.ge(0) & p.le(1)).fillna(False).all(axis=1) & meta.etiqueta.isin([0, 1])
    return p.join(meta.etiqueta), elegibles, validas


def emparejar_evento(filas):
    """Una fila por fecha elegible con etiqueta común y cuatro probabilidades válidas."""
    horas, elegibles, validas = alinear_horas(filas, MODELOS)
    return horas.loc[elegibles & validas], int((elegibles & ~validas).sum())


def puntuar(pares, mes, evento, excluidas):
    y = pares.etiqueta
    observadas = pd.DataFrame({modelo: y for modelo in MODELOS})
    brier = brier_eventos(pares[list(MODELOS)], observadas)["Brier"]
    filas = []
    for modelo in MODELOS:
        ap = precision_promedio(pd.DataFrame({"probabilidad": pares[modelo], "etiqueta": y}))
        filas.append({"mes": mes, "evento": evento, "modelo": modelo, "elegibles": len(y),
                      "positivos": int(y.sum()), "elegibles_sin_par": excluidas,
                      "Brier": brier[modelo], "AP": ap,
                      "mejora_Brier": brier[modelo] - brier["transformer"]})
    return filas


def comparar(predicciones):
    """Calcula el total reuniendo los pares horarios, también para el ranking de AP."""
    mensual, total = [], []
    for evento in EVENTOS:
        filas = predicciones.loc[predicciones.evento.eq(evento)]
        bloques = []
        descartadas = 0
        for fold, inicio in enumerate(MESES, start=1):
            pares, excluidas = emparejar_evento(filas.loc[filas.fold.eq(fold)])
            mensual.extend(puntuar(pares, inicio.strftime("%Y-%m"), evento, excluidas))
            bloques.append(pares)
            descartadas += excluidas
        todos = pd.concat(bloques, verify_integrity=True)
        total.extend(puntuar(todos, "total", evento, descartadas))
    return pd.DataFrame(mensual), pd.DataFrame(total)


def bootstrap_macd_alcista(predicciones, longitud_bloque, repeticiones, semilla):
    """IC percentil de la diferencia media en velas elegibles, con bloques móviles por fold."""
    if longitud_bloque < 1 or repeticiones < 2:
        raise ValueError("Se necesita un bloque positivo y al menos dos repeticiones.")
    modelos = ("transformer", "ewma_empirica")
    filas = predicciones.loc[predicciones.evento.eq("macd_alcista") & predicciones.modelo.isin(modelos)]
    rng = np.random.default_rng(semilla)
    desplazamientos = np.arange(longitud_bloque)
    sumas = np.zeros(repeticiones)
    conteos = np.zeros(repeticiones, dtype=int)
    suma_observada = 0.0
    n_observado = 0
    for fold, grupo in filas.groupby("fold", sort=True):
        horas, elegibles, validas = alinear_horas(grupo, modelos)
        n = len(horas)
        if longitud_bloque > n:
            raise ValueError(f"El bloque de {longitud_bloque} horas supera la longitud del fold {fold}.")
        if not horas.index.to_series().diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
            raise ValueError(f"Hay discontinuidades horarias en el fold {fold}.")
        y = horas.etiqueta
        diferencia = ((horas.ewma_empirica - y).pow(2) - (horas.transformer - y).pow(2)).to_numpy(dtype=float)
        incluir = (elegibles & validas).to_numpy(dtype=bool)
        suma_observada += diferencia[incluir].sum()
        n_observado += int(incluir.sum())
        # Recorta el último bloque al tamaño del fold.
        numero_bloques = (n + longitud_bloque - 1) // longitud_bloque
        for i in range(repeticiones):
            inicios = rng.integers(0, n - longitud_bloque + 1, size=numero_bloques)
            indices = (inicios[:, None] + desplazamientos).ravel()[:n]
            indices_elegibles = indices[incluir[indices]]
            sumas[i] += diferencia[indices_elegibles].sum()
            conteos[i] += len(indices_elegibles)
    if not n_observado or (conteos == 0).any():
        raise ValueError("La muestra observada o alguna réplica carece de pares elegibles.")
    inferior, superior = np.quantile(sumas / conteos, [0.025, 0.975])
    return {"bloque_horas": longitud_bloque, "repeticiones": repeticiones, "semilla": semilla,
            "elegibles": n_observado, "mejora_Brier": suma_observada / n_observado,
            "IC95_inferior": inferior, "IC95_superior": superior}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrida", required=True, type=Path, help="Carpeta de la corrida de seis folds.")
    parser.add_argument("--abril", type=Path, help="Parquet previo de abril para verificar y reutilizar.")
    parser.add_argument("--bloques", type=int, nargs="+", metavar="HORAS",
                        help="Calcula el bootstrap para cada longitud indicada (por ejemplo, 24 72).")
    parser.add_argument("--repeticiones", type=int, default=5000, help="Réplicas por longitud de bloque.")
    parser.add_argument("--semilla", type=int, default=42, help="Semilla del remuestreo.")
    args = parser.parse_args()
    predicciones = leer_folds(args.corrida, args.abril)
    if args.bloques:
        resultados = [bootstrap_macd_alcista(predicciones, horas, args.repeticiones, args.semilla)
                      for horas in args.bloques]
        print("MACD alcista | Brier EWMA empírica menos Brier Transformer")
        print(pd.DataFrame(resultados).to_string(index=False, float_format=lambda x: f"{x:.8f}"))
        return
    mensual, total = comparar(predicciones)
    for nombre, tabla in [("mensual", mensual), ("total", total)]:
        ruta = args.corrida / f"eventos_elegibles_{nombre}.csv"
        tabla.to_csv(ruta, index=False, na_rep="indefinida")
        print(f"{len(tabla)} filas en {ruta}")
    print(total.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
