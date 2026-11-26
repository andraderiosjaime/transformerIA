"""Muestras horarias de BTCUSDT y objetivos directos de indicadores a una o seis horas."""

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from math import isclose, isfinite, log
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from btc_forecast.features.indicators import MACDIncremental, RSIIncremental
from btc_forecast.features.labels import EVENTOS, etiquetar_velas
from btc_forecast.features.transforms import HORA, calcular_variables


class ObjetivosDirectos(NamedTuple):
    delta_rsi: float
    delta_macd_sobre_cierre: float


@dataclass
class Muestra:
    X: pd.DataFrame
    y: float
    etiquetas: dict[str, int]
    fecha_prediccion: pd.Timestamp
    fecha_objetivo: pd.Timestamp
    cierre_t: float
    cierre_t1: float
    objetivos_directos: ObjetivosDirectos | tuple[ObjetivosDirectos, ...] | None = None

    @property
    def fin_etiqueta(self) -> pd.Timestamp:
        """Cierre del último horizonte, también utilizado por splits como fecha_objetivo."""
        return self.fecha_objetivo


def construir_muestras(tabla, longitud=96, calentamiento=200, *, regresion_directa=False,
                       horizontes=1) -> Iterator[Muestra]:
    """Ventanas hasta t con cierres UTC. El retorno y las etiquetas son a una hora.

    En regresión directa se añade (ΔRSI, ΔMACD/cierre[t]): un par con H=1
    o seis pares con H=6, todos respecto a t.
    """
    if type(longitud) is not int or longitud < 1:
        raise ValueError("La longitud de la ventana debe ser un entero positivo.")
    if type(horizontes) is not int or horizontes not in (1, 6):
        raise ValueError("El número de horizontes debe ser 1 o 6.")
    if horizontes != 1 and not regresion_directa:
        raise ValueError("Seis horizontes requieren el modo de regresión directa.")
    velas = tabla.sort_values("open_time")
    aperturas = pd.to_datetime(velas["open_time"], unit="ms", utc=True)
    fechas = pd.to_datetime(velas["close_time"], unit="ms", utc=True)
    if not fechas.eq(aperturas + HORA - pd.Timedelta(milliseconds=1)).all():
        raise ValueError("Las fechas de cierre deben corresponder a velas Binance de una hora.")
    velas = velas.loc[fechas < pd.Timestamp.now(tz="UTC")]
    variables = calcular_variables(velas, calentamiento=calentamiento)
    etiquetas = etiquetar_velas(velas, calentamiento=calentamiento).loc[:, list(EVENTOS)]
    cierres = velas["close"].astype(float).to_numpy()
    if regresion_directa:
        # Desnormaliza el MACD con el cierre de la misma fila.
        rsi = variables["rsi_100"].to_numpy() * 100
        macd = variables["macd_cierre"].to_numpy() * cierres
    # Las variables están disponibles al cierre, no en la apertura.
    variables.index = pd.DatetimeIndex(
        variables.index + HORA - pd.Timedelta(milliseconds=1), name="cierre_utc"
    )
    fechas = variables.index
    paso_horario = fechas.to_series().diff().eq(HORA).to_numpy()
    completas = variables.notna().all(axis=1).to_numpy()
    observadas = etiquetas.notna().all(axis=1).to_numpy()
    consecutivas = 0
    for i in range(len(fechas) - horizontes):
        if not paso_horario[i]:
            consecutivas = 0
        consecutivas = consecutivas + 1 if completas[i] else 0
        fin = i + horizontes
        if consecutivas < longitud or not observadas[i] or not paso_horario[i + 1:fin + 1].all():
            continue
        cierre_t, cierre_t1 = cierres[i], cierres[i + 1]
        directos = None
        if regresion_directa:
            pares = tuple(ObjetivosDirectos(rsi[j] - rsi[i], (macd[j] - macd[i]) / cierre_t)
                          for j in range(i + 1, fin + 1))
            if not all(isfinite(valor) for par in pares for valor in par):
                continue
            directos = pares[0] if horizontes == 1 else pares
        yield Muestra(
            X=variables.iloc[i - longitud + 1:i + 1].copy(),
            y=log(cierre_t1 / cierre_t),
            etiquetas={evento: int(valor) for evento, valor in etiquetas.iloc[i].items()},
            fecha_prediccion=fechas[i],
            fecha_objetivo=fechas[fin],
            cierre_t=cierre_t,
            cierre_t1=cierre_t1,
            objetivos_directos=directos,
        )


def mostrar_regresion_directa(velas, muestra):
    """Verifica la reconstrucción contra los indicadores actualizados con cierres reales."""
    fechas = pd.to_datetime(velas["close_time"], unit="ms", utc=True)
    cierres = velas.set_index(fechas)["close"].sort_index()
    anterior = None
    for fecha, precio in cierres.loc[:muestra.fecha_prediccion].items():
        if anterior is None or fecha - anterior != HORA:
            rsi, macd = RSIIncremental(), MACDIncremental()
        anterior = fecha
        rsi_t = rsi.actualizar(float(precio))
        valores_macd = macd.actualizar(float(precio))
    macd_t = valores_macd[2]
    pares = muestra.objetivos_directos
    if isinstance(pares, ObjetivosDirectos):
        pares = (pares,)
    filas = []
    origen = muestra.fecha_prediccion
    futuros = cierres.loc[origen + HORA:muestra.fin_etiqueta]
    for (fecha, precio), delta in zip(futuros.items(), pares, strict=True):
        rsi_real = rsi.actualizar(float(precio))
        macd_real = macd.actualizar(float(precio))[2]
        rsi_rec = rsi_t + delta.delta_rsi
        macd_rec = macd_t + muestra.cierre_t * delta.delta_macd_sobre_cierre
        if not (isclose(rsi_rec, rsi_real, rel_tol=1e-12, abs_tol=1e-12) and
                isclose(macd_rec, macd_real, rel_tol=1e-12, abs_tol=1e-12)):
            raise AssertionError(f"La reconstrucción difiere de los indicadores observados en {fecha}.")
        filas.append((origen, fecha, *delta, rsi_rec, macd_rec))
    print(f"Cierre de origen {muestra.cierre_t:.2f} USDT | RSI {rsi_t:.8f} | MACD {macd_t:.8f}")
    print(pd.DataFrame(filas, columns=["Origen UTC", "Objetivo UTC", "Delta RSI", "Delta MACD relativo",
                                      "RSI reconstruido", "MACD reconstruido"]).to_string(
        index=False, float_format=lambda v: f"{v:.10f}"))
    print(f"Reconstrucción verificada hasta fin_etiqueta {muestra.fin_etiqueta}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--longitud", type=int, default=96)
    parser.add_argument("--horizontes", type=int, choices=(1, 6), default=1)
    parser.add_argument("--regresion-directa", action="store_true", help="Muestra y verifica los dos objetivos directos.")
    args = parser.parse_args()
    ruta = Path("data/raw/binance_spot/BTCUSDT/1h/2025-01.parquet")
    velas = pd.read_parquet(ruta)
    muestras = construir_muestras(velas, longitud=args.longitud, regresion_directa=args.regresion_directa,
                                  horizontes=args.horizontes)
    primera = next(muestras, None)
    if primera is None:
        raise SystemExit("No hay ventanas completas con esa longitud y un objetivo observado.")
    if args.regresion_directa:
        mostrar_regresion_directa(velas, primera)
        if args.horizontes == 6:
            una_hora = next(m for m in construir_muestras(velas, longitud=args.longitud, regresion_directa=True)
                            if m.fecha_prediccion == primera.fecha_prediccion)
            assert primera.objetivos_directos[0] == una_hora.objetivos_directos
            print("El primer par coincide exactamente con el objetivo del modo de una hora.")
        return
    print(f"X ({args.longitud} filas, fechas de cierre UTC)")
    print(primera.X.to_string(float_format=lambda v: f"{v:.8f}"))
    print(f"Predicción: {primera.fecha_prediccion} | Cierre t: {primera.cierre_t:.2f}")
    print(f"Objetivo conocido: {primera.fecha_objetivo} | Cierre t+1: {primera.cierre_t1:.2f}")
    print(f"y = ln({primera.cierre_t1:.2f} / {primera.cierre_t:.2f}) = {primera.y:.10f}")
    print("Etiquetas:", primera.etiquetas)


if __name__ == "__main__":
    main()
