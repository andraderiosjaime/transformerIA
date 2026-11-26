"""Compara el RSI(14) observado con la emisión guardada 45 minutos antes."""
import argparse
import time
from datetime import date

import pandas as pd

from demo_en_vivo import CARPETA, MS, PASO, historia_hasta, hora_peru


HORIZONTE = 3 * PASO


def verificar(origen, esperar=True):
    objetivo = origen + HORIZONTE
    ruta = CARPETA / f"pronostico_{origen:%Y%m%dT%H%M%SZ}.parquet"
    print(f"BTC/USDT Spot | RSI(14) | velas de 15 minutos | hora de Perú\n"
          f"Origen: {hora_peru(origen + MS)} | Objetivo: {hora_peru(objetivo + MS)}")
    emision = None
    if ruta.is_file():
        tabla = pd.read_parquet(ruta)
        if len(tabla) != 1 or tabla.index[0] != origen:
            raise ValueError("La emisión guardada no corresponde al origen solicitado.")
        emision = tabla.iloc[0]
        if emision.fecha_rsi_45m != objetivo or not origen <= emision.emitido_utc < objetivo:
            raise ValueError("La fecha objetivo o la hora de emisión no permiten verificar este pronóstico.")
        print(f"Emitido a las {hora_peru(emision.emitido_utc)}. RSI del origen: {emision.rsi:.2f}. "
              f"Pronóstico: {emision.pred_rsi_45m:.2f}.")
    else:
        print("Sin pronóstico guardado para este origen. Se mostrará el RSI observado.")

    disponible = objetivo + MS
    if pd.Timestamp.now(tz="UTC") < disponible:
        print("RSI real pendiente del cierre objetivo.", flush=True)
        if not esperar:
            return
        while (restante := (disponible - pd.Timestamp.now(tz="UTC")).total_seconds()) > 0:
            time.sleep(min(restante, 30))
        time.sleep(2)  # Dar tiempo a que Binance publique la última vela cerrada.

    for intento in range(3):
        try:
            datos = historia_hasta(objetivo, descargar=True)
            break
        except RuntimeError:
            if intento == 2:
                raise
            time.sleep(3)
    real = float(datos.at[objetivo, "rsi"])
    if pd.isna(real):
        raise ValueError("Historial insuficiente para obtener el RSI(14).")
    print(f"Cierre BTC/USDT: {datos.at[objetivo, 'cierre']:.2f} USDT | RSI(14) real: {real:.4f}")
    if emision is not None:
        print(f"Error absoluto del Transformer: {abs(real-emision.pred_rsi_45m):.4f} puntos\n"
              f"Error absoluto de persistencia: {abs(real-emision.ref_rsi_45m):.4f} puntos")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fecha", type=date.fromisoformat, help="Fecha en Perú (AAAA-MM-DD), hoy por defecto.")
    horas = parser.add_mutually_exclusive_group()
    horas.add_argument("--origen", metavar="HH:MM", help="Hora de la última vela de entrada (19:45 por defecto).")
    horas.add_argument("--objetivo", metavar="HH:MM", help="Hora del RSI que quieres observar.")
    parser.add_argument("--sin-espera", action="store_true", help="Salir si el objetivo sigue pendiente.")
    args = parser.parse_args()
    dia = args.fecha or pd.Timestamp.now(tz="America/Lima").date()
    hora = args.objetivo or args.origen or "19:45"
    try:
        instante = pd.Timestamp(f"{dia} {hora}", tz="America/Lima").tz_convert("UTC")
    except ValueError:
        parser.error("Usa una hora local de Perú en formato HH:MM.")
    if instante != instante.floor(PASO):
        parser.error("Elige un cierre a los minutos 00, 15, 30 o 45.")
    origen = instante - MS - (HORIZONTE if args.objetivo else pd.Timedelta(0))
    verificar(origen, esperar=not args.sin_espera)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(str(error)) from None
