"""Descarga velas horarias de BTCUSDT Spot y guarda la respuesta en Parquet."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from btc_forecast.data.audit import auditar
from btc_forecast.data.ingest import descargar_rango


COLUMNAS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume", "ignore",
]
DESTINO = Path("data/raw/binance_spot/BTCUSDT/1h/2025-01.parquet")


def fecha_utc(texto):
    try:
        fecha = datetime.fromisoformat(texto.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Usa una fecha ISO como 2025-01-01.") from error
    if fecha.tzinfo is None:
        return fecha.replace(tzinfo=timezone.utc)
    return fecha.astimezone(timezone.utc)


def guardar(tabla, destino, verificar=None):
    destino.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destino.parent, suffix=".tmp", delete=False) as archivo:
        temporal = Path(archivo.name)
    try:
        tabla.to_parquet(temporal, engine="pyarrow", index=False)
        if verificar is not None:
            verificar(pd.read_parquet(temporal))
        temporal.replace(destino)
    finally:
        temporal.unlink(missing_ok=True)


def descargar_por_meses(inicio, fin, carpeta):
    """Comprueba las aperturas de [inicio, fin) y publica cada mes por separado."""
    cursor = pd.Timestamp(inicio).ceil("h")
    limite = pd.Timestamp(fin).ceil("h")
    print("Mes | Esperadas | Recibidas | Huecos | Estado")
    while cursor < limite:
        mes = cursor.normalize().replace(day=1)
        siguiente = mes + pd.offsets.MonthBegin(1)
        hasta = min(siguiente, limite)
        esperadas = int((hasta - cursor) / pd.Timedelta(hours=1))
        destino = carpeta / f"{mes:%Y-%m}.parquet"
        desde_ms, hasta_ms = cursor.value // 1_000_000, hasta.value // 1_000_000

        def revisar(tabla):
            if list(tabla.columns) != COLUMNAS:
                raise ValueError("Esquema de columnas incorrecto.")
            if not tabla.open_time.between(mes.value // 1_000_000, siguiente.value // 1_000_000,
                                           inclusive="left").fillna(False).all():
                raise ValueError("El archivo incluye aperturas inválidas o de otro mes.")
            dentro = tabla.open_time.between(desde_ms, hasta_ms, inclusive="left")
            problemas = auditar(tabla.loc[dentro], cursor, hasta, pd.Timestamp.now(tz="UTC"))
            if problemas:
                conteos = {motivo: len(fechas) for motivo, fechas in problemas.items()}
                raise ValueError(f"{dentro.sum()} filas; auditoría: {conteos}")
            return int(dentro.sum())

        existente = None
        estado = "reutilizado"
        if destino.exists():
            try:
                existente = pd.read_parquet(destino)
                recibidas = revisar(existente)
            except (OSError, ValueError, TypeError) as error:
                print(f"{destino.name}: {error}")
                estado = "descargado"
        else:
            estado = "descargado"
        if estado == "descargado":
            nueva = pd.DataFrame(descargar_rango(cursor, hasta), columns=COLUMNAS)
            recibidas = revisar(nueva)
            # Conserva las horas del mes fuera del rango solicitado.
            if existente is not None and (cursor != mes or hasta != siguiente):
                if list(existente.columns) != COLUMNAS:
                    raise ValueError(f"No se pueden combinar esquemas distintos en {destino}.")
                fuera = ~existente.open_time.between(desde_ms, hasta_ms, inclusive="left")
                nueva = pd.concat([existente.loc[fuera], nueva], ignore_index=True).sort_values("open_time")
            guardar(nueva, destino, revisar)
        print(f"{mes:%Y-%m} | {esperadas} | {recibidas} | 0 | {estado}", flush=True)
        cursor = hasta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inicio", type=fecha_utc, default="2025-01-01", help="Inicio inclusivo (UTC).")
    parser.add_argument("--fin", type=fecha_utc, default="2025-02-01", help="Fin exclusivo (UTC).")
    parser.add_argument("--por-meses", action="store_true", help="Guarda y verifica un Parquet por mes.")
    parser.add_argument("--destino", type=Path, help="Archivo de salida, o carpeta con --por-meses.")
    args = parser.parse_args()
    if args.inicio >= args.fin:
        parser.error("--fin debe ser posterior a --inicio.")
    if args.por_meses:
        descargar_por_meses(args.inicio, args.fin, args.destino or DESTINO.parent)
        return
    destino = args.destino or DESTINO
    tabla = pd.DataFrame(descargar_rango(args.inicio, args.fin), columns=COLUMNAS)
    guardar(tabla, destino)
    guardada = pd.read_parquet(destino, engine="pyarrow")
    aperturas = pd.to_datetime(guardada["open_time"], unit="ms", utc=True)
    print(f"Ruta: {destino.resolve()}")
    print(f"Filas: {len(guardada)}")
    print(f"Columnas: {', '.join(guardada.columns)}")
    print(f"Primera apertura UTC: {aperturas.min()}")
    print(f"Última apertura UTC: {aperturas.max()}")


if __name__ == "__main__":
    main()
