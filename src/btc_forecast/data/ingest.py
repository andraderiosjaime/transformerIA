import json
import argparse
from datetime import datetime, timedelta, timezone
from http.client import HTTPException
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


INTERVALOS = {"1h": timedelta(hours=1), "15m": timedelta(minutes=15)}


def consultar_velas(inicio, fin, tamano_pagina=5, *, intervalo="1h"):
    """Consulta una página de BTCUSDT Spot, por apertura en [inicio, fin)."""
    if intervalo not in INTERVALOS:
        raise ValueError("El intervalo debe ser 1h o 15m.")
    if type(tamano_pagina) is not int or not 1 <= tamano_pagina <= 1000:
        raise ValueError("El tamaño de página debe ser un entero entre 1 y 1000.")
    if inicio.utcoffset() is None or fin.utcoffset() is None:
        raise ValueError("Las fechas deben incluir su zona horaria.")
    parametros = urlencode({
        "symbol": "BTCUSDT",
        "interval": intervalo,
        "startTime": int(inicio.timestamp() * 1000),
        "endTime": int(fin.timestamp() * 1000) - 1,
        "limit": tamano_pagina,
    })
    url = f"https://data-api.binance.vision/api/v3/klines?{parametros}"
    with urlopen(url, timeout=30) as respuesta:
        return json.load(respuesta)


def descargar_rango(inicio, fin, tamano_pagina=1000, *, intervalo="1h"):
    """Descarga velas por apertura y avanza una vela completa entre páginas."""
    if intervalo not in INTERVALOS:
        raise ValueError("El intervalo debe ser 1h o 15m.")
    if inicio.utcoffset() is None or fin.utcoffset() is None or fin < inicio:
        raise ValueError("Se requiere un rango ordenado con zona horaria.")
    cursor = inicio.astimezone(timezone.utc)
    velas = []
    ultima = None
    while cursor < fin:
        try:
            pagina = consultar_velas(cursor, fin, tamano_pagina, intervalo=intervalo)
            print(f"Petición desde {cursor.isoformat()}: {len(pagina)} filas")
            for vela in pagina:
                apertura = datetime.fromtimestamp(vela[0] / 1000, tz=timezone.utc)
                if cursor <= apertura < fin and (ultima is None or apertura > ultima):
                    velas.append(vela)
                    ultima = apertura
            if ultima is None or ultima < cursor:
                raise ValueError("La página no contiene aperturas nuevas.")
        except (OSError, HTTPException, ValueError, TypeError, IndexError, KeyError, OverflowError) as error:
            progreso = ultima.isoformat() if ultima else "ninguna"
            raise RuntimeError(
                f"Descarga interrumpida. Última apertura recibida: {progreso}. "
                f"Inicio de la petición: {cursor.isoformat()}. Motivo: {error}"
            ) from error
        cursor = ultima + INTERVALOS[intervalo]
    return velas


def leer_rango_local(inicio, fin, carpeta=Path("data/raw/binance_spot/BTCUSDT/1h")):
    """Lee las columnas originales, ordenadas por apertura, y audita [inicio, fin)."""
    import pandas as pd
    from btc_forecast.data.audit import auditar

    inicio, fin = pd.Timestamp(inicio), pd.Timestamp(fin)
    if inicio.tzinfo is None or fin.tzinfo is None or fin <= inicio:
        raise ValueError("Se requiere un rango creciente con zona horaria.")
    inicio, fin = inicio.tz_convert("UTC"), fin.tz_convert("UTC")
    mes = inicio.normalize().replace(day=1)
    partes = []
    while mes < fin:
        ruta = Path(carpeta) / f"{mes:%Y-%m}.parquet"
        if not ruta.is_file():
            raise FileNotFoundError(f"Falta el mes {mes:%Y-%m}: {ruta}")
        tabla = pd.read_parquet(ruta)
        fechas = pd.to_datetime(tabla["open_time"], unit="ms", utc=True)
        partes.append(tabla.loc[fechas.ge(inicio) & fechas.lt(fin)])
        mes += pd.offsets.MonthBegin(1)
    serie = pd.concat(partes, ignore_index=True).sort_values("open_time", ignore_index=True)
    primera = inicio.ceil("h")
    if primera < fin or not serie.empty:
        problemas = auditar(serie, primera, fin.ceil("h"), pd.Timestamp.now(tz="UTC"))
        if problemas:
            raise ValueError("\n".join(f"{motivo}: {len(fechas)}; primera: {fechas[0]}"
                                       for motivo, fechas in problemas.items()))
    return serie


def demostrar_lectura_local():
    import pandas as pd
    from btc_forecast.features.indicators import MACDIncremental, RSIIncremental

    serie = leer_rango_local("2025-01-31T00:00Z", "2025-02-02T00:00Z")
    fechas = pd.to_datetime(serie.open_time, unit="ms", utc=True)
    frontera = pd.Timestamp("2025-02-01T00:00Z")
    alrededor = (fechas >= frontera - pd.Timedelta(hours=2)) & (fechas < frontera + pd.Timedelta(hours=2))
    print(serie.set_index(fechas).loc[alrededor.to_numpy(), ["open", "high", "low", "close", "volume"]].to_string())

    def actualizar(cierres, rsi, macd):
        return [(rsi.actualizar(float(c)), macd.actualizar(float(c))) for c in cierres]

    continua = actualizar(serie.close, RSIIncremental(), MACDIncremental())
    rsi, macd = RSIIncremental(), MACDIncremental()
    por_mes = actualizar(serie.close[fechas < frontera], rsi, macd)
    por_mes.extend(actualizar(serie.close[fechas >= frontera], rsi, macd))
    assert continua == por_mes
    ultimo_rsi, ultimo_macd = continua[-1]
    print(f"{len(serie)} velas; cálculos idénticos. Último RSI: {ultimo_rsi:.8f}; MACD: {ultimo_macd[2]:.8f}")


def demostrar_descarga(intervalo="1h"):
    inicio = datetime(2025, 1, 1, tzinfo=timezone.utc)
    fin = inicio + timedelta(days=1)
    try:
        velas = descargar_rango(inicio, fin, tamano_pagina=10, intervalo=intervalo)
    except RuntimeError as error:
        raise SystemExit(str(error))

    paso = INTERVALOS[intervalo]
    esperadas = [int((inicio + i * paso).timestamp() * 1000) for i in range((fin - inicio) // paso)]
    if [vela[0] for vela in velas] != esperadas:
        raise SystemExit(f"Binance no devolvió las {len(esperadas)} aperturas esperadas de {intervalo}.")
    print(f"Total verificado: {len(velas)} velas")

    formato = "{:<16}  {:>16}  {:>16}  {:>16}  {:>16}  {:>16}"
    print(formato.format("Apertura UTC", "open", "high", "low", "close", "volumen"))
    for vela in velas:
        apertura = datetime.fromtimestamp(vela[0] / 1000, tz=timezone.utc)
        print(formato.format(apertura.strftime("%Y-%m-%d %H:%M"), *vela[1:6]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Demuestra la lectura de enero y febrero.")
    parser.add_argument("--intervalo", choices=INTERVALOS, default="1h", help="Intervalo para la descarga.")
    args = parser.parse_args()
    if args.local and args.intervalo != "1h":
        parser.error("La demostración local usa el auditor horario.")
    if args.local:
        demostrar_lectura_local()
    else:
        demostrar_descarga(args.intervalo)
