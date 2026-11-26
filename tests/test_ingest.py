import json
from datetime import datetime, timedelta, timezone
from io import BytesIO
from urllib.parse import parse_qs, urlsplit

from btc_forecast.data import ingest


def test_descargar_cinco_horas_en_paginas_de_dos(monkeypatch):
    inicio = datetime(2025, 1, 1, tzinfo=timezone.utc)
    fin = inicio + timedelta(hours=5)
    hora_ms = 3_600_000
    base_ms = int(inicio.timestamp() * 1000)
    # La sexta apertura coincide con el fin exclusivo.
    aperturas = [base_ms + hora * hora_ms for hora in range(6)]
    inicios_pedidos = []

    def urlopen_simulado(url, timeout):
        parametros = parse_qs(urlsplit(url).query)
        desde = int(parametros["startTime"][0])
        hasta = int(parametros["endTime"][0])
        limite = int(parametros["limit"][0])
        inicios_pedidos.append(desde)
        assert hasta == aperturas[5] - 1
        assert limite == 2
        disponibles = [t for t in aperturas if desde <= t <= hasta]
        filas = [
            [t, "2", "3", "1", "2", "4", t + hora_ms - 1, "8", 2, "1", "2", "0"]
            for t in disponibles[:limite]
        ]
        return BytesIO(json.dumps(filas).encode("utf-8"))

    monkeypatch.setattr(ingest, "urlopen", urlopen_simulado)
    velas = ingest.descargar_rango(inicio, fin, tamano_pagina=2)

    assert [vela[0] for vela in velas] == aperturas[:5]
    assert inicios_pedidos == [aperturas[0], aperturas[2], aperturas[4]]
