# Transformer decoder-only para BTC/USDT: ¿qué tanto se puede anticipar a 45-90 minutos?

Es un Transformer decoder-only hecho en Python con PyTorch. Recibe las últimas 96 velas cerradas de BTC/USDT Spot (15 minutos cada una) y, en una sola pasada, predice el RSI a +45 min, el MACD a +60, +75 y +90 min, y una puntuación que indica qué tan probable es un primer cruce alcista del histograma entre 60 y 90 minutos.

Resumen de lo que salió en la prueba de junio de 2026: el RSI mejora 4,90 % en error absoluto medio frente a persistencia. Con el nivel del MACD no hubo mejora, ya que la referencia de cierre constante tuvo menos error en los tres plazos.


## Instalación

Lo probé en Windows con PowerShell, Python 3.14.0 y PyTorch 2.14.1+cu130. El entrenamiento de referencia corrió en una RTX 4050 . Inferencia, evaluación y pruebas también andan en CPU. Las dependencias (y los extras para tests y figuras) están en [pyproject.toml](pyproject.toml).

Desde la raíz del repositorio:

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,presentacion]"
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
~~~

## Monitor en vivo (Binance Spot)

Con los pesos guardados y los datos actuales de Binance:

~~~powershell
.\.venv\Scripts\python.exe -B scripts\monitor_en_vivo.py --abrir
~~~

Se abre `http://127.0.0.1:8765/` en el navegador con el panel "Binance Spot · En vivo". Deja PowerShell abierto mientras lo uses y para detenerlo pulsa `Ctrl+C`.

La cotización se actualiza cada 10 segundos más o menos, y el modelo emite un pronóstico cada vez que cierra una vela de 15 minutos. Por ejemplo, si son las 22:20 en Perú, el último origen es el de las 22:15: su RSI vence a las 23:00 y sus tres valores de MACD vencen a las 23:15, 23:30 y 23:45. El eje del panel llega hasta las 23:50.

Cómo leer el panel:

| Elemento | Qué muestra |
|---|---|
| RSI y MACD | Arriba, el nivel observado, el pronóstico y la referencia. Abajo, cuánto corrigió el modelo a la referencia y cuánto hacía falta corregirla. |
| Emisión y evaluación | Al elegir una emisión se marca con un rombo. Su resultado queda pendiente hasta que cierra la vela objetivo. El MAE del panel usa las emisiones registradas con resultado conocido en las últimas 48 horas, y las curvas históricas incluyen pronósticos reconstruidos. |
| Precio de BTC | Viene de Binance. El precio implícito es el que tendría que sostenerse durante todas las velas del horizonte para que el MACD saliera como se pronosticó, así que es una estimación que depende de ese supuesto. |
| Alertas | El aviso anticipado indica un posible primer cruce alcista entre 60 y 90 minutos. El de confirmación aparece cuando el histograma observado cruza cero al cierre, sea hacia arriba o hacia abajo. |

Para practicar sin depender del mercado, puedes reproducir junio de 2026 desde el Parquet local, en otro puerto. Avanza una vela cada ocho segundos y la cabecera indica que es una reproducción. Código en [scripts/monitor_en_vivo.py](scripts/monitor_en_vivo.py).

~~~powershell
.\.venv\Scripts\python.exe -B scripts\monitor_en_vivo.py --demo --puerto 8766 --abrir
~~~

## Verificar una predicción guardada

El monitor guarda cada emisión en `artifacts/demo_en_vivo/pronostico_*.parquet`. Si quieres generar una emisión suelta y comparar más tarde sus cuatro niveles con los cierres reales, corre estos comandos cuando corresponda. Por defecto `verificar` toma la última emisión guardada, pero le puedes pasar la ruta de otro Parquet. El script está en [scripts/demo_en_vivo.py](scripts/demo_en_vivo.py).

~~~powershell
.\.venv\Scripts\python.exe -B scripts\demo_en_vivo.py emitir --abrir
.\.venv\Scripts\python.exe -B scripts\demo_en_vivo.py verificar --abrir
~~~

Para revisar solo el RSI(14), el siguiente ejemplo toma la vela de origen de las 19:45 (hora de Perú) y, si hace falta, espera hasta el cierre de las 20:30. Imprime el RSI real y, si hay una emisión guardada, los errores del Transformer y de persistencia. Sin `--fecha` usa el día actual en Perú. Si lo que quieres es el RSI observado a las 19:45, cambia `--origen 19:45` por `--objetivo 19:45`. Script: [scripts/verificar_rsi.py](scripts/verificar_rsi.py).

~~~powershell
.\.venv\Scripts\python.exe -u -B scripts\verificar_rsi.py --fecha 2026-11-25 --origen 19:45
~~~

## Datos

El [descargador](src/btc_forecast/data/ingest.py) usa la API pública de Binance Spot (`https://data-api.binance.vision/api/v3/klines`) con `symbol=BTCUSDT`, `interval=15m` y hasta 1.000 velas por página. Avanza desde la última apertura recibida y guarda los meses completos en `data/raw/binance_spot/BTCUSDT/15m/AAAA-MM.parquet`. Al leer, se revisan las fechas UTC, la continuidad, que no haya duplicados y que precios y volumen tengan sentido. De enero a junio de 2026 son 17.376 velas.

Los datos de la corrida ya están en el repo. Si te faltan meses, agrega `--descargar-faltantes` al comando de entrenamiento, evaluación o inferencia. Para las columnas de las descargas nuevas se usa como plantilla el [Parquet de enero de 2026](data/raw/binance_spot/BTCUSDT/15m/2026-01.parquet), así que en una copia del proyecto sin datos hay que recuperar primero ese archivo.

## Reproducir el experimento

### Un pronóstico con los pesos guardados

El origen debe coincidir con el cierre UTC de una vela. Este ejemplo usa solo la historia disponible hasta el primer origen evaluable de junio:

~~~powershell
.\.venv\Scripts\python.exe -B scripts\clasificar_cruce_macd.py pronosticar-plazos --origen 2026-06-01T00:14:59.999Z --dispositivo cpu
~~~

Deberías ver más o menos RSI 52,68 y MACD 21,73 / 20,25 / 20,63 USDT para los cuatro plazos. La puntuación de cruce es 0,0679, por debajo del umbral de aviso (0,2782). Para usar la última vela cerrada, cambia el instante por `--origen ahora --descargar-faltantes`.

### Evaluar junio sin reentrenar

El script lee `plazos.pt` y escribe el Parquet de predicciones en `--destino`. Esta secuencia crea una carpeta aparte para no pisar la corrida publicada:

~~~powershell
$revision = Join-Path 'artifacts' ('revision_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $revision | Out-Null
Copy-Item artifacts\anticipacion_45_90m_s43\plazos.pt $revision
.\.venv\Scripts\python.exe -B scripts\clasificar_cruce_macd.py evaluar-plazos --mes 2026-06 --dispositivo cpu --destino $revision
~~~

### Entrenar desde cero en CUDA

Entrena y después evalúa esos pesos nuevos:

~~~powershell
$corrida = Join-Path 'artifacts' ('entrenamiento_' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe -B scripts\clasificar_cruce_macd.py ajustar-plazos --dispositivo cuda:0 --destino $corrida
.\.venv\Scripts\python.exe -B scripts\clasificar_cruce_macd.py evaluar-plazos --mes 2026-06 --dispositivo cuda:0 --destino $corrida
~~~

## Arquitectura

~~~text
96 velas cerradas × 11 variables (24 horas de historia)
  → escalado por columna con las filas únicas del ajuste
  → proyección lineal por vela, 11 → 64
  → 2 bloques de decoder
      LayerNorm → autoatención causal de 4 cabezas con RoPE → residual
      LayerNorm → FFN 64 → 256 → 64 con GELU → residual
  → normalización final → último token [B, 64]
  → capa lineal 64 → 5
      4 correcciones → invertir escalado → sumar referencias → niveles
      1 logit → sigmoide → puntuación de cruce (solo en orígenes elegibles)
~~~

Las ocho variables base son: log-retorno, rango relativo, volumen relativo a 20 velas, RSI de Wilder(14)/100, MACD (EMA12 menos EMA26), señal (EMA9 del MACD), histograma (estos tres últimos divididos por el cierre) y volatilidad EWMA. A eso se suman la distancia al precio crítico en unidades de volatilidad y la pendiente y aceleración del histograma. Las primeras 200 velas se enmascaran como calentamiento.

El decoder mantiene las 96 posiciones hasta leer el último token, y la atención causal deja que cada vela mire las anteriores y a sí misma. Como los cinco resultados salen del mismo origen, con una sola pila de decoder alcanza. Código: [decoder](src/btc_forecast/models/decoder.py) y [la clase RedPlazos](scripts/clasificar_cruce_macd.py).

Para el RSI, la referencia es simplemente su valor actual. Para el MACD, se actualizan las dos EMA suponiendo cierres iguales al de origen. El Transformer no predice el nivel directamente sino el error de cada referencia; en el MACD, ese error se divide además por el cierre de origen antes de estandarizar.

En la clasificación, un caso positivo es el que, partiendo de un histograma ≤ 0, tiene un primer histograma futuro positivo en la vela 4, 5 o 6. Si el primer cruce ocurre en alguna de las tres primeras velas, cuenta como negativo aunque después haya otro.

## Parámetros y protocolo temporal

| Parámetro | Valor |
|---|---|
| Ventana, dimensión y cabezas | 96 velas, dimensión 64, 4 cabezas de 16 |
| Decoder | 2 bloques, FFN de 256, dropout 0,1. El modelo completo tiene 101.189 parámetros |
| Optimización | AdamW, tasa 0,0003, lotes de 64, clipping de gradiente a 1 |
| Pérdida | Huber sobre las cuatro correcciones estandarizadas, más entropía cruzada binaria en orígenes elegibles |
| Selección | Máximo 12 épocas, semilla 43. La mejor fue la época 3; con paciencia de tres épocas el entrenamiento se detuvo en la 6 y se restauraron los pesos de la 3 |
| Regla de aviso | Puntaje estrictamente mayor al P95 de cada modelo en calibración: 0,278238 para la red y 0,260528 para la logística |

| Periodo (2026) | Para qué se usó | Ventanas |
|---|---|---:|
| Enero a febrero | Ajuste de la red, escalas y referencia logística | 5.363 |
| 1 a 15 de marzo | Selección por mínima pérdida conjunta | 1.434 |
| 16 a 31 de marzo | Fijar los umbrales de aviso | 780 elegibles |
| Junio | Prueba con modelos y reglas ya congelados | 2.874 (1.501 elegibles) |

La última etiqueta de cada ventana tiene que conocerse dentro de su propio tramo. Abril y mayo ya se habían usado en experimentos anteriores, por eso quedaron fuera. Ojo con una diferencia: el clasificador histórico de +15 minutos usaba cruce con desplazamiento, filtro de volatilidad y P99, mientras que la variante de junio usa primer cruce simple y P95 sobre orígenes elegibles. El modelo Student-t es de la etapa horaria anterior.

## Resultados de junio de 2026

### Niveles

El MAE está en puntos de RSI o en USDT para el MACD. La mejora se calcula como `100 × (1 − MAE del Transformer / MAE de la referencia)`. Los intervalos del 95 % salen de remuestrear días UTC emparejados, con 10.000 réplicas.

| Objetivo | MAE Transformer | MAE referencia | Mejora | IC 95 % de la mejora |
|---|---:|---:|---:|---:|
| RSI +45 min | 5,586 | 5,874 | +4,90 % | [+3,52; +6,19] % |
| MACD +60 min | 39,566 | 39,406 | −0,41 % | [−1,03; +0,22] % |
| MACD +75 min | 49,381 | 49,243 | −0,28 % | [−0,97; +0,41] % |
| MACD +90 min | 58,797 | 58,414 | −0,65 % | [−1,69; +0,41] % |

### Cruces

Hubo 278 orígenes positivos entre 1.501 elegibles, que corresponden a 99 cruces alcanzables. Como clasificador de referencia se usa una regresión logística con la distancia al precio crítico, el retorno, la volatilidad y la pendiente y aceleración del histograma en el origen. La AP (precisión promedio) mide qué tan bien ordenadas quedan las puntuaciones. Los avisos usan los umbrales congelados en marzo, y varios avisos correctos pueden apuntar al mismo cruce.

| Método | AP | Avisos | Aciertos | Falsos | Precisión de avisos | Cruces distintos detectados |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 0,268 | 53 | 21 | 32 | 39,6 % | 15 de 99 |
| Logística | 0,233 | 67 | 20 | 47 | 29,9 % | 8 de 99 |

Las gráficas de niveles muestran solo las primeras 192 fechas objetivo (48 horas), pero las tablas de arriba son del mes completo. 

## Pruebas, figuras y presentación

.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -B scripts\plot_anticipacion_junio.py
~~~

Los tests cubren ingestión, indicadores, umbrales, referencias, causalidad, cortes temporales y el pipeline histórico. Hay una prueba marcada como fallo esperado por el redondeo del RSI justo en el umbral de 70. El generador de figuras lee las predicciones de la corrida publicada.



## Estructura del repo

~~~text
data/raw/binance_spot/BTCUSDT/15m/   Velas mensuales
scripts/clasificar_cruce_macd.py    Datos, RedPlazos, entrenamiento e inferencia
scripts/monitor_en_vivo.py          Monitor local, cotización y alertas de cruce
scripts/demo_en_vivo.py             Emisión puntual, verificación y gráficas
scripts/verificar_rsi.py            RSI observado al cierre objetivo
scripts/plot_anticipacion_junio.py  Figuras de la evaluación de junio
src/btc_forecast/data/              Ingesta, auditoría y cortes históricos
src/btc_forecast/features/          Indicadores, variables, umbrales y etiquetas
src/btc_forecast/models/            Decoder, referencias y cabezas históricas
src/btc_forecast/training/          Entrenador de los experimentos de regresión
src/btc_forecast/evaluation/        Métricas y comparaciones
artifacts/anticipacion_45_90m_s43/  Checkpoint, predicciones y tres figuras
tests/                              Pruebas del proyecto
~~~