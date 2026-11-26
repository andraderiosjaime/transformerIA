"""Panel local de BTC/USDT. Ejecutar desde la raíz con --abrir o --demo --abrir."""
import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading
import time
from urllib.request import urlopen
import webbrowser

import numpy as np
import pandas as pd

from clasificar_cruce_macd import PLAZOS, emitir_plazos, leer_mes, preparar, resultados_plazos
from demo_en_vivo import CARPETA, MS, PASO, cargar_red, guardar_emision, ultimo_cierre
from btc_forecast.data.ingest import descargar_rango

HISTORIA = 1000
PERIODO = pd.Timedelta(hours=48)
TICKER = "https://data-api.binance.vision/api/v3/ticker/price?symbol=BTCUSDT"


def milis(t):
    # En la pantalla el cierre 19:44:59.999 se identifica como 19:45.
    return (t + MS).value // 1_000_000


def velas_cerradas(fin):
    hasta = fin + MS
    desde = hasta - HISTORIA * PASO
    filas = descargar_rango(desde.to_pydatetime(), hasta.to_pydatetime(), intervalo="15m")
    esperado = pd.date_range(desde, periods=HISTORIA, freq=PASO)
    aperturas = pd.to_datetime([v[0] for v in filas], unit="ms", utc=True)
    cierres = pd.to_datetime([v[6] for v in filas], unit="ms", utc=True)
    if not esperado.equals(aperturas) or not (esperado + PASO - MS).equals(cierres):
        raise ValueError("Binance todavía no ha publicado todas las velas cerradas.")
    d = pd.DataFrame([v[1:6] for v in filas], columns=["open","high","low","close","volume"],
                     index=cierres, dtype=float)
    if (not np.isfinite(d).all().all() or (d.low <= 0).any() or (d.volume < 0).any()
            or (d.low > d[["open","close"]].min(axis=1)).any()
            or (d.high < d[["open","close"]].max(axis=1)).any()):
        raise ValueError("OHLCV inválido en Binance.")
    return d


def precio_implicito(cierre, macd, referencia, pasos):
    """Supone un único precio para los pasos 1..h y despeja ese precio de la EMA."""
    sensibilidad = (1 - 2/27)**pasos - (1 - 2/13)**pasos
    return cierre + (macd - referencia) / sensibilidad


def panel(velas, red, cfg, demo=False, anteriores_demo=None):
    d = preparar(velas)
    origen = d.index[-1]
    desde = origen - PERIODO
    indices = np.flatnonzero(d.ventana_valida & (d.index >= desde - 6*PASO))
    if not len(indices) or indices[-1] != len(d)-1:
        raise ValueError("Ventana de indicadores incompleta.")
    g = emitir_plazos(d, indices, red, cfg, "cpu")
    campos = list(g.columns)
    g["emitido_utc"] = pd.Series(pd.NaT, index=g.index, dtype="datetime64[ns, UTC]")
    g["umbral_aviso"] = cfg["umbrales"][0]
    if demo:
        recibos = anteriores_demo
        recibo = g.iloc[[-1]].copy()
        recibo["emitido_utc"] = origen + MS
        recibos.setdefault(origen, recibo)
        for t in list(recibos):
            if t < g.index[0]:
                del recibos[t]
    else:
        recibos = {}
        for t in g.index:
            archivo = CARPETA / f"pronostico_{t:%Y%m%dT%H%M%SZ}.parquet"
            if archivo.is_file():
                recibos[t] = pd.read_parquet(archivo)
            elif t == origen and ultimo_cierre() == origen:
                recibo = g.iloc[[-1]].copy()
                recibo["emitido_utc"] = pd.Timestamp.now(tz="UTC")
                recibos[t] = guardar_emision(recibo, archivo)
    if recibos:
        guardados = pd.concat(recibos.values())
        if not guardados.index.is_unique or not guardados.index.isin(g.index).all():
            raise ValueError("Origen de emisión incompatible.")
        if not ((guardados.emitido_utc >= guardados.index)
                & (guardados.emitido_utc < guardados.fecha_rsi_45m)).all():
            raise ValueError("Hora de emisión incompatible.")
        for col in campos + ["emitido_utc"]:
            g.loc[guardados.index, col] = guardados[col]

    hist = d.loc[desde:, ["cierre","rsi","macd","histograma"]]
    niveles = [[milis(t), *valores] for t, *valores in hist.itertuples()]
    series = {}
    for indicador,k in PLAZOS:
        clave = f"{indicador}_{15*k}m"
        filas = []
        precios = []
        cols = ["fecha_"+clave,"pred_"+clave,"ref_"+clave,"emitido_utc","cierre"]
        for t, fecha, pred, ref, emitido, cierre in g[cols].itertuples():
            if fecha != t+k*PASO:
                raise ValueError(f"Fecha desalineada en {clave}.")
            if fecha < desde:
                continue
            registro = [milis(fecha), pred, ref, milis(t), pd.notna(emitido)]
            filas.append(registro)
            if indicador == "macd":
                precios.append([registro[0], precio_implicito(cierre,pred,ref,k), cierre,
                                registro[3], registro[4]])
        series[clave] = filas
        if precios:
            series[f"btc_{15*k}m"] = precios

    histograma = d.histograma
    alza = histograma.shift().le(0) & histograma.gt(0)
    baja = histograma.shift().ge(0) & histograma.lt(0)
    eventos = []
    for t in hist.index[alza.loc[hist.index] | baja.loc[hist.index]]:
        eventos.append([milis(t), "Cruce alcista confirmado" if alza[t] else "Cruce bajista confirmado"])
    for t in g.index[g.emitido_utc.notna() & g.aviso_red & (g.index >= desde)]:
        eventos.append([milis(t), "Aviso del modelo · posible primer cruce alcista a 60–90 min"])
    _, primero = resultados_plazos(d)
    etiquetas = pd.Series(primero, index=d.index).reindex(g.index)
    evaluados = g[(g.index >= desde) & g.emitido_utc.notna() & etiquetas.ge(0)]
    aciertos = evaluados.aviso_red & etiquetas.loc[evaluados.index].ge(4)
    ultimo = g.iloc[-1]
    return dict(origen=milis(origen), historia=niveles, series=series,
                eventos=sorted(eventos, reverse=True), elegible=bool(ultimo.elegible),
                puntaje=float(ultimo.p_red) if ultimo.elegible else None, umbral=cfg["umbrales"][0],
                eventosEvaluados=len(evaluados), avisos=int(evaluados.aviso_red.sum()),
                aciertos=int(aciertos.sum()))


def cotizar():
    with urlopen(TICKER, timeout=6) as respuesta:
        ticker = json.load(respuesta)
    precio = float(ticker["price"])
    if ticker.get("symbol") != "BTCUSDT" or not math.isfinite(precio) or precio <= 0:
        raise ValueError("Cotización inválida.")
    return [int(time.time()*1000),precio]


class Monitor:
    def __init__(self, demo):
        self.demo = demo
        self.estado = dict(demo=demo, panel=None, ticker=None, error="", errorTicker="")
        self.parar = threading.Event()

    def ejecutar(self):
        red, cfg = cargar_red()
        if self.demo:
            mes = leer_mes("2026-06", False)
            posicion = HISTORIA
            recibos = {}
        ultimo = None
        while not self.parar.is_set():
            estado = self.estado.copy()
            try:
                fin = mes.index[posicion] if self.demo else ultimo_cierre()
                if fin != ultimo:
                    velas = mes.iloc[posicion-HISTORIA+1:posicion+1] if self.demo else velas_cerradas(fin)
                    estado["panel"] = panel(velas, red, cfg, self.demo, recibos if self.demo else None)
                    ultimo = fin
                    if self.demo:
                        estado["ticker"] = [milis(fin),float(velas.close.iloc[-1])]
                        posicion = min(posicion+1,len(mes)-1)
                estado["error"] = ""
            except Exception as exc:
                estado["error"] = f"Velas sin actualizar: {exc}"
                print(estado["error"], flush=True)
            self.estado = estado.copy()
            if not self.demo:
                try:
                    estado["ticker"] = cotizar()
                    estado["errorTicker"] = ""
                except Exception as exc:
                    estado["errorTicker"] = f"Cotización sin actualizar: {exc}"
                self.estado = estado
            self.parar.wait(8 if self.demo else 10)

    def iniciar(self):
        try:
            self.ejecutar()
        except Exception as exc:
            self.estado = {**self.estado,"error": f"No se pudo iniciar el monitor: {exc}"}
            print(self.estado["error"], flush=True)


class Vista(BaseHTTPRequestHandler):
    def __init__(self, *args, monitor, **kwargs):
        self.monitor = monitor
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == "/":
            cuerpo, tipo = VISTA.encode(), "text/html; charset=utf-8"
        elif self.path == "/estado":
            cuerpo = json.dumps(self.monitor.estado, allow_nan=False).encode()
            tipo = "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(cuerpo)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puerto", type=int, default=8765)
    parser.add_argument("--abrir", action="store_true", help="Abrir el navegador.")
    parser.add_argument("--demo", action="store_true", help="Reproducir junio local: una vela cada 8 s.")
    args = parser.parse_args()
    monitor = Monitor(args.demo)
    with ThreadingHTTPServer(("127.0.0.1",args.puerto),partial(Vista,monitor=monitor)) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Monitor BTC/USDT: {url}  |  Ctrl+C para detener", flush=True)
        threading.Thread(target=monitor.iniciar, daemon=True).start()
        if args.abrir:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            monitor.parar.set()


VISTA = r"""<!doctype html>
<html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC/USDT · Monitor de pronósticos</title>
<style>
:root{color-scheme:light;--ink:#173944;--soft:#587078;--edge:#ccdad8;--accent:#a84422;--green:#167566;--paper:#f4f6f3}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px Inter,"Segoe UI",sans-serif;font-variant-numeric:tabular-nums}
header{background:#123842;color:white;padding:16px max(4vw,20px);display:flex;align-items:center;justify-content:space-between;gap:16px}h1{font-size:clamp(19px,3vw,30px);margin:0;letter-spacing:-.03em}
main{max-width:1540px;margin:auto;padding:22px 3vw 40px}h2{font-size:20px;letter-spacing:-.025em;margin:0}p{margin:6px 0;color:var(--soft);line-height:1.4}
#modo{color:#b9e6d7}.resumen{display:grid;grid-template-columns:1.15fr 1fr 1fr 1.5fr;background:white;border:1px solid var(--edge);border-radius:10px;padding:18px 4px}
.resumen>div{padding:0 18px;border-right:1px solid var(--edge)}.resumen>div:last-child{border:0}
small{display:block;color:var(--soft);font-size:12px}.resumen strong{display:block;margin:5px 0;font-size:clamp(17px,2vw,26px)}
.barra,.titulo,.controles,.leyenda{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.barra,.titulo{justify-content:space-between}.barra{margin:20px 0 12px}.leyenda{font-size:13px}.leyenda i{display:inline-block;width:21px;border-top:3px solid #254f65;vertical-align:middle;margin-right:4px}.leyenda .pr{border-top:dashed #ad4521}.leyenda .rf{border-top:dotted #007467}
select,button{font:inherit;color:inherit;background:white;border:1px solid #b9cac7;padding:5px 8px;border-radius:5px;cursor:pointer}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.card,section{background:white;border:1px solid var(--edge);border-radius:10px;padding:16px 18px;min-width:0}
section{margin-top:14px}.card p,section>p{font-size:13px}.chart{position:relative}canvas{width:100%;height:228px;display:block}#precio{height:205px}
output{position:absolute;pointer-events:none;display:none;background:#173944;color:white;border-radius:5px;font-size:12px;padding:7px 9px;bottom:25px;left:12%;white-space:normal}
.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px;margin-top:10px}th,td{text-align:right;padding:7px 9px;border-bottom:1px solid #e1e8e5;white-space:nowrap}td:first-child,th:first-child{text-align:left}th{color:var(--soft);font-weight:500}
#alerta{position:fixed;top:15px;right:15px;z-index:5;max-width:430px;background:#fff2d3;border:2px solid #a85c18;padding:16px;border-radius:8px;box-shadow:0 5px 25px #132e4133}#alerta button{float:right;margin-left:12px}
#problema:not(:empty){padding:12px;background:#fff0dd}.eventos{max-height:150px;overflow:auto}.eventos p{color:var(--ink)}
@media(max-width:850px){.grid{grid-template-columns:1fr}.resumen{grid-template-columns:1fr 1fr;row-gap:15px}.resumen>div:nth-child(2){border:0}main{padding:12px}.card,section{padding:12px}header{padding:14px}}
</style>
<header><h1>BTC/USDT · Monitor de pronósticos</h1><span id="modo">Conectando…</span></header>
<main>
<div class="resumen"><div><small>Bitcoin · USDT</small><strong id="cot">—</strong><small id="horaCot">Esperando datos</small></div>
<div><small>Última vela cerrada</small><strong id="ultimo">—</strong><small>Binance Spot · 15 minutos</small></div>
<div><small>Próximo cierre · Perú</small><strong id="proximo">—</strong><small id="hora">—</small></div>
<div><small>Primer cruce alcista a 60–90 min</small><strong id="aviso">—</strong><small id="umbral">Esperando el modelo</small></div></div>
<p id="problema" role="status"></p>
<div class="barra"><div class="leyenda"><span><i></i>Observado</span><span><i class="pr"></i>Pronóstico</span><span><i class="rf"></i>Referencia</span></div>
<div class="controles"><label>Historia <select id="horas"><option>6</option><option>12</option><option>24</option><option>48</option></select> h</label>
<label>MACD y precio <select id="horizonte"><option>60</option><option>75</option><option>90</option></select> min</label></div></div>
<div class="grid"><article class="card"><h2>RSI · 45 minutos</h2><p>Arriba, nivel en puntos. Abajo, cambio frente al RSI del origen.</p><div class="chart"><canvas id="rsi"></canvas><output></output></div></article>
<article class="card"><h2 id="titMacd">MACD</h2><p>Arriba, nivel en USDT. Abajo, corrección al MACD con cierre constante.</p><div class="chart"><canvas id="macd"></canvas><output></output></div></article></div>
<section><h2 id="titPrecio">Precio de BTC</h2><p>Estimación implícita: precio constante en las próximas velas que produciría el MACD previsto. La referencia mantiene el último cierre. El punto violeta indica la última cotización consultada.</p>
<div class="chart"><canvas id="precio"></canvas><output></output></div></section>
<p>Hora de Perú · El sombreado empieza en el último cierre. Cada rombo destaca el pronóstico del origen seleccionado. Las curvas anteriores incluyen reconstrucciones históricas.</p>
<section><div class="titulo"><h2>Evaluación por cierre objetivo</h2><label>Emisión <select id="origen"><option value="">Última</option></select></label></div>
<p>El error absoluto se completa cuando cierra la vela objetivo. El MAE compara únicamente emisiones registradas con un objetivo ya observado en las últimas 48 h.</p>
<div class="scroll"><table><thead><tr><th>Objetivo</th><th>Hora objetivo</th><th>Pronóstico</th><th>Referencia</th><th>Real</th><th>Error absoluto</th><th>MAE / ref.</th><th>Casos</th></tr></thead><tbody id="filas"></tbody></table></div></section>
<section><h2>Cruces del MACD</h2><p id="cuenta"></p><div class="eventos" id="eventos"></div></section>
</main><div hidden role="alert" id="alerta"><button id="cerrar">Cerrar</button><strong id="mensaje"></strong></div>
<script>
const $=id=>document.getElementById(id), minuto=60000, paso=15*minuto;
const fHora=new Intl.DateTimeFormat("es-PE",{timeZone:"America/Lima",hour:"2-digit",minute:"2-digit",hourCycle:"h23"});
const fFecha=new Intl.DateTimeFormat("es-PE",{timeZone:"America/Lima",day:"2-digit",month:"2-digit"});
const hh=t=>fHora.format(t), fecha=t=>fFecha.format(t)+" "+hh(t), fmt=v=>Number.isFinite(v)?v.toLocaleString("es-PE",{minimumFractionDigits:2,maximumFractionDigits:2}):"—";
let state={},hist=new Map(),ultimaVista=null,ultimoAviso=0,fallo="";
const cod=(tipo,h)=>tipo==="rsi"?"rsi_45m":tipo+"_"+h+"m";
const idx={rsi:2,macd:3,btc:1}, colores=["#254f65","#ad4521","#007467"];
function reloj(){return state.demo?state.panel.origen:Date.now()}
function chart(id,tipo,key,origen){
 const canvas=$(id), ctx=canvas.getContext("2d"), w=canvas.clientWidth,h=canvas.clientHeight,ratio=window.devicePixelRatio||1;
 canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);ctx.scale(ratio,ratio);
 const p=state.panel, ahora=reloj(), inicio=p.origen-Number($("horas").value)*60*minuto,fin=ahora+90*minuto;
 const iz=55,de=w-12, x=t=>iz+(t-inicio)/(fin-inicio)*(de-iz), targets=p.series[key], campo=idx[tipo];
 const real=p.historia.filter(r=>r[0]>=inicio).map(r=>[r[0],r[campo]]);
 const pts=targets.filter(r=>r[0]>=inicio), elegido=pts.find(r=>r[3]===origen);
 const alto=tipo==="btc"?h-32:h*.54;
 function panelGraf(top,bot,correccion){
   const trazos=correccion?[
    pts.filter(r=>hist.has(r[0])).map(r=>[r[0],hist.get(r[0])[campo]-r[2]]),
    pts.map(r=>[r[0],r[1]-r[2]]),[]]:
    [real,pts.map(r=>[r[0],r[1]]),pts.map(r=>[r[0],r[2]])];
   const vals=trazos.flat().map(v=>v[1]);if(correccion)vals.push(0);
   if(!correccion && tipo==="btc" && state.ticker)vals.push(state.ticker[1]);
   let lo=Math.min(...vals),hi=Math.max(...vals),pad=Math.max((hi-lo)*.12,.05);lo-=pad;hi+=pad;
   const y=v=>bot-(v-lo)/(hi-lo)*(bot-top);
   ctx.fillStyle="#f2f5ed";ctx.fillRect(x(p.origen),top,de-x(p.origen),bot-top);
   ctx.font="11px Segoe UI";ctx.textAlign="right";
   for(let j=0;j<3;j++){const v=lo+(hi-lo)*j/2;ctx.fillStyle="#50666c";ctx.fillText(v.toLocaleString("es-PE",{maximumFractionDigits:tipo==="btc"?0:1}),iz-5,y(v)+4);ctx.strokeStyle="#dce4e1";ctx.setLineDash([]);ctx.beginPath();ctx.moveTo(iz,y(v));ctx.lineTo(de,y(v));ctx.stroke()}
   if(correccion){ctx.strokeStyle="#93a39d";ctx.beginPath();ctx.moveTo(iz,y(0));ctx.lineTo(de,y(0));ctx.stroke()}
   for(const j of [2,0,1]){ctx.strokeStyle=colores[j];ctx.lineWidth=j===0?2:1.7;ctx.setLineDash([[],[6,4],[2,4]][j]);ctx.beginPath();let prev=null;
     for(const [t,v] of trazos[j]){if(prev===null||t-prev>paso)ctx.moveTo(x(t),y(v));else ctx.lineTo(x(t),y(v));prev=t}ctx.stroke()}
   ctx.lineWidth=1;ctx.setLineDash([3,3]);ctx.strokeStyle="#667c7a";ctx.beginPath();ctx.moveTo(x(p.origen),top);ctx.lineTo(x(p.origen),bot);ctx.stroke();ctx.setLineDash([]);
   if(elegido){ctx.save();ctx.translate(x(elegido[0]),y(elegido[1]-(correccion?elegido[2]:0)));ctx.rotate(Math.PI/4);ctx.fillStyle=colores[1];ctx.fillRect(-3.5,-3.5,7,7);ctx.restore()}
   if(tipo==="btc" && state.ticker && state.ticker[0]>=inicio && state.ticker[0]<=fin){ctx.fillStyle="#824ba2";ctx.beginPath();ctx.arc(x(state.ticker[0]),y(state.ticker[1]),4,0,2*Math.PI);ctx.fill()}
 }
 panelGraf(9,alto,false);if(tipo!=="btc")panelGraf(alto+17,h-31,true);
 ctx.fillStyle="#526674";ctx.font="11px Segoe UI";ctx.textAlign="center";
 const opciones=[15,30,60,120,240,360,720,1440],espacio=opciones.find(v=>v*minuto>=(fin-inicio)/Math.max(3,Math.floor((de-iz)/76)))*minuto;
 for(let t=Math.ceil(inicio/espacio)*espacio;t<fin;t+=espacio){if(de-x(t)>43)ctx.fillText(hh(t),x(t),h-9)}
 ctx.textAlign="right";ctx.fillText(hh(fin),de,h-9);
 canvas.onpointermove=e=>{const c=e.clientX-canvas.getBoundingClientRect().left;
 const t=Math.round((inicio+(c-iz)/(de-iz)*(fin-inicio))/paso)*paso,r=pts.find(row=>row[0]===t),v=hist.get(t);
 const o=canvas.nextElementSibling;o.textContent=fecha(t)+" · Real "+fmt(v?.[campo])+" · Pronóstico "+fmt(r?.[1])+" · Ref. "+fmt(r?.[2]);o.style.display="block"};
 canvas.onpointerleave=()=>{canvas.nextElementSibling.style.display="none"}
}
function dibujar(){
 const p=state.panel;if(!p)return;
 const ahora=reloj(),H=Number($("horizonte").value),quien=Number($("origen").value)||p.origen;
 $("ultimo").textContent=hh(p.origen);$("hora").textContent="Ahora "+hh(ahora)+" · Perú";
 $("proximo").textContent=hh(p.origen+paso);$("cot").textContent=fmt(state.ticker?.[1]);
 $("horaCot").textContent=state.ticker?(state.demo?"Cierre simulado · ":"Consulta spot · ")+hh(state.ticker[0]):"Esperando cotización";
 $("aviso").textContent=!p.elegible?"Origen inelegible":p.puntaje>p.umbral?"Aviso activo":"Sin aviso";
 $("umbral").textContent=p.elegible?"Puntuación "+p.puntaje.toFixed(3)+" / umbral "+p.umbral.toFixed(3):"Histograma positivo";
 $("titMacd").textContent="MACD · "+H+" minutos";$("titPrecio").textContent="Bitcoin · cierre implícito a "+H+" minutos";
 chart("rsi","rsi",cod("rsi",45),quien);chart("macd","macd",cod("macd",H),quien);chart("precio","btc",cod("btc",H),quien);
 const tbody=$("filas");tbody.replaceChildren();
 for(const [key,rows] of Object.entries(p.series)){
   const tipo=key.split("_")[0],fila=rows.find(r=>r[3]===quien);if(!fila)continue;
   let n=0,a=0,b=0;for(const r of rows){const real=hist.get(r[0])?.[idx[tipo]];if(r[4]&&Number.isFinite(real)){a+=Math.abs(real-r[1]);b+=Math.abs(real-r[2]);n++}}
   const v=hist.get(fila[0])?.[idx[tipo]],tr=tbody.insertRow();
   for(const txt of [key.replace("_"," +").toUpperCase().replace("BTC","BTC implícito"),fecha(fila[0]),fmt(fila[1]),fmt(fila[2]),Number.isFinite(v)?fmt(v):"Pendiente",fmt(Math.abs(v-fila[1])),n?fmt(a/n)+" / "+fmt(b/n):"—",n])tr.insertCell().textContent=txt;
 }
 $("cuenta").textContent="Últimas 48 h · "+p.eventosEvaluados+" orígenes elegibles registrados con resultado · "+p.aciertos+" avisos correctos de "+p.avisos+" emitidos con resultado.";
 $("eventos").replaceChildren(...p.eventos.map(([t,msg])=>{const el=document.createElement("p");el.textContent=fecha(t)+"   "+msg;return el}));
 if(!p.eventos.length)$("eventos").textContent="Sin cruces en el tramo visible.";
}
function actualizar(nuevo){
 state=nuevo;const p=state.panel;$("modo").textContent=state.demo?"Reproducción histórica · junio 2026":"Binance Spot · En vivo";
 if(!p)return;
 hist=new Map(p.historia.map(r=>[r[0],r]));
 if(ultimaVista!==p.origen){
   const select=$("origen"),anterior=select.value;select.replaceChildren(new Option("Última · "+hh(p.origen),""));
   const minOrigen=p.origen-48*60*minuto;
   for(const r of [...p.series.rsi_45m].reverse())if(r[4]&&r[3]>=minOrigen&&r[3]!==p.origen)select.add(new Option(fecha(r[3]),String(r[3])));
   if([...select.options].some(o=>o.value===anterior))select.value=anterior;
   const nuevos=p.eventos.filter(([t])=>t>(ultimaVista??p.origen-paso) && t>ultimoAviso);
   if(nuevos.length){$("mensaje").textContent=nuevos.map(([t,m])=>hh(t)+" · "+m).join(" / ");$("alerta").hidden=false;ultimoAviso=nuevos[0][0]}
   ultimaVista=p.origen;
 }
 dibujar();
}
async function consultar(){try{const r=await fetch("/estado",{cache:"no-store"});if(!r.ok)throw Error(r.status);actualizar(await r.json());fallo=""}catch(e){fallo="Sin conexión con el monitor local."}
 const retraso=state.panel&&!state.demo&&Date.now()-state.panel.origen>paso+30000?"Esperando el último cierre de Binance.":"";
 $("problema").textContent=[fallo,state.error,state.errorTicker,retraso,!state.panel&&!state.error?"Cargando velas y checkpoint…":""].filter(Boolean).join("  ");setTimeout(consultar,4000)}
for(const id of ["horas","horizonte","origen"])$(id).onchange=dibujar;
$("cerrar").onclick=()=>{$("alerta").hidden=true};window.addEventListener("resize",dibujar);consultar();
</script></html>"""


if __name__ == "__main__":
    main()

