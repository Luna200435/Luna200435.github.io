#!/usr/bin/env python3
"""
Buscador de Tipos de Algoritmos de Búsqueda (aplicación web local)
===================================================================
Aplicación en Python que usa el modelo entrenado en Machine Learning for Kids
(proyecto de texto "Tipos de Algoritmos de Busqueda en Listas") para decir a
qué tipo pertenece un algoritmo de búsqueda a partir de su nombre o de una
descripción.

Tipos que reconoce (etiquetas del proyecto en ML for Kids):
  busqueda_secuencial            lineal, con centinela
  busqueda_binaria               binaria, ternaria, bisección
  busqueda_hash                  tablas hash / diccionarios
  busqueda_saltos_interpolacion  saltos, interpolación, exponencial, Fibonacci
  otro_tema                      todo lo que no es búsqueda en listas

Además de clasificar, la aplicación explica el tipo (complejidad, si necesita
datos ordenados) y SIMULA paso a paso cómo ese algoritmo encontraría un número.

Uso:
  python app_buscador.py                       -> abre la interfaz en el navegador
  python app_buscador.py --texto "divide a la mitad la lista ordenada"
  python app_buscador.py --pruebas             -> frases de prueba en consola
  python app_buscador.py --local               -> fuerza el modelo local
  python app_buscador.py --puerto 8080

La clave API de ML for Kids se toma de: --key, la variable ML4K_API_KEY o
config.json. Sin clave (o sin internet) se usa un modelo local de respaldo
entrenado con los mismos datos de datos_entrenamiento/.
Solo requiere Python 3.8+ (sin librerías externas).
"""
import argparse
import json
import math
import os
import re
import sys
import threading
import unicodedata
import urllib.parse
import urllib.request
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CARPETA = os.path.dirname(os.path.abspath(__file__))
CARPETA_DATOS = os.path.join(CARPETA, "datos_entrenamiento")
URL_API = "https://machinelearningforkids.co.uk/api/scratch/{key}/classify?data={data}"
UMBRAL_CONFIANZA = 45.0  # por debajo de esto la app dice "no estoy seguro"

# ----------------------------------------------------------------------
# Información de cada tipo
# ----------------------------------------------------------------------
INFO = {
    "busqueda_secuencial": {
        "nombre": "Búsqueda secuencial (lineal)",
        "resumen": "Revisa los elementos uno por uno, de principio a fin, hasta "
                   "encontrar el valor buscado o agotar la lista.",
        "complejidad": "O(n)",
        "ordenado": "No necesita datos ordenados",
        "ejemplos": "Búsqueda lineal, búsqueda con centinela",
    },
    "busqueda_binaria": {
        "nombre": "Búsqueda binaria",
        "resumen": "Compara con el elemento central de una lista ordenada y descarta "
                   "la mitad donde el dato no puede estar; repite con la otra mitad.",
        "complejidad": "O(log n)",
        "ordenado": "Necesita datos ordenados",
        "ejemplos": "Búsqueda binaria, ternaria, por bisección",
    },
    "busqueda_hash": {
        "nombre": "Búsqueda por hash (tabla de dispersión)",
        "resumen": "Aplica una función hash a la clave para calcular directamente la "
                   "casilla donde está el dato, sin recorrer la colección.",
        "complejidad": "O(1) promedio, O(n) en el peor caso",
        "ordenado": "No necesita datos ordenados",
        "ejemplos": "Tabla hash, diccionario (dict), HashMap, set",
    },
    "busqueda_saltos_interpolacion": {
        "nombre": "Búsqueda por saltos / interpolación",
        "resumen": "Sobre una lista ordenada, en vez de partir por la mitad, salta "
                   "por bloques o estima la posición probable según el valor buscado.",
        "complejidad": "O(√n) saltos · O(log log n) interpolación (datos uniformes)",
        "ordenado": "Necesita datos ordenados",
        "ejemplos": "Jump search, interpolación, exponencial, Fibonacci",
    },
    "otro_tema": {
        "nombre": "Otro tema (no es búsqueda en listas)",
        "resumen": "El texto no describe un algoritmo de búsqueda en listas o tablas: "
                   "puede ser ordenamiento, inteligencia artificial u otro asunto.",
        "complejidad": "—",
        "ordenado": "—",
        "ejemplos": "Ordenamiento, minimax, redes neuronales, otros temas",
    },
}

# ----------------------------------------------------------------------
# Simulaciones paso a paso (lista de ejemplo y valor buscado)
# ----------------------------------------------------------------------
LISTA = [4, 9, 15, 23, 31, 38, 46, 52, 67, 80]
OBJETIVO = 46


def sim_secuencial(a, x):
    pasos = []
    for i, v in enumerate(a):
        ok = v == x
        pasos.append(f"Posición {i}: {v} {'==' if ok else '!='} {x}" + ("  ✔ encontrado" if ok else ""))
        if ok:
            return pasos
    pasos.append("No se encontró")
    return pasos


def sim_binaria(a, x):
    pasos, ini, fin = [], 0, len(a) - 1
    while ini <= fin:
        med = (ini + fin) // 2
        pasos.append(f"inicio={ini}, fin={fin}, medio={med} → a[{med}]={a[med]}")
        if a[med] == x:
            pasos.append(f"{a[med]} == {x}  ✔ encontrado en la posición {med}")
            return pasos
        if a[med] < x:
            pasos.append(f"{a[med]} < {x}: descarta la mitad izquierda")
            ini = med + 1
        else:
            pasos.append(f"{a[med]} > {x}: descarta la mitad derecha")
            fin = med - 1
    pasos.append("No se encontró")
    return pasos


def sim_hash(a, x, tam=7):
    tabla = [[] for _ in range(tam)]
    for v in a:
        tabla[v % tam].append(v)
    pasos = [f"Tabla de {tam} cubos, función hash: h(k) = k % {tam}"]
    for i, cubo in enumerate(tabla):
        pasos.append(f"  cubo {i}: {cubo if cubo else '[]'}")
    h = x % tam
    pasos.append(f"Buscar {x}: h({x}) = {x} % {tam} = {h} → solo se revisa el cubo {h}")
    if x in tabla[h]:
        pasos.append(f"{x} está en el cubo {h}  ✔ encontrado (sin recorrer los demás)")
    else:
        pasos.append("No se encontró")
    return pasos


def sim_saltos(a, x):
    n = len(a)
    paso = int(math.sqrt(n))
    pasos = [f"n={n}, tamaño del salto = √n ≈ {paso}"]
    prev, act = 0, paso
    while act < n and a[act - 1] < x:
        pasos.append(f"Salta hasta a[{act - 1}]={a[act - 1]} < {x}: sigue saltando")
        prev, act = act, act + paso
    pasos.append(f"El bloque [{prev}..{min(act, n) - 1}] puede contener el {x}: búsqueda lineal ahí")
    for i in range(prev, min(act, n)):
        if a[i] == x:
            pasos.append(f"a[{i}]={a[i]} == {x}  ✔ encontrado")
            return pasos
        pasos.append(f"a[{i}]={a[i]} != {x}")
    pasos.append("No se encontró")
    return pasos


def sim_interpolacion(a, x):
    pasos, lo, hi = ["Variante: interpolación (estima la posición)"], 0, len(a) - 1
    while lo <= hi and a[lo] <= x <= a[hi]:
        if a[hi] == a[lo]:
            pos = lo
        else:
            pos = lo + (x - a[lo]) * (hi - lo) // (a[hi] - a[lo])
        pasos.append(f"pos = {lo} + ({x}-{a[lo]})*({hi}-{lo})/({a[hi]}-{a[lo]}) = {pos} → a[{pos}]={a[pos]}")
        if a[pos] == x:
            pasos.append(f"{a[pos]} == {x}  ✔ encontrado")
            return pasos
        if a[pos] < x:
            lo = pos + 1
        else:
            hi = pos - 1
    pasos.append("No se encontró")
    return pasos


def simular(etiqueta):
    if etiqueta == "busqueda_secuencial":
        return sim_secuencial(LISTA, OBJETIVO)
    if etiqueta == "busqueda_binaria":
        return sim_binaria(LISTA, OBJETIVO)
    if etiqueta == "busqueda_hash":
        return sim_hash(LISTA, OBJETIVO)
    if etiqueta == "busqueda_saltos_interpolacion":
        return sim_saltos(LISTA, OBJETIVO) + [""] + sim_interpolacion(LISTA, OBJETIVO)
    return []


# ----------------------------------------------------------------------
# Modelo en línea (Machine Learning for Kids)
# ----------------------------------------------------------------------
def cargar_clave(arg_key=None):
    if arg_key:
        return arg_key.strip()
    if os.environ.get("ML4K_API_KEY"):
        return os.environ["ML4K_API_KEY"].strip()
    ruta = os.path.join(CARPETA, "config.json")
    if os.path.exists(ruta):
        try:
            with open(ruta, encoding="utf-8") as f:
                clave = json.load(f).get("api_key", "").strip()
            if clave and not clave.startswith("PEGA_AQUI"):
                return clave
        except (OSError, ValueError):
            pass
    return None


def clasificar_mlkids(texto, clave):
    """Devuelve [(etiqueta, confianza_0_a_100), ...] ordenado de mayor a menor."""
    url = URL_API.format(key=urllib.parse.quote(clave), data=urllib.parse.quote(texto))
    with urllib.request.urlopen(url, timeout=20) as r:
        resultado = json.loads(r.read().decode("utf-8"))
    if not resultado:
        raise RuntimeError("respuesta vacía: ¿ya entrenaste el modelo en ML for Kids?")
    lista = [(x["class_name"], float(x["confidence"])) for x in resultado]
    return sorted(lista, key=lambda t: -t[1])


# ----------------------------------------------------------------------
# Modelo local de respaldo: TF-IDF (palabras + trigramas de letras) y
# vecinos más cercanos por similitud coseno. Sin librerías externas.
# ----------------------------------------------------------------------
def _limpiar(texto):
    texto = texto.lower().replace("√", " raiz ").replace("*", " por ")
    texto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def caracteristicas(texto):
    palabras = re.findall(r"[a-z0-9]+", _limpiar(texto))
    feats = list(palabras)
    feats += [a + "_" + b for a, b in zip(palabras, palabras[1:])]
    for p in palabras:
        marco = f"^{p}$"
        feats += ["#" + marco[i:i + 3] for i in range(len(marco) - 2)]
    return feats


class ModeloLocal:
    def __init__(self, carpeta=CARPETA_DATOS, k=3, temperatura=14.0):
        self.k, self.temp = k, temperatura
        textos, etiquetas = [], []
        for nombre in sorted(os.listdir(carpeta)):
            if not nombre.endswith(".txt"):
                continue
            with open(os.path.join(carpeta, nombre), encoding="utf-8") as f:
                for linea in f:
                    if linea.strip():
                        textos.append(linea.strip())
                        etiquetas.append(nombre[:-4])
        if not textos:
            raise RuntimeError("No se encontraron datos en " + carpeta)
        self.etiquetas = etiquetas
        self.clases = sorted(set(etiquetas))
        docs = [Counter(caracteristicas(t)) for t in textos]
        df = Counter(f for d in docs for f in d)
        n = len(docs)
        self.idf = {f: math.log((1 + n) / (1 + c)) + 1 for f, c in df.items()}
        self.vectores = [self._vector(d) for d in docs]

    def _vector(self, conteo):
        v = {f: c * self.idf[f] for f, c in conteo.items() if f in self.idf}
        norma = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {f: x / norma for f, x in v.items()}

    def clasificar(self, texto):
        q = self._vector(Counter(caracteristicas(texto)))
        sims = {c: [] for c in self.clases}
        for vec, et in zip(self.vectores, self.etiquetas):
            sims[et].append(sum(w * vec.get(f, 0.0) for f, w in q.items()))
        puntaje = {}
        for c, lista in sims.items():
            mejores = sorted(lista, reverse=True)[: self.k]
            puntaje[c] = sum(mejores) / len(mejores)
        m = max(puntaje.values())
        exp = {c: math.exp(self.temp * (p - m)) for c, p in puntaje.items()}
        s = sum(exp.values())
        return sorted(((c, 100 * e / s) for c, e in exp.items()), key=lambda t: -t[1])


class Clasificador:
    def __init__(self, clave=None, forzar_local=False):
        self.clave = None if forzar_local else clave
        self._local = None
        self._lock = threading.Lock()

    def local(self):
        with self._lock:
            if self._local is None:
                self._local = ModeloLocal()
            return self._local

    def clasificar(self, texto):
        aviso, fuente, ranking = None, "Modelo local de respaldo", None
        if self.clave:
            try:
                ranking = clasificar_mlkids(texto, self.clave)
                fuente = "Machine Learning for Kids"
            except Exception as e:  # sin internet, clave inválida, modelo sin entrenar...
                aviso = f"No se pudo usar ML for Kids ({e}). Se usó el modelo local."
        if ranking is None:
            ranking = self.local().clasificar(texto)
        etiqueta, conf = ranking[0]
        seguro = conf >= UMBRAL_CONFIANZA
        info = INFO.get(etiqueta, {})
        return {
            "texto": texto,
            "etiqueta": etiqueta,
            "confianza": round(conf, 1),
            "seguro": seguro,
            "fuente": fuente,
            "aviso": aviso,
            "info": info,
            "ranking": [{"etiqueta": e, "nombre": INFO.get(e, {}).get("nombre", e),
                         "confianza": round(c, 1)} for e, c in ranking],
            "simulacion": simular(etiqueta) if seguro else [],
            "lista": LISTA,
            "objetivo": OBJETIVO,
        }


FRASES_PRUEBA = [
    "Va revisando cada elemento de la lista hasta dar con el que busca",
    "Corta la lista ordenada en dos y se queda con la mitad correcta",
    "Calcula una fórmula con la clave y va directo a la casilla donde está",
    "Avanza de bloque en bloque y luego revisa dentro del bloque",
    "Estima dónde debe estar el valor según qué tan grande es",
    "Colisiones y factor de carga",
    "Ordena los números de menor a mayor intercambiando vecinos",
    "Hoy hace mucho calor en la ciudad",
]

# ----------------------------------------------------------------------
# Interfaz web
# ----------------------------------------------------------------------
PAGINA = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Buscador de Tipos de Algoritmos de Búsqueda</title>
<style>
:root{--bg:#f4f1ea;--card:#fffdf8;--ink:#1f2328;--mut:#6b6f76;--line:#e3ddd0;--acc:#0f766e;--acc2:#b45309;--bar:#e8e2d4;--code:#1f2937;--codeink:#d1fae5}
@media (prefers-color-scheme:dark){:root{--bg:#14171a;--card:#1c2024;--ink:#ece8df;--mut:#9aa0a6;--line:#2c3237;--acc:#2dd4bf;--acc2:#fbbf24;--bar:#2a3036;--code:#0b0e10;--codeink:#a7f3d0}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:860px;margin:0 auto;padding:28px 16px 56px}
h1{font:700 1.7rem/1.2 Georgia,"Times New Roman",serif;margin:0 0 4px}
.sub{color:var(--mut);margin:0 0 22px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin-bottom:16px}
textarea{width:100%;min-height:84px;resize:vertical;border:1px solid var(--line);border-radius:10px;padding:12px;font:inherit;background:var(--bg);color:var(--ink)}
textarea:focus{outline:2px solid var(--acc);outline-offset:1px}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:12px}
button{font:inherit;cursor:pointer;border-radius:10px;border:1px solid var(--line);padding:9px 14px;background:var(--card);color:var(--ink)}
button.main{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}
@media (prefers-color-scheme:dark){button.main{color:#06201c}}
button:hover{filter:brightness(1.07)}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
.chip{font-size:.85rem;padding:5px 10px;border-radius:999px;color:var(--mut)}
.tipo{font:700 1.35rem/1.25 Georgia,serif;margin:2px 0 4px}
.badge{display:inline-block;font-size:.78rem;padding:2px 9px;border-radius:999px;background:var(--bar);color:var(--mut);margin-right:6px}
.dudoso{color:var(--acc2)}
.bars{margin-top:14px}
.b{display:grid;grid-template-columns:minmax(120px,230px) 1fr 52px;gap:10px;align-items:center;margin:6px 0;font-size:.9rem}
.track{height:10px;background:var(--bar);border-radius:6px;overflow:hidden}
.fill{height:100%;background:var(--acc);border-radius:6px;transition:width .5s}
.b.top .fill{background:var(--acc2)}
.b span:last-child{text-align:right;color:var(--mut);font-variant-numeric:tabular-nums}
pre{background:var(--code);color:var(--codeink);border-radius:10px;padding:14px;overflow:auto;font:.85rem/1.55 ui-monospace,Menlo,Consolas,monospace;margin:10px 0 0}
.mut{color:var(--mut);font-size:.9rem}
.aviso{border-left:3px solid var(--acc2);padding:6px 12px;margin-bottom:12px;color:var(--mut);font-size:.9rem}
.hide{display:none}
footer{color:var(--mut);font-size:.82rem;margin-top:22px}
@media (max-width:520px){.b{grid-template-columns:1fr 1fr 48px}}
</style>
</head>
<body>
<main>
  <h1>¿Qué tipo de algoritmo de búsqueda es?</h1>
  <p class="sub">Escribe el nombre o la descripción de un algoritmo. El modelo entrenado en Machine Learning for Kids dirá a qué tipo pertenece.</p>

  <section class="card">
    <textarea id="txt" placeholder="Ej.: Parte la lista ordenada por la mitad y compara con el elemento del centro" autofocus></textarea>
    <div class="row">
      <button class="main" id="go">Identificar tipo</button>
      <button id="clear">Limpiar</button>
      <span class="mut" id="modo"></span>
    </div>
    <div class="chips" id="chips"></div>
  </section>

  <section class="card hide" id="res">
    <div id="aviso" class="aviso hide"></div>
    <div class="mut" id="eltexto"></div>
    <div class="tipo" id="tipo"></div>
    <div><span class="badge" id="conf"></span><span class="badge" id="fuente"></span></div>
    <p id="resumen" style="margin:12px 0 4px"></p>
    <div class="mut" id="datos"></div>
    <div class="bars" id="bars"></div>
  </section>

  <section class="card hide" id="simcard">
    <strong>Así buscaría el número <span id="obj"></span> en esta lista:</strong>
    <div class="mut" id="lista"></div>
    <pre id="sim"></pre>
  </section>

  <footer>Proyecto individual · Bases de Datos en la Nube / Machine Learning for Kids · Uriel Vazquez Luna · Universidad Tecnológica de Puebla</footer>
</main>
<script>
const EJ=["Linear search","Divide la lista ordenada a la mitad en cada paso","Calcula hash(clave) % tamaño para ir directo a la casilla","Salta de bloque en bloque de tamaño raíz de n","Estima la posición con una regla de tres","Quicksort con pivote"];
const $=id=>document.getElementById(id);
$("chips").innerHTML=EJ.map(e=>'<button class="chip" type="button">'+e+'</button>').join("");
$("chips").onclick=e=>{if(e.target.classList.contains("chip")){$("txt").value=e.target.textContent;ir()}};
$("clear").onclick=()=>{$("txt").value="";$("res").classList.add("hide");$("simcard").classList.add("hide");$("txt").focus()};
$("go").onclick=ir;
$("txt").addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();ir()}});
fetch("/estado").then(r=>r.json()).then(s=>{$("modo").textContent="Modelo: "+s.modo}).catch(()=>{});
async function ir(){
  const texto=$("txt").value.trim(); if(!texto)return;
  $("go").disabled=true;
  try{
    const r=await fetch("/clasificar",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({texto})});
    const d=await r.json(); if(!r.ok) throw new Error(d.error||"Error");
    pintar(d);
  }catch(err){alert("No se pudo clasificar: "+err.message)}
  $("go").disabled=false;
}
function esc(s){return String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
function pintar(d){
  $("res").classList.remove("hide");
  $("eltexto").textContent="«"+d.texto+"»";
  const t=$("tipo");
  t.textContent=d.seguro?d.info.nombre:"No estoy seguro";
  t.className="tipo"+(d.seguro?"":" dudoso");
  $("conf").textContent="Confianza "+d.confianza.toFixed(1)+"%";
  $("fuente").textContent=d.fuente;
  $("resumen").textContent=d.seguro?d.info.resumen:"La confianza es baja. Lo más parecido es «"+d.info.nombre+"», pero conviene reformular el texto con más detalle.";
  $("datos").innerHTML=d.seguro&&d.etiqueta!=="otro_tema"?"<b>Complejidad:</b> "+esc(d.info.complejidad)+" · <b>"+esc(d.info.ordenado)+"</b>":"";
  const a=$("aviso"); if(d.aviso){a.textContent=d.aviso;a.classList.remove("hide")}else a.classList.add("hide");
  $("bars").innerHTML=d.ranking.map((x,i)=>'<div class="b'+(i===0?" top":"")+'"><span>'+esc(x.nombre)+'</span><div class="track"><div class="fill" style="width:'+x.confianza+'%"></div></div><span>'+x.confianza.toFixed(1)+'%</span></div>').join("");
  const sc=$("simcard");
  if(d.simulacion.length){sc.classList.remove("hide");$("obj").textContent=d.objetivo;$("lista").textContent="["+d.lista.join(", ")+"]";$("sim").textContent=d.simulacion.join("\n")}
  else sc.classList.add("hide");
}
</script>
</body>
</html>
"""


class Manejador(BaseHTTPRequestHandler):
    clasificador = None  # se asigna en main()

    def _enviar(self, codigo, cuerpo, tipo="application/json; charset=utf-8"):
        datos = cuerpo if isinstance(cuerpo, bytes) else cuerpo.encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(datos)))
        self.end_headers()
        self.wfile.write(datos)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._enviar(200, PAGINA, "text/html; charset=utf-8")
        elif self.path == "/estado":
            modo = "Machine Learning for Kids" if self.clasificador.clave else "local de respaldo"
            self._enviar(200, json.dumps({"modo": modo}))
        else:
            self._enviar(404, json.dumps({"error": "No encontrado"}))

    def do_POST(self):
        if self.path != "/clasificar":
            return self._enviar(404, json.dumps({"error": "No encontrado"}))
        try:
            largo = int(self.headers.get("Content-Length", 0))
            if largo > 10_000:
                raise ValueError("texto demasiado largo")
            texto = json.loads(self.rfile.read(largo).decode("utf-8")).get("texto", "").strip()
            if not texto:
                raise ValueError("escribe un texto")
            self._enviar(200, json.dumps(self.clasificador.clasificar(texto), ensure_ascii=False))
        except Exception as e:
            self._enviar(400, json.dumps({"error": str(e)}, ensure_ascii=False))

    def log_message(self, *args):  # sin ruido en consola
        pass


def mostrar_consola(r):
    print(f"\n  Texto      : {r['texto']}")
    if r["seguro"]:
        print(f"  Tipo       : {r['info']['nombre']}")
    else:
        print(f"  Tipo       : No estoy seguro (lo más parecido: {r['info']['nombre']})")
    print(f"  Confianza  : {r['confianza']:.1f}%")
    print(f"  Fuente     : {r['fuente']}")
    if r["aviso"]:
        print(f"  [aviso] {r['aviso']}")
    if r["seguro"] and r["etiqueta"] != "otro_tema":
        print(f"  Complejidad: {r['info']['complejidad']} · {r['info']['ordenado']}")
    print()


def main():
    p = argparse.ArgumentParser(description="Identifica el tipo de algoritmo de búsqueda.")
    p.add_argument("--texto", help="clasifica un texto en consola y termina")
    p.add_argument("--pruebas", action="store_true", help="ejecuta frases de prueba en consola")
    p.add_argument("--key", help="clave API del proyecto de ML for Kids")
    p.add_argument("--local", action="store_true", help="usar solo el modelo local")
    p.add_argument("--puerto", type=int, default=8000, help="puerto de la interfaz web (8000)")
    p.add_argument("--sin-navegador", action="store_true", help="no abrir el navegador solo")
    a = p.parse_args()

    clasificador = Clasificador(cargar_clave(a.key), a.local)
    if a.pruebas or a.texto:
        for t in FRASES_PRUEBA if a.pruebas else [a.texto]:
            mostrar_consola(clasificador.clasificar(t))
        return

    Manejador.clasificador = clasificador
    try:
        servidor = ThreadingHTTPServer(("127.0.0.1", a.puerto), Manejador)
    except OSError:
        sys.exit(f"El puerto {a.puerto} está ocupado. Prueba con --puerto 8080")
    url = f"http://127.0.0.1:{a.puerto}"
    modo = "Machine Learning for Kids" if clasificador.clave else "modelo local de respaldo"
    print(f"Aplicación lista en {url}  (modelo: {modo})\nCtrl+C para cerrar.")
    if not a.sin_navegador:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nCerrado.")


if __name__ == "__main__":
    main()
