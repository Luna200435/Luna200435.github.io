# Buscador de Tipos de Algoritmos de Búsqueda (Machine Learning for Kids + Python)

Proyecto individual. Un modelo de **reconocimiento de texto** entrenado en Machine Learning for Kids
identifica a qué **tipo de algoritmo de búsqueda en listas/estructuras de datos** pertenece un nombre o
una descripción, y una **aplicación web local en Python** usa ese modelo, muestra la confianza de cada
tipo y simula paso a paso cómo ese algoritmo encontraría un número.

## Nombres usados

| Elemento | Nombre |
|---|---|
| Proyecto en ML for Kids | `Tipos de Algoritmos de Busqueda en Listas` |
| Tipo de proyecto | Reconocer **texto** (idioma: español) |
| Etiqueta 1 | `busqueda_secuencial` (lineal, con centinela) |
| Etiqueta 2 | `busqueda_binaria` (binaria, ternaria, bisección) |
| Etiqueta 3 | `busqueda_hash` (tabla hash, diccionario) |
| Etiqueta 4 | `busqueda_saltos_interpolacion` (saltos, interpolación, exponencial, Fibonacci) |
| Etiqueta 5 | `otro_tema` (ordenamiento, IA, otros temas) |

> Las etiquetas deben escribirse **exactamente** así en ML for Kids.

## Contenido

```
buscador_tipos_algoritmos/
├── README.md
├── app_buscador.py        <- la aplicación (interfaz web local, sin librerías externas)
├── config.json            <- aquí pegas tu clave API de ML for Kids
├── requirements.txt
└── datos_entrenamiento/
    ├── busqueda_secuencial.txt            (35 ejemplos, uno por línea)
    ├── busqueda_binaria.txt               (35)
    ├── busqueda_hash.txt                  (34)
    ├── busqueda_saltos_interpolacion.txt  (34)
    ├── otro_tema.txt                      (35)
    └── ejemplos_completos.csv             (todos con su etiqueta)
```

## Paso 1: Entrenar el modelo en Machine Learning for Kids

1. https://machinelearningforkids.co.uk → **Get started** → **Projects** → **Add a new project**.
2. Nombre: `Tipos de Algoritmos de Busqueda en Listas` · Reconocer: **texto** · Idioma: **Spanish**.
3. Entra al proyecto → **Train** → **Add new label** y crea las 5 etiquetas de la tabla.
4. En cada etiqueta pulsa **Add example** y pega las líneas del `.txt` correspondiente
   (una línea = un ejemplo).
5. Vuelve al proyecto → **Learn & Test** → **Train new machine learning model**.
6. Prueba el modelo en esa pantalla con frases nuevas (puedes usar las de abajo).

## Paso 2: Conectar la aplicación

1. En el proyecto ve a **Make** → **Python** y copia la **API key**.
2. Pégala en `config.json`:
   ```json
   { "api_key": "tu-clave-aqui" }
   ```
   (también sirve `--key CLAVE` o la variable de entorno `ML4K_API_KEY`).

## Paso 3: Ejecutar

```bash
python app_buscador.py                 # abre la interfaz en http://127.0.0.1:8000
python app_buscador.py --texto "Corta la lista ordenada en dos"
python app_buscador.py --pruebas       # 8 frases de prueba en consola
python app_buscador.py --local         # solo el modelo local
python app_buscador.py --puerto 8080   # si el 8000 está ocupado
```

La interfaz muestra: el tipo detectado, la confianza, la barra de probabilidad de cada tipo,
la complejidad, si requiere datos ordenados y la simulación con la lista
`[4, 9, 15, 23, 31, 38, 46, 52, 67, 80]` buscando el `46`.

## Frases para probar

| Frase | Tipo esperado |
|---|---|
| Va revisando cada elemento de la lista hasta dar con el que busca | secuencial |
| Corta la lista ordenada en dos y se queda con la mitad correcta | binaria |
| Calcula una fórmula con la clave y va directo a la casilla donde está | hash |
| Avanza de bloque en bloque y luego revisa dentro del bloque | saltos/interpolación |
| Ordena los números de menor a mayor intercambiando vecinos | otro tema |

## Notas

- Si la confianza es menor a 45 % la app responde **"No estoy seguro"** en lugar de forzar un tipo.
- Si no hay clave, no hay internet o el modelo de ML for Kids no está entrenado, la app usa un
  **modelo local de respaldo** (TF-IDF + vecinos más cercanos) entrenado con los mismos datos, y lo avisa.
- Los modelos de la cuenta gratuita de ML for Kids caducan si no se usan; si deja de responder,
  vuelve a entrenarlo en *Learn & Test*.
- Autor: Uriel Vazquez Luna · Universidad Tecnológica de Puebla.
