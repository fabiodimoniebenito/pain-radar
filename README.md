# pain_radar

Descarga textos públicos de un nicho (Hacker News, Reddit, tus propias reseñas o notas), extrae el dolor de cada uno con un LLM local, agrupa los dolores parecidos y los puntúa. Genera **hipótesis** para contrastar con entrevistas, no validación.

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # Python 3.11+
ollama pull qwen2.5:7b-instruct          # o qwen2.5:3b-instruct si va lento
ollama pull nomic-embed-text
```

Necesitas una versión de Ollama reciente (salida estructurada con esquema JSON y `/api/embed`).

Reddit es opcional: crea una app tipo "script" en reddit.com/prefs/apps y exporta
`REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` (y, si quieres, `REDDIT_USER_AGENT`). Respeta sus términos de la API y úsalo solo para investigación personal. Sin credenciales se omite Reddit y se usa solo Hacker News.

## Uso

```bash
python pain_radar.py collect --niche hosteleria     # recoge textos
python pain_radar.py analyze --limit 100            # LLM local; prueba con pocos primero
python pain_radar.py report --niche hosteleria      # out/report_hosteleria.md y .csv
python pain_radar.py run                            # todo lo anterior para todos los nichos
python pain_radar.py stats                          # cuántos textos hay por nicho
python pain_radar.py niches                         # nichos disponibles en queries.toml
```

Añade tus propias fuentes (reseñas de herramientas competidoras copiadas de G2/Trustpilot, notas de entrevistas):

```bash
python pain_radar.py import resenas.csv --niche autonomos --source g2   # CSV con columna "text" (y opcional "url")
python pain_radar.py import notas.txt --niche hosteleria                # bloques separados por una línea ---
```

Edita `queries.toml` para cambiar nichos, consultas y subreddits (los nombres de subreddit son ejemplos: comprueba que existen). `run` sin `--niche` procesa todos los nichos, que es mucho para un 7B sin GPU; ve nicho a nicho.

## Cómo se puntúa un cluster

`score = n × (0.6 + 0.2·urgencia media) × (1 + 0.5·%workaround) × (1 + 0.5·%coste explícito) × (1 + 0.5·pago medio) × (1 + 0.2·(fuentes − 1))`

Premia frecuencia, urgencia, que ya improvisen una solución, que mencionen coste y que busquen o paguen una herramienta, y que aparezca en varias fuentes. Está en `score_cluster`, fácil de cambiar.

## Límites a tener en cuenta

- **Velocidad**: sin GPU, un 7B tarda del orden de varios segundos por texto. Empieza con `--limit 50`. Todo se cachea en SQLite y puedes cortar y seguir.
- **Ruido**: un LLM pequeño se equivoca. Lee siempre las citas del informe antes de fiarte de un cluster. Si salen clusters demasiado mezclados baja `cluster_distance`; si salen demasiado fragmentados, súbela.
- **Sesgo de fuente**: Hacker News y Reddit sobrerrepresentan perfil tech y anglosajón. Para hostelería local española, tus entrevistas valen más que este script.
- **Privacidad**: solo se guarda el texto público y su URL, nunca el autor.
- `python test_pipeline.py` ejecuta una prueba offline con un LLM falso.
