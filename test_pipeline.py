"""Prueba offline: sustituye Ollama por funciones falsas y comprueba BD, clustering, puntuación e informe."""
import json, tempfile, zlib
from pathlib import Path

import numpy as np

import pain_radar as pr

THEMES = {
    "schedule": ("Managers waste hours building staff schedules in spreadsheets", "restaurant manager"),
    "waste": ("Kitchens lose money on food waste because inventory is tracked by hand", "restaurant owner"),
    "invoice": ("Freelancers chase clients for late invoice payments manually", "freelancer"),
}
TEXTS = []
for i in range(6):
    TEXTS.append(f"schedule: every week I spend {i+2} hours on shift scheduling in Excel, it is a nightmare and takes forever for my team")
for i in range(5):
    TEXTS.append(f"waste: we throw away so much food, inventory counting by hand costs us {i+100} euros a month and nobody tracks it")
for i in range(4):
    TEXTS.append(f"invoice: clients pay late again, chasing every invoice by email is exhausting, variant {i} of this same complaint here")
TEXTS += [
    "hiring: We are hiring a senior backend engineer for our remote team, apply now with your resume and portfolio please",
    "BROKEN: this one makes the fake llm raise invalid json so we can test the failure path and the retry on next run",
]


def fake_chat(cfg, system, user, schema):
    if system is pr.CLUSTER_SYSTEM:
        first = user.splitlines()[0][2:60]
        return {"label": first, "summary": "Resumen de prueba.", "solution_ideas": ["Idea A", "Idea B"]}
    body = user.lower()
    if "invalid json" in body:
        raise json.JSONDecodeError("x", "", 0)
    for key, (pain, who) in THEMES.items():
        if body.split('"""')[1].strip().startswith(key):
            return {"is_pain": True, "pain": pain, "who": who,
                    "workaround": "excel" if key != "waste" else "",
                    "cost_signal": "hours per week" if key == "schedule" else "",
                    "urgency": 4 if key == "schedule" else 3, "payment_signal": 1}
    return {"is_pain": False, "pain": "", "who": "", "workaround": "", "cost_signal": "", "urgency": 0, "payment_signal": 0}


def fake_embed(cfg, texts):
    out = []
    for t in texts:
        v = np.zeros(128)
        for w in t.lower().split():
            v[zlib.crc32(w.encode()) % 128] += 1
        out.append(v.tolist())
    return out


pr.ollama_chat_json, pr.ollama_embed = fake_chat, fake_embed

with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    (d / "in.txt").write_text("\n---\n".join(TEXTS), encoding="utf-8")
    cfg = pr.load_config(None)
    cfg["analysis"]["min_chars"] = 30
    conn = pr.open_db(str(d / "t.db"))

    assert pr.import_file(conn, str(d / "in.txt"), "demo", "manual", 30) == len(TEXTS)
    assert pr.import_file(conn, str(d / "in.txt"), "demo", "manual", 30) == 0, "importar dos veces no debe duplicar"

    done, failed = pr.analyze_items(conn, cfg, "demo", None)
    assert (done, failed) == (len(TEXTS) - 1, 1), (done, failed)
    done2, failed2 = pr.analyze_items(conn, cfg, "demo", None)
    assert (done2, failed2) == (0, 1), "solo se reintenta el fallido"

    md, csvp = pr.run_report(conn, cfg, "demo", d / "out")
    clusters = pr.build_clusters(pr.load_pain_rows(conn, cfg, "demo"), cfg)
    sizes = sorted(c["n"] for c in clusters)
    assert sizes == [4, 5, 6], sizes
    assert clusters[0]["n"] == 6, "el dolor con más textos, urgencia y workaround debe quedar primero"
    report = md.read_text(encoding="utf-8")
    assert "Radar de dolores: demo" in report and "Citas representativas" in report
    print(report[:900])
    print("CSV líneas:", len(csvp.read_text().splitlines()))
    conn.close()  # en Windows no se puede borrar la carpeta temporal con la BD abierta
    print("OK")