#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indexer — v5 (TagsMax): exhaustive tagging + robust branching + block-first chunking
-----------------------------------------------------------------------------------
Goals:
- Maximize tag coverage so routing/branching is reliable (canonical clinical tags).
- Preserve decision-tree structure (Oui/Non branches; Si...alors...; imperatives).
- Keep knowledge coverage ≈1x using a block-first chunker (low duplication).
- Store both STEP (workflow) and CONTENT (knowledge) vectors (1536d, cosine).

Compat:
  qdrant-client==1.15.1, openai, python-dotenv

Run:
  python index_pain_docs_v5_tagsmax.py --data_dir /path/to/pain_docs \
    --collection pain_docs_v5_tagsmax --workers 4

Env (.env):
  OPENAI_API_KEY=...
  QDRANT_URL=http://localhost:6333
  QDRANT_API_KEY=...
  RAG_ENRICH_LLM=1            # default ON for workflow nodes
  RAG_ENRICH_MODEL=gpt-4o
"""

import os, re, uuid, time, argparse
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ---------------------- Config ----------------------

PAIN_DOC_BASENAMES = {
    "foot_ankle_final_pain_report.md": "foot_ankle",
    "knees_pain_final_report.md": "knee",
    "hips_final_pain_report.md": "hip",
    "shoulder_pain_final_report.md": "shoulder",
    "neck_pain_final_report.md": "neck",
    "lower_back_pain_final_report.md": "low_back",
}

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# Block-first chunking targets
CHUNK_CHARS = 1100
CHUNK_OVERLAP = 220  # chars

# LLM enrichment defaults ON for workflow nodes (can disable via env)
USE_LLM_ENRICH = os.getenv("RAG_ENRICH_LLM", "1") == "1"
LLM_MODEL = os.getenv("RAG_ENRICH_MODEL", "gpt-4o")

REGION_FR_NOUN = {
    "knee": "genou",
    "foot_ankle": "cheville/pied",
    "hip": "hanche",
    "shoulder": "épaule",
    "neck": "cou",
    "low_back": "bas du dos",
}

# ---------------------- Canonical tags & patterns ----------------------
# Map of canonical tag -> list of regex synonyms to detect
TAG_PATTERNS = {
    # roots
    "trauma": [r"\b(chute|accident|torsion|tordu|impact|collision|coup|entorse|luxation|blessure)\b"],
    "no_trauma": [r"\b(progressif|graduel|surutilisation|posture|assis|bureau|chronique|r[ée]p[ée]titif|tendin(?:ite|opathie))\b"],
    "general": [],

    # swelling
    "swelling": [r"\b(gonfl|enfl|[éè]panchement|h[ée]marthrose|o[œ]d[èe]me)\b"],
    "swelling_immediate": [r"\b(imm[ée]diat|tout de suite|dans l'heure|dans l’heure)\b"],
    "swelling_delayed": [r"\b(tardif|soir|le lendemain|progressif)\b"],

    # stability
    "locking": [r"\b(bloqu|verrouill|coinc|coincement|blocage|locking|d[ée]clic|clic)\b"],
    "instability": [r"\b(instabil|d[ée]rob|d[ée]bo[iî]t|sublux|appr[ée]hension)\b"],
    "pop": [r"\b(pop|craquement|claquement|crack)\b"],

    # mechanics / triggers
    "stairs": [r"\b(escaliers?|mont[ée]e|descente)\b"],
    "running": [r"\b(course|footing|running)\b"],
    "squat": [r"\b(accroupi|squat)\b"],
    "overhead": [r"\b(overhead|au[- ]dessus de la t[êe]te|lancer|throw)\b"],
    "load": [r"\b(port de charge|soulever|halt[èe]res?)\b"],

    # location & anatomy common
    "anterior": [r"\bant[ée]rieur\b"],
    "posterior": [r"\bpost[ée]rieur\b"],
    "medial": [r"\bm[ée]dial\b"],
    "lateral": [r"\blat[ée]ral\b"],
    "internal": [r"\binterne\b"],
    "external": [r"\bexterne\b"],

    # region specifics
    # knee
    "patell": [r"\b(rotule|patell)\w*"],
    "menisque": [r"\bm[ée]nisque\w*"],
    "pfps": [r"\bf[ée]moro[- ]?patell"],
    "itb": [r"\bbandelette ilio[- ]?tibiale|ITB\b"],
    "anserine": [r"\banserine|patte d[’']?oie"],

    # shoulder
    "rotator_cuff": [r"\bcoiffe|\brotator cuff\b"],
    "biceps": [r"\bbiceps\b"],
    "impingement": [r"\bimpingement|accrochage\b"],
    "calcific": [r"\bcalcifiante?\b"],
    "shoulder_instab": [r"\binstabilit[ée]|d[ée]bo[iî]t\b"],

    # hip
    "trochanter": [r"\btrochanter\w*"],
    "fai": [r"\bf[ée]moro[- ]?ac[ée]tabulaire|FAI\b"],
    "labrum": [r"\blabrum\b"],
    "groin": [r"\baine\b|\binguinal\w*"],
    "snapping_hip": [r"\bclaquement|snapping\b"],

    # foot/ankle
    "achille": [r"\bachille\b|\btendon d[’']?achille\b"],
    "fasciite": [r"\bfasci(?:ite|a) plantaire\b|\bfascia plantaire\b"],
    "heel_spur": [r"\b[ée]pine calcan[ée]enne\b"],
    "lat_sprain": [r"\bentorse lat[ée]rale\b"],
    "stress_fracture": [r"\bfracture de stress\b"],
    "tibial_post": [r"\btibial post[ée]rieur\b"],
    "metatarsalgia": [r"\bm[ée]tatarsalgie\b"],

    # neck
    "cervico_brachial": [r"\bcervico[- ]?brachial\w*"],
    "radicular": [r"\bradicul\w*"],
    "paresthesia": [r"\bparesth[ée]s\w*|fourmillement\w*|engourdissement\w*"],
    "vertigo": [r"\bvertige\w*|dizziness\b"],
    "whiplash": [r"\bwhiplash|coup du lapin\b"],

    # low back
    "sciatica": [r"\bsciatique\w*|radicul\w*"],
    "night_pain": [r"\bdouleur nocturne\b"],
    "morning_stiff": [r"\braideur matinale\b"],
    "cauda_equina": [r"\bcauda equina\b|anesth[ée]sie en selle|incontinence|r[ée]tention|selles?\b"],

    # systemic red flags
    "fever": [r"\bfi[èe]vre\b"],
    "weight_loss": [r"\bperte de poids\b"],
    "infection": [r"\binfection\b"],
    "cancer_history": [r"\bcancer\b|\bant[ée]c[ée]dent de cancer\b"],
}

# Family grouping for audits/routing
TAG_FAMILIES = {
    "roots": {"trauma", "no_trauma", "general"},
    "swelling": {"swelling", "swelling_immediate", "swelling_delayed"},
    "stability": {"locking", "instability", "pop", "shoulder_instab"},
    "mechanics": {"stairs", "running", "squat", "overhead", "load"},
    "location": {"anterior", "posterior", "medial", "lateral", "internal", "external"},
    "knee": {"patell", "menisque", "pfps", "itb", "anserine"},
    "shoulder": {"rotator_cuff", "biceps", "impingement", "calcific", "shoulder_instab"},
    "hip": {"trochanter", "fai", "labrum", "groin", "snapping_hip"},
    "foot_ankle": {"achille", "fasciite", "heel_spur", "lat_sprain", "stress_fracture", "tibial_post", "metatarsalgia"},
    "neck": {"cervico_brachial", "radicular", "paresthesia", "vertigo", "whiplash"},
    "low_back": {"sciatica", "night_pain", "morning_stiff", "cauda_equina"},
    "systemic": {"fever", "weight_loss", "infection", "cancer_history"},
}

# ---------------------- Regex helpers ----------------------
HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)")
BULLET_RE = re.compile(r'^\s*(?:[-*•]|\d+\.)\s+')
Q_END_PAT = re.compile(r"[?]\s*$")
BULLET_Q_PAT = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+(.+?\?)")
BOLD_Q_PAT = re.compile(r"^\s*(?:\*\*|__)(.+?\?)(?:\*\*|__)")
IMPERATIVE_PAT = re.compile(r"^\s*(Demande|V[ée]rifie|Confirme|Pr[ée]cise|Poser la question)\b", re.I)
OUI_NON_PAT = re.compile(r"^\s*(?:[-*•]|\d+\.)\s*(Oui|Non)\b[:\s-]?(.*)$", re.I)
SI_ALORS_PAT = re.compile(r"^\s*Si\s+(.+?)\s+alors\s+(.+)$", re.I)

def normalize_text_for_path(s: str) -> str:
    s = s.replace("’", "'").replace("œ", "oe").replace("Œ", "Oe")
    t = s.strip().lower()
    t = re.sub(r"[^\wÀ-ÿ\-\s]", "", t)
    t = re.sub(r"\s+", "_", t)
    return t.strip("_")

# ---------------------- Data classes ----------------------
from dataclasses import dataclass, field

@dataclass
class Section:
    path: List[str]
    text: str

@dataclass
class WorkflowStep:
    doc_id: str
    region: str
    section_path: str
    question_text: str
    path_id: str
    path_index: int
    conditions_in: List[str] = field(default_factory=list)
    red_flag_gate: bool = False
    branch: str = "none"  # yes|no|none

# ---------------------- Markdown → Sections ----------------------
def split_into_sections(md_text: str) -> List[Section]:
    sections: List[Section] = []
    stack: List[Tuple[int, str]] = []
    buf: List[str] = []

    def flush():
        if buf:
            path = [p for (_, p) in stack if p] or ["root"]
            section_text = "\n".join(buf).strip()
            sections.append(Section(path=path[:], text=section_text))
            buf.clear()

    for line in md_text.splitlines():
        m = HEADER_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = normalize_text_for_path(m.group(2))
            while stack and stack[-1][0] >= level:
                stack.pop()
            if title:
                stack.append((level, title))
        else:
            buf.append(line)

    flush()
    if not sections:
        sections.append(Section(path=["root"], text=md_text.strip()))
    return sections

# ---------------------- Block-first chunking ----------------------
def _split_blocks(txt: str) -> List[str]:
    lines = txt.splitlines()
    blocks, buf = [], []
    for ln in lines:
        if BULLET_RE.match(ln):
            if buf:
                bl = "\n".join(buf).strip()
                if bl: blocks.append(bl); buf = []
            blocks.append(ln.strip())
        elif ln.strip() == "":
            if buf:
                bl = "\n".join(buf).strip()
                if bl: blocks.append(bl); buf = []
        else:
            buf.append(ln)
    if buf:
        bl = "\n".join(buf).strip()
        if bl: blocks.append(bl)
    return [b for b in blocks if b]

def chunk_text(txt: str, size: int, overlap: int) -> List[str]:
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    if not txt: return []
    blocks = _split_blocks(txt)
    chunks, buf = [], []
    buf_len = 0
    def flush_buf():
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n\n".join(buf))
            # carry overlap
            carry, rem = [], overlap
            for part in reversed(buf):
                need = len(part) + 2
                if need <= rem:
                    carry.insert(0, part); rem -= need
                else:
                    break
            buf = carry
            buf_len = sum(len(x)+2 for x in buf) if buf else 0
    for b in blocks:
        add = len(b) if buf_len == 0 else len(b) + 2
        if buf_len and (buf_len + add) > size:
            flush_buf()
        buf.append(b); buf_len += add
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks

# ---------------------- Tagging ----------------------
def canonical_tags_from_text(text: str, region: str, section_path: str) -> List[str]:
    tags = set()
    t = f"{text or ''} {section_path or ''}".lower()
    for tag, patterns in TAG_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, t):
                tags.add(tag)
                break
    # roots inference by section path if none
    if "trauma" not in tags and "no_trauma" not in tags:
        if "trauma" in section_path or "traumatisme" in section_path:
            tags.add("trauma")
        if "progressif" in section_path or "surutilisation" in section_path:
            tags.add("no_trauma")
    # region hint
    tags.add(region)
    # add tokens from section_path as lightweight hints
    for seg in (section_path or "").split("/"):
        seg = seg.strip("_")
        if seg and len(seg) > 2:
            # keep only alphanum core for tags (optional)
            pass
    return sorted(tags)

def red_flag_detect(text: str) -> bool:
    t = text.lower()
    if re.search(r"\b(fi[èe]vre|perte de poids|anesth[ée]sie en selle|incontinence|r[ée]tention|selles?|douleur nocturne)\b", t):
        return True
    if re.search(r"\b(cancer|infection)\b", t):
        return True
    return False

def build_path_id(region: str, section_path: str, tags: List[str], branch: str | None) -> str:
    root = "general"
    if "trauma" in tags: root = "trauma"
    elif "no_trauma" in tags: root = "no_trauma"
    base = f"{region}__{root}"
    if branch in ("yes", "no"):
        base += f"__{branch}"
    return base

# ---------------------- Step extraction ----------------------
def transform_si_alors(line: str) -> str | None:
    m = SI_ALORS_PAT.match(line)
    if not m: return None
    cond = m.group(1).strip().rstrip(".")
    cond = re.sub(r"^il y (a|avait)\s+", "", cond, flags=re.I)
    return f"Est-ce que {cond} ?"

def header_to_question(title: str, region: str) -> str:
    title_l = title.lower().replace("_", " ")
    noun = REGION_FR_NOUN.get(region, "zone douloureuse")
    if "blocage" in title_l or "verrouill" in title_l or "coinc" in title_l:
        return f"Ton {noun} est-il bloqué (impossibilité de l'étendre complètement) ?"
    if "gonflement" in title_l or "e panchement" in title_l or "œd" in title_l or "oed" in title_l:
        return f"As-tu observé un gonflement du {noun} ? Si oui, est-il apparu rapidement après l'incident ?"
    if "instabil" in title_l or "d[ée]rob" in title_l:
        return f"Ton {noun} te paraît-il instable (sensation qu'il 'part') ?"
    if "arbre de décision" in title_l or "décision" in title_l:
        return f"Pour préciser, {noun}: est-ce suite à un incident ou douleur progressive ?"
    # generic
    return f"Concernant « {title_l} », peux-tu préciser ?"

def extract_steps_from_section(sec_text: str) -> List[Tuple[str, str]]:
    steps = []
    parent_q = None
    for raw in sec_text.splitlines():
        line = raw.strip()
        if not line: continue

        m = BULLET_Q_PAT.match(line)
        if m:
            q = m.group(1).strip()
            steps.append((q, "none"))
            parent_q = q
            continue

        m = BOLD_Q_PAT.match(line)
        if m:
            q = m.group(1).strip()
            steps.append((q, "none"))
            parent_q = q
            continue

        if Q_END_PAT.search(line):  # ends with ?
            steps.append((line, "none"))
            parent_q = line
            continue

        if IMPERATIVE_PAT.match(line):
            # Try to extract quoted question inside line
            qs = re.findall(r"(?:«|\"|')([^\"»']+\?)", line)
            if qs:
                for q in qs:
                    steps.append((q.strip(), "none"))
                parent_q = qs[-1].strip()
            else:
                imp = re.sub(r"^\s*(Demande|V[ée]rifie|Confirme|Pr[ée]cise|Poser la question)\s*(de|si|:)?\s*", "", line, flags=re.I)
                if imp:
                    q = imp.rstrip(".").strip()
                    if not q.endswith("?"):
                        q = q[0].upper() + q[1:] + " ?"
                    steps.append((q, "none"))
                    parent_q = q
            continue

        m = OUI_NON_PAT.match(line)
        if m and parent_q:
            yn = m.group(1).strip().lower()  # oui/non
            branch = "yes" if yn == "oui" else "no"
            composed = parent_q + f" → {yn.capitalize()}"
            steps.append((composed, branch))
            continue

        q2 = transform_si_alors(line)
        if q2:
            steps.append((q2, "none"))
            parent_q = q2
            continue

    return steps

# ---------------------- Embedding helper ----------------------
def embed_texts(oai: OpenAI, texts: List[str], max_retries: int = 5) -> List[List[float]]:
    if not texts: return []
    out = []
    B = 64
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        attempt = 0
        while True:
            try:
                resp = oai.embeddings.create(model=EMBED_MODEL, input=batch)
                out.extend([d.embedding for d in resp.data])
                break
            except Exception:
                attempt += 1
                if attempt >= max_retries: raise
                time.sleep(min(2 ** attempt, 10))
    return out

# ---------------------- Optional LLM enrichment ----------------------
def llm_enrich(oai: OpenAI, question_text: str, section_path: str, region: str) -> Tuple[List[str], bool]:
    if not USE_LLM_ENRICH:
        return [], False
    try:
        sys_prompt = (
            "Tu reçois une question clinique (FR) pour un assistant d'évaluation de douleur.\n"
            "Ajoute 3-8 tags concis (mots-clés) cliniques et anatomiques utiles (ex: trauma, swelling_immediate, locking, patell, menisque, impingement, trochanter, radicular, fasciite, cauda_equina, etc.). "
            "Utilise des slugs anglais quand c'est standard (ex: rotator_cuff, impingement). "
            "Indique aussi si c'est un 'red_flag_gate' (true/false).\n"
            "Réponds en JSON: {\"tags\": [...], \"red_flag_gate\": true/false}"
        )
        user_msg = f"question: {question_text}\nsection_path: {section_path}\nregion: {region}"
        chat = oai.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role":"system","content":sys_prompt},
                      {"role":"user","content":user_msg}],
            temperature=0.2,
        )
        content = chat.choices[0].message.content or "{}"
        import json
        data = json.loads(content)
        tags = data.get("tags") or []
        rfg = bool(data.get("red_flag_gate", False))
        # normalize
        def norm(t:str)->str:
            t = t.strip().lower().replace(" ", "_")
            t = t.replace("’","'")
            t = re.sub(r"[^a-z0-9_]", "", t)
            return t
        tags = [norm(t) for t in tags if isinstance(t, str)]
        return tags, rfg
    except Exception:
        return [], False

# ---------------------- Qdrant helpers ----------------------
def ensure_collection(client: QdrantClient, name: str):
    try:
        exists = False
        try:
            exists = client.collection_exists(name)
        except Exception:
            cols = [c.name for c in client.get_collections().collections]
            exists = name in cols
        if not exists:
            client.create_collection(
                collection_name=name,
                vectors_config={
                    "content": VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
                    "step": VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
                },
                on_disk_payload=True,
            )
    except Exception:
        pass

def upsert_points(client: QdrantClient, collection: str, points: List[PointStruct]):
    if not points: return
    B = 128
    for i in range(0, len(points), B):
        client.upsert(collection_name=collection, points=points[i:i+B])

# ---------------------- Per-file worker ----------------------
def process_doc(data_dir: str, fname: str, collection: str) -> Tuple[str, int, int]:
    region = PAIN_DOC_BASENAMES[fname]
    doc_id = fname
    oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                          api_key=os.getenv("QDRANT_API_KEY"))
    with open(os.path.join(data_dir, fname), "r", encoding="utf-8", errors="ignore") as fh:
        md = fh.read()

    sections = split_into_sections(md)

    # ---- Steps ----
    wf_steps: List[WorkflowStep] = []
    wf_texts: List[str] = []
    path_seq = defaultdict(int)

    for sec in sections:
        section_path = "/".join([s for s in sec.path if s]) if sec.path else "root"
        steps = extract_steps_from_section(sec.text)
        if not steps:
            # Only create header-derived prompts for clinically-relevant sections
            title = (sec.path[-1] if sec.path else "general")
            if any(k in title for k in ["blocage","verrouill","coinc","gonflement","instabil","décision","arbre_de_décision","limites_de_sécurité"]):
                qh = header_to_question(title, region)
                if qh: steps = [(qh, "none")]
        for q_text, branch in steps:
            base_tags = canonical_tags_from_text(q_text, region, section_path)
            red_gate = red_flag_detect(q_text)
            pid = build_path_id(region, section_path, base_tags, branch if branch!="none" else None)
            idx = path_seq[(doc_id, pid)]
            path_seq[(doc_id, pid)] += 1

            llm_tags, llm_red = llm_enrich(oai, q_text, section_path, region)
            tags = sorted(set(base_tags + llm_tags))

            wf_steps.append(WorkflowStep(
                doc_id=doc_id,
                region=region,
                section_path=section_path,
                question_text=q_text,
                path_id=pid,
                path_index=idx,
                conditions_in=tags,
                red_flag_gate=bool(red_gate or llm_red),
                branch=branch,
            ))
            wf_texts.append(q_text)

    wf_vecs = embed_texts(oai, wf_texts)
    wf_points: List[PointStruct] = []
    for step, vec in zip(wf_steps, wf_vecs):
        pid = str(uuid.uuid4())
        payload = {
            "doc_id": step.doc_id,
            "region": step.region,
            "section_path": step.section_path,
            "question_text": step.question_text,
            "path_id": step.path_id,
            "path_index": step.path_index,
            "conditions_in": step.conditions_in,
            "red_flag_gate": step.red_flag_gate,
            "branch": step.branch,
            "node_type": "workflow",
            "lang": "fr",
        }
        vectors = {"step": vec, "content": vec}
        wf_points.append(PointStruct(id=pid, vector=vectors, payload=payload))

    # ---- Knowledge ----
    kb_texts: List[str] = []
    kb_payloads: List[Dict] = []
    for sec in sections:
        section_path = "/".join([s for s in sec.path if s]) if sec.path else "root"
        for piece in chunk_text(sec.text, CHUNK_CHARS, CHUNK_OVERLAP):
            if not piece.strip(): continue
            tags = canonical_tags_from_text(piece, region, section_path)
            red = red_flag_detect(piece)
            kb_payloads.append({
                "doc_id": doc_id,
                "region": region,
                "section_path": section_path,
                "text": piece,
                "tags": tags,
                "red_flag": bool(red),
                "node_type": "knowledge",
                "lang": "fr",
            })
            kb_texts.append(piece)

    kb_vecs = embed_texts(oai, kb_texts)
    kb_points: List[PointStruct] = []
    for payload, vec in zip(kb_payloads, kb_vecs):
        pid = str(uuid.uuid4())
        vectors = {"content": vec, "step": vec}
        kb_points.append(PointStruct(id=pid, vector=vectors, payload=payload))

    upsert_points(qdrant, collection, wf_points + kb_points)
    return (doc_id, len(wf_points), len(kb_points))

# ---------------------- Main ----------------------
def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "pain_docs_v5"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                          api_key=os.getenv("QDRANT_API_KEY"))
    ensure_collection(qdrant, args.collection)

    files = [f for f in os.listdir(args.data_dir) if f in PAIN_DOC_BASENAMES]
    if not files:
        raise RuntimeError("No pain docs found. Expected: " + ", ".join(PAIN_DOC_BASENAMES.keys()))

    print(f"Indexing {len(files)} pain docs with {args.workers} workers into '{args.collection}' (LLM enrich={USE_LLM_ENRICH})")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(process_doc, args.data_dir, f, args.collection) for f in files]
        for fut in as_completed(futs):
            doc_id, n_wf, n_kb = fut.result()
            print(f"Indexed {doc_id}: steps={n_wf} knowledge_chunks={n_kb}")
            results.append((doc_id, n_wf, n_kb))

    print("Done. Summary:")
    for d, w, k in results:
        print(f"  - {d}: steps={w}, kb={k}")
    print(f"Total steps={sum(x[1] for x in results)}, kb={sum(x[2] for x in results)}")

if __name__ == "__main__":
    main()
