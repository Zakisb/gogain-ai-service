# EXACT RAG Query Flow

## How RAG Queries Are Run - Step by Step

---

## **Turn 1: User says "j'ai mal au dos"**

### 1. AI Calls Tool:
```python
retrieve_contextual_pain_knowledge(
    query="douleur au dos",  # ← AI provides this
    body_region="dos",
    assessment_phase="initial"
)
```

### 2. System Builds 4 Queries (INITIAL phase):
```python
# Line 486-506 in enhanced_assessment_tools.py
queries = [
    {
        "query": "dos pain assessment protocol clinical examination questions",
        "priority": HIGH,
        "search_for_questions": True
    },
    {
        "query": "dos pain history taking essential questions",
        "priority": HIGH,
        "search_for_questions": True
    },
    {
        "query": "comment décrirais-tu douleur dos",  # French
        "priority": HIGH,
        "search_for_questions": True
    },
    {
        "query": "douleur au dos dos symptoms evaluation",  # ← Uses AI's query
        "priority": MEDIUM
    }
]
```

### 3. Executes 4 Parallel Searches:
```python
# Line 361-376: ThreadPoolExecutor runs all 4 queries at once
for query_info in queries:
    executor.submit(
        self._retrieve_with_query,
        query_info["query"],  # ← Each of the 4 queries
        body_region="dos",
        top_k=2  # 5 / 4 queries + 1 = 2 results per query
    )
```

### 4. Each Query Retrieves from Qdrant:
```python
# Query to Qdrant vector DB
retriever = VectorIndexRetriever(
    index=pain_hierarchical_chunks_index,
    similarity_top_k=6  # top_k * 3 for filtering
)
nodes = retriever.retrieve("dos pain assessment protocol...")
```

### 5. Consolidates Results:
```python
# Line 304: Merge all 4 query results
all_results = [
    # Results from query 1 (assessment protocol)
    # Results from query 2 (history taking)
    # Results from query 3 (French questions)
    # Results from query 4 (symptoms evaluation)
]

# Line 944-993: Deduplication + ranking
consolidated = _consolidate_results(all_results)  # Top 10 unique chunks
```

### 6. Returns to AI:
```json
{
  "medical_knowledge": [
    {"content": "« Peux-tu me décrire quand et comment le mal de dos est apparu ? »", "score": 0.76},
    {"content": "« Y a-t-il eu un accident ou un choc récent ? »", "score": 0.78},
    {"content": "« La douleur est-elle constante ou par épisodes ? »", "score": 0.74},
    ...  // 7 more chunks
  ]
}
```

---

## **Turn 4: User says "Deadlift"**

### 1. AI Calls Tool:
```python
retrieve_contextual_pain_knowledge(
    query="deadlift",  # ← NEW query from AI
    body_region="dos",
    assessment_phase="initial"  # Still INITIAL
)
```

### 2. System Builds 4 DIFFERENT Queries:
```python
# Same structure BUT base_query changed!
queries = [
    {
        "query": "dos pain assessment protocol clinical examination questions",
        "priority": HIGH
    },
    {
        "query": "dos pain history taking essential questions",
        "priority": HIGH
    },
    {
        "query": "comment décrirais-tu douleur dos",
        "priority": HIGH
    },
    {
        "query": "deadlift dos symptoms evaluation",  # ← NOW INCLUDES "deadlift"
        "priority": MEDIUM
    }
]
```

### 3. Executes 4 Parallel Searches (AGAIN):
- Query 1-3: **Same as before** (generic pain assessment)
- Query 4: **NEW** - includes "deadlift"

### 4. Different Results Retrieved:
```
Query 4 "deadlift dos symptoms evaluation" retrieves:
✅ Chunk with "Sais-tu si ton dos était droit ?" (score 0.71)
✅ Chunk with "As-tu augmenté les charges ?" (score 0.67)
✅ Chunk with "Pendant deadlift, douleur immédiate ou lendemain ?" (score 0.69)
```

### 5. Consolidates & Returns:
```json
{
  "medical_knowledge": [
    {"content": "« Pendant le deadlift, douleur immédiate ou lendemain ? »", "score": 0.69},
    {"content": "« As-tu augmenté les charges ? »", "score": 0.67},
    {"content": "« Sais-tu si ton dos était droit ? »", "score": 0.71},
    ...  // Other chunks
  ]
}
```

✅ **AI SEES THE TECHNICAL QUESTION**

---

## **Turn 5: User says "Immediate"**

### 1. AI Calls Tool:
```python
retrieve_contextual_pain_knowledge(
    query="immediate pain deadlift",  # ← DIFFERENT query again
    body_region="dos",
    assessment_phase="initial"
)
```

### 2. System Builds 4 DIFFERENT Queries:
```python
queries = [
    "dos pain assessment protocol clinical examination questions",
    "dos pain history taking essential questions",
    "comment décrirais-tu douleur dos",
    "immediate pain deadlift dos symptoms evaluation"  # ← NEW base_query
]
```

### 3. Query 4 Now Searches for "immediate pain":
```
Query: "immediate pain deadlift dos symptoms evaluation"

Semantic similarity ranking:
- Chunks about "pain timing" → HIGH score
- Chunks about "immediate vs delayed pain" → HIGH score
- Chunk with "Sais-tu si ton dos était droit ?" → LOWER score (less relevant to "immediate")
```

### 4. Different Results Retrieved:
```json
{
  "medical_knowledge": [
    {"content": "« La douleur est-elle présente tout le temps ou après certains efforts ? »", "score": 0.76},
    {"content": "« La douleur est apparue immédiatement ou le lendemain ? »", "score": 0.82},
    {"content": "« Depuis combien de temps as-tu mal ? »", "score": 0.71},
    ...
  ]
}
```

❌ **"Sais-tu si ton dos était droit ?" NOT IN TOP 10**
- It WAS retrieved from Qdrant (maybe rank 15)
- But didn't make top 10 after consolidation
- Because query is now about "immediate pain" not "deadlift technique"

---

## **KEY INSIGHT: The Problem**

```python
# Turn 4
base_query = "deadlift"  # ← AI decides this
queries[3] = "deadlift dos symptoms evaluation"  # Finds technique questions
→ Returns: "Sais-tu si ton dos était droit ?"

# Turn 5
base_query = "immediate pain"  # ← AI decides NEW query
queries[3] = "immediate pain deadlift dos symptoms evaluation"  # Different semantic focus
→ Returns: Questions about pain timing, NOT technique
→ Technical question GONE
```

---

## **Why It Happens:**

1. **AI controls the `query` parameter** each turn
2. **System uses that query** to build search queries
3. **Different query = different semantic matches**
4. **Top-k filtering** means lower-ranked chunks disappear
5. **No persistence** - each turn is independent

---

## **The Cache (Currently Not Used Effectively):**

Lines 307-327 show there IS caching:
```python
# Extract questions from retrieved chunks
extracted_questions = self._extract_questions_from_text(text)

# Cache them
if context_key is not None:
    _CACHED_PAIN_QUESTIONS[cache_key].add(question)
    print(f"🗂️ CACHE: Stored {len(extracted_questions)} questions")
```

**BUT:** The cached questions are **NEVER INJECTED BACK** into the results!

They're stored but not used. The AI never sees them again unless they happen to be retrieved by a future query.

---

## **Solution Required:**

### **Option A: Inject cached questions back into results**
```python
# Line 330 - After consolidation
consolidated_results = self._consolidate_results(all_results)

# ADD THIS:
if context_key is not None:
    cache_key = (str(context_key), str(body_region))
    cached_questions = _CACHED_PAIN_QUESTIONS.get(cache_key, set())

    # Inject cached questions that haven't been asked yet
    for q in cached_questions:
        if not already_in_results(q, consolidated_results):
            consolidated_results.append({
                "text": q,
                "score": 0.95,  # High priority
                "priority": "cached",
                "source": "persistent_cache"
            })
```

### **Option B: Make queries persistent for activity context**
```python
# If last query mentioned activity, keep it in next query
if "deadlift" in previous_queries:
    base_query = f"{base_query} deadlift technique form"
```

### **Option C: Phase-based persistence**
```python
# When user mentions activity, enter TECHNIQUE_FOCUS phase
# Stay in that phase for 3-4 turns
# All queries during that phase include activity keywords
```

---

## **Current Status:**

❌ Cache exists but is **write-only** (stores questions, never reads them back)
❌ Each turn gets **fresh RAG results** based on new query
❌ No **query persistence** - "deadlift" context lost immediately
❌ No **injection** of critical questions back into results

**The system CAN'T remember what it saw 2 turns ago.**
