# GoGain AI Pinecone Database Structure

```
PINECONE CLOUD DATABASE
├── gogain-pain (INDEX) 
│   ├── Dimension: 1024 vectors (Voyage AI embeddings)
│   ├── Total Vectors: 2,882
│   ├── Index Fullness: 0.0%
│   │
│   ├── 📁 knee-pain (NAMESPACE)
│   │   ├── 🔢 1,026 vectors
│   │   ├── 📄 Source: knees_pain_final_report.md
│   │   └── 🏷️  Metadata: body_region=knee, document_type=pain_assessment
│   │
│   ├── 📁 shoulder-pain (NAMESPACE) 
│   │   ├── 🔢 316 vectors
│   │   ├── 📄 Source: shoulder_pain_final_report.md
│   │   └── 🏷️  Metadata: body_region=shoulder, document_type=pain_assessment
│   │
│   ├── 📁 lower-back-pain (NAMESPACE)
│   │   ├── 🔢 307 vectors
│   │   ├── 📄 Source: lower_back_pain_final_report.md
│   │   └── 🏷️  Metadata: body_region=back, document_type=pain_assessment
│   │
│   ├── 📁 neck-pain (NAMESPACE)
│   │   ├── 🔢 236 vectors
│   │   ├── 📄 Source: neck_pain_final_report.md
│   │   └── 🏷️  Metadata: body_region=neck, document_type=pain_assessment
│   │
│   ├── 📁 hips-pain (NAMESPACE)
│   │   ├── 🔢 582 vectors  
│   │   ├── 📄 Source: hips_final_pain_report.md
│   │   └── 🏷️  Metadata: body_region=hip, document_type=pain_assessment
│   │
│   └── 📁 foot-ankle-pain (NAMESPACE)
│       ├── 🔢 415 vectors
│       ├── 📄 Source: foot_ankle_final_pain_report.md
│       └── 🏷️  Metadata: body_region=foot, document_type=pain_assessment
│
└── gogain-app (INDEX) - Fallback
    ├── Dimension: 1536 vectors (OpenAI embeddings) 
    ├── 📁 default (NAMESPACE)
    └── 🔢 Various vectors (legacy health insights, etc.)

═══════════════════════════════════════════════════════════════

AGENT ACCESS PATTERN:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  PainAssessment │───▶│ KnowledgeRetriever│───▶│  Pinecone Query │
│      Agent      │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
   User Query:            Maps body_region:         Searches:
   "knee pain"      ────▶ "knee" → "knee-pain"  ──▶ gogain-pain::knee-pain
                           namespace                    1,026 vectors

METADATA STRUCTURE:
┌────────────────────────────────────────────────┐
│ Vector Metadata (each of 2,882 vectors):      │
├────────────────────────────────────────────────┤
│ • file_path: "documents/knees_pain_final..."   │
│ • file_name: "knees_pain_final_report.md"     │
│ • file_type: "text/markdown"                  │
│ • file_size: 67799 bytes                      │
│ • creation_date: "2025-08-31"                 │
│ • last_modified_date: "2025-08-31"            │
│ • source_file: "knees_pain_final_report.md"   │
│ • document_type: "pain_assessment"            │
│ • body_region: "knee" | "shoulder" | ...      │
│ • contains_assessment_info: true              │
└────────────────────────────────────────────────┘

EMBEDDING PIPELINE:
Document → Chunks → Voyage AI (1024-dim) → Pinecone (with namespace + metadata)
```

**Key Statistics:**
- **Total Indexes:** 2 (gogain-pain + gogain-app fallback)
- **Total Namespaces:** 6 pain-specific + 1 default
- **Total Vectors:** 2,882 in pain index + legacy vectors in gogain-app
- **Embedding Model:** Voyage AI (1024 dimensions)
- **Storage:** Cloud-hosted Pinecone

**Query Flow:**
1. User mentions pain → PainAssessmentAgent activated
2. Agent calls retrieve_pain_knowledge() tool
3. Body region mapped to namespace (knee → knee-pain)
4. Vector similarity search in gogain-pain::knee-pain
5. Top-k results retrieved with metadata
6. AI generates contextual response using retrieved knowledge