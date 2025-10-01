# 🧪 Knowledge System Testing Plan

  

## Overview

  

This is your **systematic testing roadmap** to validate the knowledge management system before integrating it with your agents. We'll go from simple to complex, catching issues early.

  

## 🎯 Why Test This Way?

  

**Incremental Validation**: Each phase builds on the previous one

  

- **Phase 1**: Basic infrastructure (can we connect to things?)

- **Phase 2**: Individual components (do they work in isolation?)

- **Phase 3**: Integration (do they work together?)

- **Phase 4**: Agent enhancement (does it improve your agent?)

  

**Early Problem Detection**: Fix issues at the right level

  

- Infrastructure problems → fix environment

- Component problems → fix specific classes

- Integration problems → fix interfaces

- Agent problems → fix tool integration

  

---

  

## 🚀 Phase 1: Infrastructure Validation (5 minutes)

  

**Goal**: Verify all dependencies and connections work

  

### Step 1.1: Start Qdrant

  

```bash

# Start Qdrant in a separate terminal

docker run -p 6333:6333 qdrant/qdrant

```

  

### Step 1.2: Check Environment Variables

  

```bash

echo $OPENAI_API_KEY # Should show your OpenAI key

echo $GOOGLE_API_KEY # Should show your Google key (for Gemini)

```

  

### Step 1.3: Run Infrastructure Tests

  

```bash

python test_infrastructure.py

```

  

**Expected Output**:

  

```

🧪 INFRASTRUCTURE TESTING

==================================================

🔍 Testing imports...

✅ qdrant_client imported successfully

✅ llama_index.core imported successfully

✅ OpenAI embeddings imported successfully

✅ Gemini LLM imported successfully

  

🔍 Testing Qdrant connection...

✅ Qdrant connected successfully

Found 0 existing collections

  

🔍 Testing embedding model...

✅ Embedding model initialized successfully

  

🔍 Testing document availability...

✅ Document found: documents/shoulder_pain_final_report.md

  

📊 RESULTS

====================

Passed: 4/4 tests

🎉 All infrastructure tests passed! Ready to proceed.

```

  

**If anything fails**: Fix the infrastructure issue before proceeding

  

- Missing packages → `pip install -r requirements.txt`

- Qdrant not running → check Docker command

- Missing API keys → set environment variables

- Missing document → check file location

  

---

  

## 🔧 Phase 2: Component Testing (10 minutes)

  

**Goal**: Test each component individually

  

### Step 2.1: Run Component Tests

  

```bash

python test_components.py

```

  

**What this tests**:

  

1. **DocumentProcessor**: Can it parse your shoulder document?

2. **EmbeddingsPipeline**: Can it create vector embeddings?

3. **KnowledgeManager**: Can it orchestrate everything?

4. **End-to-End**: Can it do the full pipeline?

  

**Expected Output**:

  

```

🧪 COMPONENT TESTING

==================================================

🔍 Testing DocumentProcessor...

✅ DocumentProcessor initialized

✅ Document loaded: 1 document(s)

✅ Document processed: 156 nodes created

- Hierarchical nodes: 89

- Leaf nodes: 67

- Nodes with questions: 23

- Nodes with red flags: 8

- Nodes with exercise info: 31

  

🔍 Testing EmbeddingsPipeline...

✅ EmbeddingsPipeline initialized

✅ Test collection created: True

✅ Collections listed: 1 found

📝 Testing embedding with 5 sample nodes...

✅ Nodes embedded successfully

✅ Index loaded successfully

✅ Collection stats: 5 points

  

🔍 Testing End-to-End Processing...

📝 Processing document: documents/shoulder_pain_final_report.md

✅ End-to-end processing successful

🔍 Testing query: What are the main causes of shoulder pain?

✅ Query successful

Response length: 847 characters

  

📊 COMPONENT TEST RESULTS

==============================

DocumentProcessor ✅ PASS

EmbeddingsPipeline ✅ PASS

KnowledgeManager ✅ PASS

End-to-End ✅ PASS

  

Passed: 4/4 tests

🎉 All component tests passed! Ready for agent integration.

```

  

**If anything fails**:

  

- **DocumentProcessor fails** → Check document format, imports

- **EmbeddingsPipeline fails** → Check Qdrant connection, API keys

- **End-to-End fails** → Usually an integration issue

  

---

  

## 📚 Phase 3: Knowledge Base Creation (5 minutes)

  

**Goal**: Create your actual shoulder pain knowledge base

  

### Step 3.1: Embed the Shoulder Document

  

```bash

python scripts/embed_documents.py embed documents/shoulder_pain_final_report.md

```

  

**Expected Output**:

  

```

✓ Using Gemini for metadata extraction

✓ Successfully embedded document: documents/shoulder_pain_final_report.md

Collection: gogain_pain_shoulder

Points: 156

Status: green

```

  

### Step 3.2: Verify Knowledge Base

  

```bash

python scripts/embed_documents.py list

```

  

**Expected Output**:

  

```

Found 1 knowledge bases:

--------------------------------------------------------------------------------

Body Region Type Status Points Collection

--------------------------------------------------------------------------------

shoulder pain green 156 gogain_pain_shoulder

```

  

### Step 3.3: Test Knowledge Retrieval

  

```bash

python scripts/embed_documents.py test shoulder --query "What are the main causes of shoulder pain in fitness?"

```

  

**Expected Output**:

  

```

Query: What are the main causes of shoulder pain in fitness?

--------------------------------------------------

Based on the fitness and exercise context, the main causes of shoulder pain include:

  

1. **Muscle Imbalances**: Particularly anterior/posterior imbalances where front muscles (pectorals, anterior deltoids) are overdeveloped compared to posterior stabilizers (rotator cuff, middle/lower trapezius, rhomboids)...

  

[Detailed response about shoulder pain causes]

```

  

---

  

## 🤖 Phase 4: Agent Integration Testing (10 minutes)

  

**Goal**: Test the knowledge-enhanced agent

  

### Step 4.1: Run the Integration Demo

  

```bash

python examples/pain_agent_with_knowledge.py

```

  

**What this demonstrates**:

  

1. **Knowledge Retrieval**: Direct queries to the knowledge base

2. **Enhanced Agent**: Pain agent with knowledge tools

3. **Intelligent Conversation**: Dynamic responses based on your expertise

  

**Expected Output**:

  

```

🏋️ GoGain Pain Assessment Agent with Knowledge Integration

============================================================

  

🔍 Testing Knowledge Retrieval

========================================

  

📝 Query: What are the main causes of shoulder pain in fitness training?

🔍 Response:

The main causes include muscle imbalances, technique deficiencies, overuse...

  

📝 Query: What questions should I ask to assess shoulder pain?

🔍 Response:

Key assessment questions include asking about pain location, triggers...

  

🤖 Creating enhanced pain assessment agent...

Available knowledge bases: 1

✓ Added shoulder pain knowledge tool

✓ Enhanced pain agent created with 1 knowledge tools

  

============================================================

PAIN ASSESSMENT SIMULATION

============================================================

  

👤 User: I have shoulder pain that started about 2 weeks ago. It hurts when I do overhead movements like shoulder press.

  

🤖 Agent Response:

----------------------------------------

Based on your description of shoulder pain during overhead movements, I need to gather more specific information to properly assess your condition. Let me ask you some targeted questions:

  

1. Can you describe the exact location of the pain? Is it on the front, top, back, or side of your shoulder?

  

2. On a scale of 0-10, how would you rate the pain intensity during overhead movements?

  

3. What specific exercises were you doing when the pain first started?

  

4. Does the pain occur at a specific point during the overhead movement, such as when your arm passes horizontal or at the top of the movement?

  

[Additional intelligent questions based on your knowledge base]

```

  

### Step 4.2: Test Different Scenarios

  

Try these conversation scenarios to see how the agent responds:

  

```bash

# Scenario 1: Specific movement pain

"I get sharp pain in my shoulder when I do lateral raises around 90 degrees"

  

# Scenario 2: Multiple symptoms

"My shoulder hurts during bench press and I sometimes hear clicking sounds"

  

# Scenario 3: Red flag symptoms

"I have severe shoulder pain that keeps me awake at night and my arm feels numb"

```

  

**What to observe**:

  

- Does the agent ask **relevant follow-up questions**?

- Does it **reference your document content**?

- Does it **detect red flags** and recommend medical consultation?

- Does it **avoid giving medical diagnoses**?

  

---

  

## 🎉 Phase 5: Validation Complete!

  

If all phases pass, you now have:

  

✅ **Working Infrastructure**: All dependencies connected

✅ **Validated Components**: Each piece works individually

✅ **Functional Knowledge Base**: Your expertise is embedded and queryable

✅ **Enhanced Agent**: Pain agent with intelligent knowledge access

  

## 🚀 Next Steps After Testing

  

### Immediate (Same Day)

  

1. **Integrate with your pain agent** - Modify `app/agents/pain.py`

2. **Test with real scenarios** - Try various pain assessment conversations

3. **Refine responses** - Adjust system prompts based on results

  

### Short-term (Next Few Days)

  

1. **Add more body regions** - Create knee, back, ankle documents

2. **Batch process** - `python scripts/embed_documents.py batch documents/`

3. **Test agent handoffs** - See how central router delegates to pain agent

  

### Medium-term (Next Few Weeks)

  

1. **Training knowledge** - Add exercise selection documents

2. **Nutrition knowledge** - Add meal planning documents

3. **Full integration** - All agents with relevant knowledge tools

  

## 🆘 Troubleshooting

  

### Common Issues

  

**"ImportError: No module named..."**

  

```bash

pip install -r requirements.txt

```

  

**"Connection refused" (Qdrant)**

  

```bash

docker run -p 6333:6333 qdrant/qdrant

```

  

**"Authentication failed" (OpenAI)**

  

```bash

export OPENAI_API_KEY="your_actual_key"

```

  

**"Poor retrieval quality"**

  

- Check document structure (use proper headers)

- Verify metadata extraction worked

- Try different queries

  

### Getting Help

  

1. **Check logs** - Use `-v` flag for verbose output

2. **Review documentation** - `docs/KNOWLEDGE_SYSTEM.md`

3. **Test incrementally** - Don't skip phases

  

---

  

## 🎯 Success Criteria

  

You'll know the system is working when:

  

- ✅ All test scripts pass

- ✅ Knowledge retrieval gives relevant, accurate responses

- ✅ Agent asks intelligent follow-up questions

- ✅ Agent references your document content appropriately

- ✅ Agent detects red flags and recommends medical consultation

  

**Ready to build intelligent, knowledge-driven fitness agents! 🚀**