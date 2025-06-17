# examples/pain_agent_with_knowledge.py
"""Example: Integrating KnowledgeManager with PainAssessmentAgent"""

import sys
from pathlib import Path

# Add app to Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.knowledge import KnowledgeManager
from app.agents.pain import build as build_pain_agent
from llama_index.llms.gemini import Gemini
from llama_index.core.tools import QueryEngineTool


def create_enhanced_pain_agent():
    """Create a pain assessment agent enhanced with knowledge retrieval."""
    
    # Initialize LLM
    llm = Gemini(model="models/gemini-2.0-flash")
    
    # Initialize knowledge manager
    km = KnowledgeManager(llm=llm)
    
    # Check if we have knowledge bases available
    available_kb = km.get_available_knowledge_bases()
    print(f"Available knowledge bases: {len(available_kb)}")
    
    if not available_kb:
        print("⚠ No knowledge bases found. Please embed documents first using:")
        print("  python scripts/embed_documents.py embed documents/shoulder_pain_final_report.md")
        return None
    
    # Create tools for each available knowledge base
    tools = []
    
    for kb in available_kb:
        if kb['knowledge_type'] == 'pain':  # Only pain assessment knowledge for now
            body_region = kb['body_region']
            
            # Create query engine for this body region
            query_engine = km.create_query_engine(
                body_region=body_region,
                knowledge_type='pain',
                response_mode="tree_summarize"
            )
            
            # Create a tool for this knowledge base
            tool = QueryEngineTool.from_defaults(
                query_engine=query_engine,
                name=f"{body_region}_pain_knowledge",
                description=f"Knowledge base for {body_region} pain assessment, causes, evaluation questions, red flags, and exercise criteria"
            )
            
            tools.append(tool)
            print(f"✓ Added {body_region} pain knowledge tool")
    
    # Create enhanced pain agent with knowledge tools
    pain_agent = build_pain_agent(llm)
    
    # Add tools to the agent
    if hasattr(pain_agent, 'tools'):
        pain_agent.tools.extend(tools)
    else:
        pain_agent.tools = tools
    
    print(f"✓ Enhanced pain agent created with {len(tools)} knowledge tools")
    return pain_agent


def simulate_pain_assessment():
    """Simulate a pain assessment conversation."""
    
    print("🤖 Creating enhanced pain assessment agent...")
    agent = create_enhanced_pain_agent()
    
    if not agent:
        return
    
    print("\n" + "="*60)
    print("PAIN ASSESSMENT SIMULATION")
    print("="*60)
    
    # Simulate user reporting shoulder pain
    user_message = "I have shoulder pain that started about 2 weeks ago. It hurts when I do overhead movements like shoulder press."
    
    print(f"\n👤 User: {user_message}")
    print("\n🤖 Agent Response:")
    print("-" * 40)
    
    try:
        # The agent should use its knowledge base to ask relevant questions
        response = agent.chat(user_message)
        print(response.response)
        
        print("\n" + "-" * 60)
        print("Example follow-up questions the agent might ask:")
        print("- What exercises were you doing when the pain started?")
        print("- Can you describe the type of pain (sharp, dull, pinching)?")
        print("- Does the pain occur during specific movements or positions?")
        print("- Have you had any previous shoulder injuries?")
        
    except Exception as e:
        print(f"Error during conversation: {e}")


def test_knowledge_retrieval():
    """Test direct knowledge retrieval."""
    
    print("\n🔍 Testing Knowledge Retrieval")
    print("="*40)
    
    km = KnowledgeManager()
    
    # Test queries for different aspects
    test_queries = [
        "What are the main causes of shoulder pain in fitness training?",
        "What questions should I ask to assess shoulder pain?",
        "What are the red flags for shoulder pain that require medical attention?",
        "What exercise modifications are recommended for shoulder impingement?",
    ]
    
    try:
        # Get shoulder knowledge base
        query_engine = km.create_query_engine(
            body_region="shoulder",
            knowledge_type="pain"
        )
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            print("🔍 Response:")
            response = query_engine.query(query)
            print(response.response[:200] + "..." if len(response.response) > 200 else response.response)
            print("-" * 40)
            
    except Exception as e:
        print(f"Error testing retrieval: {e}")
        print("Make sure you have embedded the shoulder document first!")


def main():
    """Main demonstration function."""
    print("🏋️ GoGain Pain Assessment Agent with Knowledge Integration")
    print("="*60)
    
    # Test knowledge retrieval first
    test_knowledge_retrieval()
    
    # Then simulate agent conversation
    simulate_pain_assessment()
    
    print("\n✅ Demo completed!")
    print("\nNext steps:")
    print("1. Add more body region documents to documents/")
    print("2. Embed them using: python scripts/embed_documents.py batch documents/")
    print("3. Integrate knowledge tools with other agents (training, nutrition)")


if __name__ == "__main__":
    main() 