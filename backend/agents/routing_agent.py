"""
Looks at the user query and determines how complex it is.

- Simple queries need fewer documents (faster, cheaper)
- complex queries need more documents (better quality)
- multi-hop queries need more documents (multiple reasoning steps)
"""

import time
from typing import Dict
from backend.agents.state import GraphState, AgentStep
from backend.services.llm_client import get_llm_client

def router_query(state: GraphState) -> Dict:
    """
    Classify the query complexity to determine the retrieval strategy
    """

    query = state["query"]
    
    print(f"\n{'='*60}")
    print(f"Routing Agent")
    print(f"{'='*60}")

    system_prompt = """You are an expert Query Classifier for a RAG system. Your Job is to analyze queries and classify their complexity"""
    user_prompt = f"""
        Analyze this query and classify its complexity:
        Query: "{query}"

        Classification Categories:
        1. SIMPLE_LOOKUP
            - Direct factual questions
            - Needs 1-2 sources to answer
            - Examples: 'What is X?', 'Who created Y?', 'Why did Z happen?'

        2. COMPLEX_REASONING
            - Requires synthesis of multiple concepts
            - Needs 5-7 sources
            - Examples: 'How does X work?', 'Compare X and Y', 'Explain the relationship between A and B"

        3. MULTI_HOP
            - Requires multiple steps of reasoning
            - Needs 10+ sources
            - Examples: 'Explain X, then use it to derive Y', 'What are the implications of X on Y and Z?'

        Respond in JSON Format:
        {{
            "query_type": "SIMPLE_LOOKUP" | "COMPLEX_REASONING" | "MULTI_HOP",
            "confidence": 0.0-1.0,
            "reasoning": "Breif explanation of why you chose this classification"
        }}
    """

    llm = get_llm_client();

    try:
        response = llm.generate_json(
            prompt=user_prompt,
            system_prompt=system_prompt
        )

        query_type = response.get("query_type", "COMPLEX_REASONING")
        confidence = response.get("confidence", 0.5)
        reasoning = response.get("reasoning", "No reasoning provided")

    except Exception as e:
        print(f"[ROUTING ERROR]\t{str(e)}")
        query_type = "COMPLEX_REASONING"
        confidence = 0.5
        reasoning = "Error in classification logic, using Default."
    
    query_type = query_type.lower().replace(" ", "_")

    print(f"Query Type:\t{query_type}")
    print(f"Confidence:\t{confidence:.2f}")
    print(f"Reasoning:\t{reasoning}")

    # creating agent record
    step = AgentStep(
        agent_name="routing_agent",
        action=f"classified as {query_type}",
        reasoning=reasoning,
        timestamp=time.time()
    )

    return {
        "query_type": query_type,
        "confidence": confidence,
        "agent_steps": [step]
    }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         