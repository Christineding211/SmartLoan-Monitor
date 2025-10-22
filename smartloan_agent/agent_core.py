"""
AI-Powered ML Governance Agent
Updated for LangChain 1.0+ using LangGraph
"""
import os
from typing import TypedDict, Annotated, Sequence
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from .agent_tools import (
    run_new_batch_processing,
    check_drift_metrics,
    run_fairness_audit,
    generate_compliance_report,
    get_current_metrics
)

SYSTEM_PROMPT = """You are an AI governance agent for ML model monitoring at SmartLoan.

Your capabilities:
- Monitor model performance (AUC, KS metrics)
- Detect data drift (PSI analysis)
- Audit fairness (disparate impact, 80% rule)
- Generate compliance reports

When users ask questions:
1. Use tools to gather current data
2. Interpret results clearly in plain English
3. Identify issues and root causes
4. Suggest concrete actions with priorities
5. Always consider regulatory compliance

Key thresholds to remember:
- PSI > 0.10: Warning
- PSI > 0.20: Alert
- AUC drop > 0.05: Alert
- Disparate Impact < 0.80: Fails fairness

Be concise, actionable, and always explain what metrics mean."""


class GovernanceAgent:
    """AI agent for ML governance using OpenAI and LangGraph"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "No API key found. Set OPENAI_API_KEY in .env file or pass api_key parameter."
            )
        
        # Use OpenAI
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=self.api_key
        )
        
        self.tools = [
            get_current_metrics,
            check_drift_metrics,
            run_fairness_audit,
            generate_compliance_report,
            run_new_batch_processing
        ]
        
        # Create agent using LangGraph's create_react_agent
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT
        )
    
    def chat(self, user_message: str) -> str:
        """Chat with the agent"""
        try:
            # Invoke the agent
            result = self.agent.invoke({
                "messages": [HumanMessage(content=user_message)]
            })
            
            # Extract the last message (AI's response)
            if result and "messages" in result:
                last_message = result["messages"][-1]
                if hasattr(last_message, 'content'):
                    return last_message.content
                return str(last_message)
            
            return "I couldn't process that request."
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def run_daily_check(self) -> str:
        """Automated daily health check"""
        return self.chat(
            "Run a daily health check. Check current metrics, drift, and performance. "
            "Flag any issues that need attention and suggest actions."
        )
    
    def run_weekly_audit(self) -> str:
        """Automated weekly audit"""
        return self.chat(
            "Run a comprehensive weekly audit. Check drift, performance, "
            "fairness metrics, and generate a compliance report. "
            "Provide an executive summary."
        )
    
    def clear_history(self):
        """Clear conversation history"""
        pass


# Convenience function for quick testing
def test_agent():
    """Test the agent with a simple query"""
    print("Initializing agent...")
    agent = GovernanceAgent()
    
    print("Sending test query...")
    response = agent.chat("What's the current model performance?")
    
    print("\n" + "="*50)
    print("Agent Response:")
    print("="*50)
    print(response)
    return response


if __name__ == "__main__":
    # Test when run directly
    test_agent()
