
"""
AI-Powered ML Governance Agent
Simple LangChain agent that orchestrates monitoring tools
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_tool_calling_agent
from langchain.agents.agent import AgentExecutor

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
    """AI agent for ML governance using OpenAI"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "No API key found. Set OPENAI_API_KEY in .env file or pass api_key parameter."
            )
        
        # CHANGED: Use OpenAI instead of Anthropic
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Fast and cheap for development
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
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )
    
    def chat(self, user_message: str) -> str:
        """Chat with the agent"""
        try:
            result = self.agent_executor.invoke({"input": user_message})
            return result.get("output", "I couldn't process that request.")
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
    agent = GovernanceAgent()
    response = agent.chat("What's the current model performance?")
    print("\n" + "="*50)
    print("Agent Response:")
    print("="*50)
    print(response)
    return response


if __name__ == "__main__":
    # Test when run directly
    test_agent()
