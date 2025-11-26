"""
Agents package for JOLT Platform
Contains both CrewAI and LangChain agents
"""

# Don't import at package level to avoid circular imports
# Import directly from modules when needed:
# from agents.crewai_jolt_agent import JoltSpecificationCreator
# from agents.langchain_validation_agent import JoltValidator

__all__ = ['crewai_jolt_agent', 'langchain_validation_agent']
