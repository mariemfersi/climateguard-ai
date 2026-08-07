"""
Agents module for ClimateGuard AI
"""

from agents.cli_demo import (
    GeneralAgent,
    SystemInfo,
    QueryPattern,
    ResponseTemplate,
    create_fallback_response,
    get_system_summary,
    is_general_query
)

# Try to import other agents if available
try:
    from agents.pricing_agent.pricing_agent import PricingAgent
except ImportError:
    PricingAgent = None

try:
    from agents.regulatory_agent.regulatory_agent import RegulatoryAgent
except ImportError:
    RegulatoryAgent = None

try:
    from agents.scenario_agent.scenario_agent import ScenarioAgent
except ImportError:
    ScenarioAgent = None

try:
    from agents.report_writer_agent.report_writer_agent import ReportWriterAgent
except ImportError:
    ReportWriterAgent = None

try:
    from agents.orchestrator.orchestrator import Orchestrator, Intent, IntentResult, BaseAgent
except ImportError:
    Orchestrator = None
    Intent = None
    IntentResult = None
    BaseAgent = None

__all__ = [
    'GeneralAgent',
    'SystemInfo',
    'QueryPattern',
    'ResponseTemplate',
    'create_fallback_response',
    'get_system_summary',
    'is_general_query',
    'PricingAgent',
    'RegulatoryAgent',
    'ScenarioAgent',
    'ReportWriterAgent',
    'Orchestrator',
    'Intent',
    'IntentResult',
    'BaseAgent'
]