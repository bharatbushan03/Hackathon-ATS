from __future__ import annotations

import operator
from typing import Annotated, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from .agents import AGENTS, run_final, run_specialist
from .config import Settings
from .ingestion import ParsedDocument
from .llm import _as_llms
from .schemas import AgentReport, FinalEvaluation


class EvalState(TypedDict, total=False):
    document: ParsedDocument
    problem_innovation: Optional[AgentReport]
    technical: Optional[AgentReport]
    impact_presentation: Optional[AgentReport]
    final: Optional[FinalEvaluation]
    errors: Annotated[list[str], operator.add]


def build_graph(llm, settings: Settings):
    """START -> 3 specialists (parallel) -> final evaluator -> END."""
    llms = _as_llms(llm)

    def make_node(agent_id: str):
        spec = AGENTS[agent_id]

        def node(state: EvalState) -> dict:
            try:
                rep = run_specialist(
                    llms[agent_id], spec, state["document"], settings.max_retries, settings.retry_backoff
                )
                return {agent_id: rep}
            except Exception as e:  # noqa: BLE001
                return {agent_id: None, "errors": [f"{spec.title}: {e}"]}

        return node

    def final_node(state: EvalState) -> dict:
        reports = {k: state.get(k) for k in AGENTS}
        if not any(reports.values()):
            return {"final": None, "errors": ["All specialist agents failed; final evaluation skipped."]}
        try:
            fin = run_final(
                llms["final"], state["document"], reports, settings.max_retries, settings.retry_backoff
            )
            return {"final": fin}
        except Exception as e:  # noqa: BLE001
            return {"final": None, "errors": [f"Final Evaluator: {e}"]}

    g = StateGraph(EvalState)
    for aid in AGENTS:
        g.add_node(aid, make_node(aid))
        g.add_edge(START, aid)
    g.add_node("final", final_node)
    g.add_edge(list(AGENTS), "final")
    g.add_edge("final", END)
    return g.compile()
