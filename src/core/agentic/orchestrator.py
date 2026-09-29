import time
from typing import Dict, List, Optional

from src.core.agentic.evidence_grader import EvidenceGrader
from src.core.agentic.query_analyzer import QueryAnalyzer
from src.core.agentic.query_rewriter import QueryRewriter
from src.core.agentic.retrieval_planner import RetrievalPlanner
from src.core.agentic.types import (
    AgenticRetrievalResult, AgentState, RetrievalAttempt, RetrievalPlan,
    TerminationReason,
)
from src.core.query_engine.fusion import RRFusion
from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import RetrievalResult


class AgenticRAGOrchestrator:
    """Explicit bounded state machine layered on the existing retrieval primitives."""

    def __init__(
        self,
        settings: Settings,
        hybrid_search,
        reranker=None,
        analyzer: Optional[QueryAnalyzer] = None,
        planner: Optional[RetrievalPlanner] = None,
        grader: Optional[EvidenceGrader] = None,
        rewriter: Optional[QueryRewriter] = None,
        fusion: Optional[RRFusion] = None,
    ):
        cfg = settings.agentic
        self._settings = settings
        self._search = hybrid_search
        self._reranker = reranker
        llm = None
        if cfg.planner.use_llm or cfg.grader.use_llm or cfg.rewriter.use_llm:
            try:
                from src.libs.llm.llm_factory import create_llm
                llm = create_llm(settings)
            except Exception:
                llm = None
        self._analyzer = analyzer or QueryAnalyzer(use_llm=cfg.planner.use_llm, llm=llm)
        self._planner = planner or RetrievalPlanner(cfg.max_subqueries, cfg.max_candidate_results)
        self._grader = grader or EvidenceGrader(cfg.grader.min_results, cfg.grader.min_confidence, cfg.grader.use_llm, llm)
        self._rewriter = rewriter or QueryRewriter(cfg.rewriter.use_llm, llm)
        self._fusion = fusion or RRFusion()
        self._max_iterations = cfg.max_iterations
        self._max_candidates = cfg.max_candidate_results

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict] = None,
        collection: str = "default",
        trace: Optional[TraceContext] = None,
    ) -> AgenticRetrievalResult:
        final_k = top_k or self._settings.retrieval.top_k_final
        state = AgentState(original_query=query)
        analysis = self._analyzer.analyze(query)
        if trace:
            trace.record_stage("query_analysis", intent=analysis.intent.value, complexity=analysis.complexity_score, has_identifier=analysis.has_identifier, should_decompose=analysis.should_decompose, reasoning_summary=analysis.reasoning_summary, degraded=analysis.degraded)
        try:
            plan = self._planner.plan(analysis, filters=filters, candidate_k=final_k)
        except Exception as exc:
            state.degraded = True
            plan = RetrievalPlanner(1, self._max_candidates).plan(analysis, filters=filters, candidate_k=final_k)
            plan.subqueries = [query]
            if trace:
                trace.record_stage("retrieval_plan", status="degraded", error=str(exc), strategy=plan.strategy.value)
        else:
            if trace:
                trace.record_stage("retrieval_plan", strategy=plan.strategy.value, subquery_count=len(plan.subqueries), candidate_k=plan.candidate_k, reasoning_summary=plan.reasoning_summary)

        best_results: List[RetrievalResult] = []
        best_score = -1.0
        current_query = query
        for iteration in range(1, self._max_iterations + 1):
            state.iteration = iteration
            started = time.monotonic()
            try:
                candidates, calls = self._execute_plan(plan, collection, trace)
            except Exception as exc:
                state.degraded = True
                candidates, calls = self._fallback_original(query, plan, collection, trace, exc)
            state.retrieval_calls += calls
            state.subquery_count += len(plan.subqueries)
            if trace:
                trace.record_stage(f"retrieval_attempt_{iteration}", duration_ms=(time.monotonic() - started) * 1000, query=current_query, strategy=plan.strategy.value, retrieval_calls=calls, result_count=len(candidates))

            try:
                assessment = self._grader.grade(query, candidates, plan)
            except Exception as exc:
                state.degraded = True
                assessment = EvidenceGrader(
                    self._settings.agentic.grader.min_results,
                    self._settings.agentic.grader.min_confidence,
                ).grade(query, candidates, plan)
                assessment.degraded = True
                assessment.reason_codes.append("RETRIEVER_DEGRADED")
                if trace:
                    trace.record_stage(f"evidence_grade_{iteration}_fallback", status="degraded", error=str(exc))
            attempt = RetrievalAttempt(iteration, current_query, plan, candidates, assessment, calls)
            state.attempts.append(attempt)
            quality = assessment.confidence + assessment.coverage
            if quality > best_score or (quality == best_score and len(candidates) > len(best_results)):
                best_score, best_results = quality, candidates
            if trace:
                trace.record_stage(f"evidence_grade_{iteration}", sufficient=assessment.sufficient, confidence=assessment.confidence, coverage=assessment.coverage, reason_codes=assessment.reason_codes, degraded=assessment.degraded)
            if assessment.sufficient:
                state.termination_reason = TerminationReason.SUFFICIENT_EVIDENCE
                break
            if iteration >= self._max_iterations:
                state.exhausted = True
                state.termination_reason = TerminationReason.MAX_ITERATIONS if best_results else TerminationReason.NO_RESULTS
                break
            try:
                rewritten = self._rewriter.rewrite(current_query, analysis, assessment)
                if not rewritten or rewritten == current_query:
                    raise ValueError("rewriter produced no corrective query")
                state.rewrite_count += 1
                current_query = rewritten
                plan = self._planner.replan(analysis, rewritten, filters=filters, candidate_k=final_k)
                if trace:
                    trace.record_stage(f"query_rewrite_{iteration}", rewritten_query=rewritten, reason_codes=assessment.reason_codes)
            except Exception as exc:
                state.degraded = True
                state.exhausted = True
                state.termination_reason = TerminationReason.ERROR_FALLBACK
                if trace:
                    trace.record_stage(f"query_rewrite_{iteration}", status="degraded", error=str(exc))
                break

        final_results = best_results[: self._max_candidates]
        if self._reranker is not None and final_results:
            final_results = self._reranker.rerank(query, final_results, top_k=final_k, trace=trace)
            if trace:
                trace.record_stage("final_rerank", result_count=len(final_results))
        else:
            final_results = final_results[:final_k]
        reason = state.termination_reason or (TerminationReason.NO_RESULTS if not final_results else TerminationReason.MAX_ITERATIONS)
        state.llm_calls = sum(
            int(getattr(component, "llm_calls", 0))
            for component in (self._analyzer, self._grader, self._rewriter)
        )
        state.degraded = state.degraded or analysis.degraded or any(
            attempt.assessment.degraded for attempt in state.attempts
        )
        self._set_trace_metadata(trace, state, analysis, plan, reason)
        return AgenticRetrievalResult(
            results=final_results, analysis=analysis, plan=plan, attempts=state.attempts,
            iteration_count=state.iteration, retrieval_calls=state.retrieval_calls,
            rewrite_count=state.rewrite_count, subquery_count=state.subquery_count,
            llm_calls=state.llm_calls, degraded=state.degraded,
            exhausted=state.exhausted, termination_reason=reason,
        )

    def _execute_plan(self, plan: RetrievalPlan, collection: str, trace) -> tuple[List[RetrievalResult], int]:
        lists = []
        for subquery in plan.subqueries:
            lists.append(self._search.search(query=subquery, top_k=plan.candidate_k, filters=plan.filters or None, collection=collection, trace=trace))
        if len(lists) == 1:
            return lists[0][: self._max_candidates], 1
        return self._fusion.fuse(lists, top_k=self._max_candidates), len(lists)

    def _fallback_original(self, query, plan, collection, trace, exc):
        if trace:
            trace.record_stage("agent_retrieval_fallback", status="degraded", error=str(exc), fallback="original_query_hybrid")
        results = self._search.search(query=query, top_k=plan.candidate_k, filters=plan.filters or None, collection=collection, trace=trace)
        return results, 1

    @staticmethod
    def _set_trace_metadata(trace, state, analysis, plan, reason):
        if not trace:
            return
        values = {
            "retrieval_mode": "agentic", "query_intent": analysis.intent.value,
            "selected_strategy": plan.strategy.value, "iteration_count": state.iteration,
            "retrieval_calls": state.retrieval_calls, "rewrite_count": state.rewrite_count,
            "subquery_count": state.subquery_count, "termination_reason": reason.value,
            "llm_calls": state.llm_calls, "degraded": state.degraded, "exhausted": state.exhausted,
        }
        for key, value in values.items():
            trace.set_metadata(key, value)
