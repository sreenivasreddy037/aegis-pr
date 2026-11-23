"""LLM abstraction for the review agents.

Agents always compute deterministic candidates first (AST rules, contract checks, sandbox
runs). The LLM's job is to *adjudicate and explain* those candidates and optionally add
findings or extra tests; it never sees a bare diff without structural context.

* `OpenAIReviewLLM`: live mode, structured output via `with_structured_output`.
* `MockReviewLLM`: deterministic offline responder used when `MOCK_LLM=true` or no API key
  is configured. It returns the deterministic candidates unchanged, so offline reviews are
  reproducible and still meaningful (they are what the benchmark measures in CI).
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.schemas import Category, Evidence, ReviewFinding, Severity

logger = logging.getLogger(__name__)


class CandidateAssessment(BaseModel):
    candidate_id: str = Field(description="The id of the candidate finding, e.g. C1.")
    keep: bool = Field(description="False if the candidate is a false positive.")
    severity: Severity | None = None
    description: str | None = Field(default=None, description="Improved, specific explanation.")
    suggested_fix: str | None = None
    rationale: str = ""


class NewFinding(BaseModel):
    title: str
    description: str
    file_path: str
    line: int
    category: Category
    severity: Severity
    cwe: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    suggested_fix: str | None = None


class LLMReview(BaseModel):
    assessments: list[CandidateAssessment] = Field(default_factory=list)
    additional_findings: list[NewFinding] = Field(default_factory=list)


class ProposedTest(BaseModel):
    target: str = Field(description="Fully-qualified function under test, e.g. payments.fee.")
    test_code: str = Field(description="A complete, self-contained pytest module.")
    rationale: str = ""


class LLMTestProposals(BaseModel):
    tests: list[ProposedTest] = Field(default_factory=list, max_length=3)


class ReviewLLM(ABC):
    mode: Literal["mock", "live"]

    @abstractmethod
    def review(
        self, agent: str, system_prompt: str, context: str, candidates: list[ReviewFinding]
    ) -> list[ReviewFinding]: ...

    @abstractmethod
    def propose_tests(self, system_prompt: str, context: str) -> list[ProposedTest]: ...

    @abstractmethod
    def summarize(self, system_prompt: str, context: str) -> str | None: ...


class MockReviewLLM(ReviewLLM):
    mode = "mock"

    def review(self, agent, system_prompt, context, candidates):
        return [c.model_copy(deep=True) for c in candidates]

    def propose_tests(self, system_prompt, context):
        return []

    def summarize(self, system_prompt, context):
        return None


def _candidates_json(candidates: list[ReviewFinding]) -> str:
    return json.dumps(
        [
            {
                "candidate_id": f"C{i + 1}",
                "title": c.title,
                "category": c.category.value,
                "severity": c.severity.value,
                "location": c.location,
                "cwe": c.cwe,
                "verified_by_execution_or_ast": c.verified,
                "description": c.description,
                "evidence": [e.detail for e in c.evidence][:6],
            }
            for i, c in enumerate(candidates)
        ],
        indent=2,
    )


class OpenAIReviewLLM(ReviewLLM):
    mode = "live"

    def __init__(self, settings: Settings) -> None:
        from langchain_openai import ChatOpenAI

        assert settings.openai_api_key is not None
        self._chat = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_sec,
            max_retries=2,
        )

    def review(self, agent, system_prompt, context, candidates):
        from langchain_core.messages import HumanMessage, SystemMessage

        merged = [c.model_copy(deep=True) for c in candidates]
        try:
            result: LLMReview = self._chat.with_structured_output(LLMReview).invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=f"{context}\n\n## Candidate findings from deterministic analysis\n"
                        f"{_candidates_json(candidates)}"
                    ),
                ]
            )  # type: ignore[assignment]
        except Exception as exc:  # network, rate limit, schema violation: degrade gracefully
            logger.warning("%s: LLM review failed (%s); using deterministic candidates", agent, exc)
            return merged
        return merge_llm_review(agent, merged, result)

    def propose_tests(self, system_prompt, context):
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            result: LLMTestProposals = self._chat.with_structured_output(LLMTestProposals).invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=context)]
            )  # type: ignore[assignment]
            return result.tests
        except Exception as exc:
            logger.warning("regression_tester: test proposal failed (%s)", exc)
            return []

    def summarize(self, system_prompt, context):
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            msg = self._chat.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=context)]
            )
            return str(msg.content).strip() or None
        except Exception as exc:
            logger.warning("critic: summary failed (%s)", exc)
            return None


def merge_llm_review(
    agent: str, candidates: list[ReviewFinding], result: LLMReview
) -> list[ReviewFinding]:
    """Apply LLM adjudication to deterministic candidates.

    Verified candidates (execution- or AST-proven) can be re-worded but never discarded by the
    LLM; unverified candidates the LLM rejects are kept with very low confidence so the critic
    records *why* they were dropped. LLM-only findings are capped below verified confidence.
    """
    by_id = {f"C{i + 1}": c for i, c in enumerate(candidates)}
    for a in result.assessments:
        c = by_id.get(a.candidate_id)
        if c is None:
            continue
        if a.description:
            c.description = a.description
        if a.suggested_fix:
            c.suggested_fix = a.suggested_fix
        if a.severity and not c.verified:
            c.severity = a.severity
        if not a.keep and not c.verified:
            c.confidence = min(c.confidence, 0.2)
            c.evidence.append(
                Evidence(kind="llm", detail=f"LLM judged false positive: {a.rationale}")
            )
        elif a.rationale:
            c.evidence.append(Evidence(kind="llm", detail=a.rationale))
    out = list(candidates)
    for n in result.additional_findings:
        out.append(
            ReviewFinding(
                agent=agent,
                category=n.category,
                severity=n.severity,
                title=n.title,
                description=n.description,
                file_path=n.file_path,
                line=n.line,
                cwe=n.cwe,
                confidence=min(n.confidence, 0.75),
                verified=False,
                evidence=[Evidence(kind="llm", detail="Reported by LLM; not execution-verified.")],
                suggested_fix=n.suggested_fix,
            )
        )
    return out


def get_llm(settings: Settings | None = None) -> ReviewLLM:
    settings = settings or get_settings()
    if settings.offline:
        return MockReviewLLM()
    return OpenAIReviewLLM(settings)
