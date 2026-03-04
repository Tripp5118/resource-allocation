# core/llm_decision_maker.py
"""
Multi-stage LLM decision maker for Bayesian Optimization resource allocation.

Decision pipeline per iteration:
  Stage 1 — Reflection   : Analyse the outcome of the previous decision
  Stage 2 — Belief update: Synthesise patterns across all outcomes into
                           structured beliefs about the problem
  Stage 3 — Decision     : Use beliefs to select the best allocation option

The LLM backend is supplied by the caller (default: LangChain ChatOpenAI),
so the class works with any LangChain-compatible chat model or any callable
that accepts a string prompt and returns a string response.

Rate limiting
-------------
Two optional knobs control how aggressively we hit the API:
  min_seconds_between_calls : float  – minimum wall-clock gap between any two
                                        API calls (simple per-call throttle)
  max_calls_per_minute      : int    – hard QPM ceiling; the limiter tracks a
                                        rolling 60-second window and sleeps
                                        until capacity is available

If both are set both constraints are enforced.
"""

from __future__ import annotations

import json
import os
import re
import time
import logging
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple, Union

from core.decision_maker import DecisionMaker
from core.decision_state import DecisionState, StrategyOutcome

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Enforces min-delay and QPM constraints on API calls."""

    def __init__(
        self,
        min_seconds_between_calls: float = 0.0,
        max_calls_per_minute: Optional[int] = None,
    ):
        self.min_gap = min_seconds_between_calls
        self.qpm = max_calls_per_minute
        self._last_call_time: float = 0.0
        self._call_times: Deque[float] = deque()  # timestamps of recent calls

    def wait(self):
        """Block until we're allowed to make the next call."""
        now = time.monotonic()

        # 1. Min-gap constraint
        gap_needed = self.min_gap - (now - self._last_call_time)
        if gap_needed > 0:
            time.sleep(gap_needed)
            now = time.monotonic()

        # 2. QPM constraint — prune timestamps older than 60s, then sleep if full
        if self.qpm is not None:
            window = 60.0
            cutoff = now - window
            while self._call_times and self._call_times[0] < cutoff:
                self._call_times.popleft()
            if len(self._call_times) >= self.qpm:
                sleep_until = self._call_times[0] + window
                sleep_for = sleep_until - now
                if sleep_for > 0:
                    logger.info(
                        "[RateLimiter] QPM limit reached (%d/min), sleeping %.1fs",
                        self.qpm, sleep_for,
                    )
                    time.sleep(sleep_for)
                    now = time.monotonic()

        self._last_call_time = now
        self._call_times.append(now)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_llm(
    llm,
    prompt: str,
    temperature_override: Optional[float] = None,
) -> str:
    """
    Call a LangChain chat model and return the response text.

    `llm` is any LangChain BaseChatModel (or compatible callable).
    If it's a plain callable (e.g. a mock) we call it directly.
    """
    try:
        if hasattr(llm, "invoke"):
            # Standard LangChain interface
            if temperature_override is not None and hasattr(llm, "temperature"):
                # Temporarily override temperature if the model exposes it
                original = llm.temperature
                llm.temperature = temperature_override
                result = llm.invoke(prompt)
                llm.temperature = original
            else:
                result = llm.invoke(prompt)
            # HumanMessage / AIMessage → .content; plain string fallback
            return result.content if hasattr(result, "content") else str(result)
        else:
            # Treat as plain callable (useful for testing / human-in-the-loop)
            return str(llm(prompt))
    except Exception as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc


def _format_outcome_line(o: StrategyOutcome) -> str:
    batch = o.n_exploit + o.n_explore
    ratio_pct = int(round(o.n_explore / batch * 100)) if batch else 0
    return (
        f"  Iter {o.iteration}: [{o.n_exploit} exploit + {o.n_explore} explore "
        f"({ratio_pct}% explore)] → "
        f"Score: {o.score_before:.4f}→{o.score_after:.4f} ({o.improvement:+.4f}), "
        f"HV: {o.hv_before:.4f}→{o.hv_after:.4f} ({o.actual_improvement:+.4f}), "
        f"Pareto pts: {o.pareto_points_added:+d}"
    )


def _format_options(state: DecisionState) -> str:
    lines = []
    for i, opt in enumerate(state.allocation_options.options):
        batch = opt.total_batch_size if hasattr(opt, "total_batch_size") else "?"
        n_exp = opt.num_exploitation if hasattr(opt, "num_exploitation") else "?"
        n_exr = opt.num_exploration if hasattr(opt, "num_exploration") else "?"
        ehvi = opt.hypervolume_improvement if hasattr(opt, "hypervolume_improvement") else float("nan")
        mi = opt.information_gain if hasattr(opt, "information_gain") else float("nan")
        lines.append(
            f"  Option {i}: {n_exp} exploit + {n_exr} explore "
            f"(batch={batch}) | qEHVI={ehvi:.4f} | MutualInfo={mi:.4f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MultiStageLLMDecisionMaker(DecisionMaker):
    """
    Three-stage LLM decision maker.

    Parameters
    ----------
    llm :
        Any LangChain BaseChatModel (e.g. ChatOpenAI) or plain callable
        ``(prompt: str) -> str``.  This is the *only* required argument.
    problem_description : str
        Free-text description of the optimisation problem shown in Stage 3.
    obj1_name, obj2_name : str
        Names of the two objectives.
    log_dir : str | None
        Directory for per-iteration JSON logs and the final
        ``belief_history.json``.  No logs written if None.
    min_seconds_between_calls : float
        Minimum wall-clock seconds between consecutive API calls.
    max_calls_per_minute : int | None
        Hard QPM ceiling.  None means no QPM limit.
    stage1_temperature : float
        Temperature for the reflection stage (low → deterministic).
    stage2_temperature : float
        Temperature for the belief-update stage.
    stage3_temperature : float
        Temperature for the decision stage (slightly higher → some variety).
    """

    def __init__(
        self,
        llm,
        problem_description: str = "",
        obj1_name: str = "Objective 1",
        obj2_name: str = "Objective 2",
        log_dir: Optional[str] = None,
        min_seconds_between_calls: float = 1.0,
        max_calls_per_minute: Optional[int] = None,
        stage1_temperature: float = 0.2,
        stage2_temperature: float = 0.2,
        stage3_temperature: float = 0.4,
    ):
        self.llm = llm
        self.problem_description = problem_description
        self.obj1_name = obj1_name
        self.obj2_name = obj2_name
        self.log_dir = log_dir
        self.stage1_temperature = stage1_temperature
        self.stage2_temperature = stage2_temperature
        self.stage3_temperature = stage3_temperature

        self._rate_limiter = _RateLimiter(
            min_seconds_between_calls=min_seconds_between_calls,
            max_calls_per_minute=max_calls_per_minute,
        )

        # Persistent belief state — carried forward across iterations
        self.current_beliefs: Optional[Dict] = None
        self.belief_history: List[Dict] = []

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def make_decision(
        self, state: DecisionState
    ) -> Tuple[int, str, Optional[Dict]]:
        """
        Run the three-stage pipeline and return (option_idx, reasoning, beliefs).
        Falls back to the balanced (middle) option on any unrecoverable error.
        """
        num_options = len(state.allocation_options.options)
        fallback_idx = num_options // 2

        try:
            # Stage 1 — only meaningful once we have at least one outcome
            if state.strategy_outcomes:
                reflection = self._stage1_reflection(state)
            else:
                reflection = "No previous decisions to reflect on yet."

            # Stage 2 — belief update
            beliefs = self._stage2_belief_update(state, reflection)

            # Stage 3 — decision
            option_idx, reasoning = self._stage3_decision(state, beliefs)

        except Exception as exc:
            logger.warning(
                "[MultiStageLLM] Pipeline error at iteration %d: %s — "
                "falling back to balanced option %d",
                state.iteration, exc, fallback_idx,
            )
            beliefs = self.current_beliefs or {}
            option_idx = fallback_idx
            reasoning = f"[Fallback due to error: {exc}]"

        # Clamp to valid range
        option_idx = max(0, min(option_idx, num_options - 1))

        # Persist beliefs
        beliefs["_iteration"] = state.iteration
        self.current_beliefs = beliefs
        self.belief_history.append(beliefs)

        # Log
        self._log_iteration(state.iteration, reflection if state.strategy_outcomes else None,
                            beliefs, option_idx, reasoning)

        return option_idx, reasoning, beliefs

    def cleanup(self):
        """Save belief history JSON at end of run."""
        if self.log_dir and self.belief_history:
            path = os.path.join(self.log_dir, "belief_history.json")
            with open(path, "w") as f:
                json.dump(self.belief_history, f, indent=2, default=str)
            logger.info("[MultiStageLLM] Saved belief history → %s", path)

    # ------------------------------------------------------------------
    # Stage 1 — Reflection
    # ------------------------------------------------------------------

    def _stage1_reflection(self, state: DecisionState) -> str:
        last: StrategyOutcome = state.strategy_outcomes[-1]
        batch = last.n_exploit + last.n_explore
        ratio_pct = int(round(last.n_explore / batch * 100)) if batch else 0

        prompt = f"""You are helping guide a Bayesian Optimization campaign.

Previous decision (Iteration {last.iteration}):
  Allocation: {last.n_exploit} exploit + {last.n_explore} explore out of {batch} total ({ratio_pct}% exploration)

Before evaluation:
  Best score : {last.score_before:.4f}
  Pareto HV  : {last.hv_before:.4f}

After evaluation:
  Best score : {last.score_after:.4f}  (change: {last.improvement:+.4f})
  Pareto HV  : {last.hv_after:.4f}  (change: {last.actual_improvement:+.4f})
  Pareto pts : {last.pareto_points_added:+d} net change

Question: Did this allocation decision lead to meaningful progress? \
What does this outcome tell you about whether exploitation or exploration \
is currently effective in this problem?

Write a 2-3 sentence analysis. Be specific about the numbers."""

        self._rate_limiter.wait()
        print(f"\n{'─'*60}")
        print(f"[LLM Stage 1 — Reflection] Iteration {state.iteration}")
        print(f"{'─'*60}")
        print(prompt)
        response = _call_llm(self.llm, prompt, self.stage1_temperature)
        print(f"\n[LLM Stage 1 — Response]")
        print(response)
        print(f"{'─'*60}")
        logger.debug("[Stage1] Reflection:\n%s", response)
        return response.strip()

    # ------------------------------------------------------------------
    # Stage 2 — Belief update
    # ------------------------------------------------------------------

    def _stage2_belief_update(self, state: DecisionState, reflection: str) -> Dict:
        # Format recent outcome history (last 5)
        recent = state.strategy_outcomes[-5:] if state.strategy_outcomes else []
        history_lines = "\n".join(_format_outcome_line(o) for o in recent) if recent else "  (none yet)"

        # Strategy effectiveness summary
        eff = state.strategy_effectiveness
        if eff:
            eff_text = (
                f"  Exploit-heavy (ratio<0.33): avg score Δ={eff.exploit_heavy['avg_improvement']:+.4f}, "
                f"success={eff.exploit_heavy['success_rate']:.0%}, n={eff.exploit_heavy['n_times_used']}\n"
                f"  Balanced (ratio 0.33-0.67) : avg score Δ={eff.balanced['avg_improvement']:+.4f}, "
                f"success={eff.balanced['success_rate']:.0%}, n={eff.balanced['n_times_used']}\n"
                f"  Explore-heavy (ratio>0.67) : avg score Δ={eff.explore_heavy['avg_improvement']:+.4f}, "
                f"success={eff.explore_heavy['success_rate']:.0%}, n={eff.explore_heavy['n_times_used']}"
            )
        else:
            eff_text = "  (not enough history yet)"

        # Progress velocity — inlined directly into the prompt below
        vel = state.progress_velocity
        if vel:
            vel_recent_str = f"{vel.recent_improvement:+.4f}"
            vel_prev_str = f"{vel.previous_improvement:+.4f}"
        else:
            vel_recent_str = "(not enough history yet)"
            vel_prev_str = "(not enough history yet)"

        # Pareto status
        ps = state.pareto_status
        pareto_text = (
            f"Status: {ps.status} | {ps.num_points} points | "
            f"HV={ps.hypervolume:.4f} (Δ={ps.hypervolume_change:+.4f}, {ps.hypervolume_change_pct:+.1f}%)"
            if ps else "(not available)"
        )

        prompt = f"""You are helping guide a Bayesian Optimization campaign.
Update your beliefs about this optimisation problem based on all available evidence.

Recent reflection on the last decision:
{reflection}

Outcome history (most recent 5 decisions — each row is one iteration):
{history_lines}
Note: the strategy effectiveness summary below covers ALL iterations, not just these 5.

Strategy effectiveness summary (all iterations so far):
{eff_text}

Score progress velocity:
  Last 3 iterations improved by   : {vel_recent_str}
  Prior 3 iterations improved by  : {vel_prev_str}
(A positive number means the best score increased over those iterations.)

Pareto front status:
{pareto_text}

Context for interpreting acquisition values (you will see these in the next step):
  qEHVI measures expected direct improvement to the Pareto front (exploitation signal).
  MutualInfo measures information gain about unexplored regions (exploration signal).
  IMPORTANT: qEHVI and MutualInfo are on different scales and cannot be compared to
  each other directly. Use qEHVI to compare options against each other, and MutualInfo
  to compare options against each other — not qEHVI vs MutualInfo.
  When past decisions chose options with high MutualInfo, did the Pareto front grow?
  When past decisions chose options with high qEHVI, did the best score improve?
  Use the outcome history above to form your answer.

Based on all of this evidence, state your current beliefs using EXACTLY this format \
(fill in the bracketed values, keep the labels verbatim):

EXPLORATION_EFFECTIVENESS: [HIGH|MEDIUM|LOW] (confidence: 0.X)
EXPLOITATION_EFFECTIVENESS: [HIGH|MEDIUM|LOW] (confidence: 0.X)
PROBLEM_STRUCTURE: [1-2 sentence description of what you've learned about the landscape]
PARETO_FRONT_SATURATING: [YES|NO|UNSURE] (confidence: 0.X)

After the structured block, write 2-3 sentences justifying each belief \
with specific evidence from the history above."""

        self._rate_limiter.wait()
        print(f"\n{'─'*60}")
        print(f"[LLM Stage 2 — Belief Update] Iteration {state.iteration}")
        print(f"{'─'*60}")
        print(prompt)
        response = _call_llm(self.llm, prompt, self.stage2_temperature)
        print(f"\n[LLM Stage 2 — Response]")
        print(response)
        print(f"{'─'*60}")
        logger.debug("[Stage2] Belief update:\n%s", response)

        beliefs = self._parse_beliefs(response)
        beliefs["_raw_stage2"] = response.strip()
        return beliefs

    # ------------------------------------------------------------------
    # Stage 3 — Decision
    # ------------------------------------------------------------------

    def _stage3_decision(
        self, state: DecisionState, beliefs: Dict
    ) -> Tuple[int, str]:
        num_options = len(state.allocation_options.options)
        options_text = _format_options(state)

        # Summarise beliefs for the prompt
        belief_summary = (
            f"  Exploration effectiveness : {beliefs.get('exploration_effectiveness', 'UNKNOWN')} "
            f"(confidence {beliefs.get('exploration_confidence', '?')})\n"
            f"  Exploitation effectiveness: {beliefs.get('exploitation_effectiveness', 'UNKNOWN')} "
            f"(confidence {beliefs.get('exploitation_confidence', '?')})\n"
            f"  Problem structure        : {beliefs.get('problem_structure', 'Unknown')}\n"
            f"  Pareto front saturating  : {beliefs.get('stuck_in_local_optimum', 'UNSURE')} "
            f"(confidence {beliefs.get('stuck_confidence', '?')})"
        )

        # Resource context
        batch_size = state.allocation_options.options[0].total_batch_size \
            if hasattr(state.allocation_options.options[0], "total_batch_size") else "?"
        cost_this_iter = (
            float(batch_size) * state.cost_per_point
            if isinstance(batch_size, (int, float)) else "?"
        )
        iters_remaining = (
            int(state.budget_remaining / cost_this_iter)
            if isinstance(cost_this_iter, float) and cost_this_iter > 0 else "?"
        )

        pareto_pts_str = str(state.pareto_status.num_points) if state.pareto_status else "?"
        pareto_hv_str = f"{state.pareto_status.hypervolume:.4f}" if state.pareto_status else "?"

        prompt = f"""You are making a resource allocation decision for a Bayesian Optimization campaign.

Problem:
{self.problem_description}

Optimisation objectives: {self.obj1_name} and {self.obj2_name}

Your current beliefs about this problem:
{belief_summary}

Resources:
  Budget remaining  : ${state.budget_remaining:.2f}
  Time remaining    : {state.time_remaining:.1f} weeks
  Cost per iteration: ~${cost_this_iter}
  Est. iters left   : ~{iters_remaining}

Current status:
  Best score so far : {state.best_score:.4f}
  Total evaluations : {state.total_points_evaluated}
  Pareto front      : {pareto_pts_str} points, HV={pareto_hv_str}

Available allocation options (Option 0 = pure exploit, Option {num_options-1} = pure explore):
{options_text}
{"" if not state.upcoming_events else chr(10) + "Upcoming events:" + chr(10) + chr(10).join("  ⚠ " + e for e in state.upcoming_events)}

Decision instructions:
  - Your beliefs above are your primary guide. Choose the option whose exploit/explore
    balance best matches what you believe is currently effective in this problem.
  - The qEHVI value measures expected direct improvement to the Pareto front.
  - MutualInfo measures information gain about unexplored regions.
  - IMPORTANT: qEHVI and MutualInfo are on different scales and cannot be compared to
    each other directly. Compare qEHVI across options and MutualInfo across options
    separately — do not compare qEHVI against MutualInfo.
  - Use the resource context (budget, time, iterations remaining) as a secondary
    factor if it materially changes what is the rational choice.

Respond using EXACTLY this format:

SELECTED_OPTION: <integer 0 to {num_options-1}>
REASONING: <1-2 sentences connecting your beliefs to this specific choice>"""

        self._rate_limiter.wait()
        print(f"\n{'─'*60}")
        print(f"[LLM Stage 3 — Decision] Iteration {state.iteration}")
        print(f"{'─'*60}")
        print(prompt)
        response = _call_llm(self.llm, prompt, self.stage3_temperature)
        print(f"\n[LLM Stage 3 — Response]")
        print(response)
        print(f"{'─'*60}")
        logger.debug("[Stage3] Decision:\n%s", response)

        option_idx, reasoning = self._parse_decision(response, num_options)
        return option_idx, reasoning

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _parse_beliefs(self, text: str) -> Dict:
        """Extract structured fields from Stage 2 response."""
        beliefs: Dict = {}

        def _extract(pattern: str, default: str = "UNKNOWN") -> str:
            m = re.search(pattern, text, re.IGNORECASE)
            return m.group(1).strip() if m else default

        def _extract_confidence(label: str) -> str:
            m = re.search(
                rf"{label}.*?confidence:\s*([\d.]+)", text, re.IGNORECASE
            )
            return m.group(1) if m else "?"

        beliefs["exploration_effectiveness"] = _extract(
            r"EXPLORATION_EFFECTIVENESS:\s*(HIGH|MEDIUM|LOW)"
        )
        beliefs["exploration_confidence"] = _extract_confidence("EXPLORATION_EFFECTIVENESS")

        beliefs["exploitation_effectiveness"] = _extract(
            r"EXPLOITATION_EFFECTIVENESS:\s*(HIGH|MEDIUM|LOW)"
        )
        beliefs["exploitation_confidence"] = _extract_confidence("EXPLOITATION_EFFECTIVENESS")

        # Problem structure — capture rest of line
        m = re.search(r"PROBLEM_STRUCTURE:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        beliefs["problem_structure"] = m.group(1).strip() if m else "Unknown"

        beliefs["stuck_in_local_optimum"] = _extract(
            r"PARETO_FRONT_SATURATING:\s*(YES|NO|UNSURE)"
        )
        beliefs["stuck_confidence"] = _extract_confidence("PARETO_FRONT_SATURATING")

        return beliefs

    def _parse_decision(self, text: str, num_options: int) -> Tuple[int, str]:
        """Extract option index and reasoning from Stage 3 response."""
        fallback = num_options // 2

        # Option index
        m = re.search(r"SELECTED_OPTION:\s*(\d+)", text, re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            if not (0 <= idx < num_options):
                logger.warning(
                    "[Stage3] Parsed option %d out of range [0,%d), using fallback %d",
                    idx, num_options, fallback,
                )
                idx = fallback
        else:
            logger.warning("[Stage3] Could not parse SELECTED_OPTION, using fallback %d", fallback)
            idx = fallback

        # Reasoning
        m = re.search(r"REASONING:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
        reasoning = m.group(1).strip() if m else text.strip()

        return idx, reasoning

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_iteration(
        self,
        iteration: int,
        reflection: Optional[str],
        beliefs: Dict,
        option_idx: int,
        reasoning: str,
    ):
        if not self.log_dir:
            return
        payload = {
            "iteration": iteration,
            "stage1_reflection": reflection,
            "stage2_beliefs": beliefs,
            "stage3_selected_option": option_idx,
            "stage3_reasoning": reasoning,
        }
        path = os.path.join(self.log_dir, f"llm_iteration_{iteration}.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)