"""Completed IVRs survive interruption without changing native artifacts."""

import asyncio
import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel

from mellea_lrc.llm.config import OUTPUT_MODE_OPTION, PROFILE_OPTION
from mellea_lrc.llm.ivr import InstructIvrSpec, _run_metadata
from mellea_lrc.llm.journal import ReviewJournal, active_journal
from mellea_lrc.model.document import Document
from mellea_lrc.model.ivr import IvrAttempt, IvrRequirementAttempt, IvrRun
from mellea_lrc.model.site_review import SiteReview
from mellea_lrc.parsing.span import Span


class Answer(BaseModel):
    answer: str


class OtherAnswer(BaseModel):
    answer: str
    explanation: str


def _validate(context):
    return True


def _other_validate(context):
    return True


def _spec(item="one"):
    return InstructIvrSpec(
        description="Read the citation.",
        prefix="Opinion text",
        grounding_context={"source": "abc"},
        user_variables={"citation": item},
        requirements=(req("Ground the quotation.", validation_fn=_validate),),
        output_format=Answer,
    )


def _options():
    return {
        OUTPUT_MODE_OPTION: "json_schema",
        "temperature": 0,
        "max_tokens": 100,
        PROFILE_OPTION: {"name": "test", "endpoint": "https://example.invalid/v1"},
    }


def _session(model="test-model"):
    return SimpleNamespace(backend=SimpleNamespace(model_id=model))


def _result(session, spec, strategy, model_options, *, value="yes", success=True):
    options = dict(model_options)
    if PROFILE_OPTION in options:
        options[PROFILE_OPTION] = {**options[PROFILE_OPTION], "max_attempts": strategy.loop_budget}
    return IvrRun(
        success=success,
        selected_attempt=0,
        attempts=(
            IvrAttempt(
                output=json.dumps({"answer": value}),
                requirements=(
                    IvrRequirementAttempt(
                        description="Ground the quotation.",
                        passed=success,
                        reason=None if success else "Quote was not found",
                        score=None,
                    ),
                ),
                response={"usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}},
            ),
        ),
        elapsed_seconds=0.1,
        **_run_metadata(session, spec, options),
    )


async def _invoke(session, spec, **kwargs):
    return _result(session, spec, **kwargs)


async def _call(journal, spec=None, *, invoke=_invoke, session=None, strategy=None, options=None):
    return await journal.run(
        session or _session(),
        spec or _spec(),
        strategy=strategy or MultiTurnStrategy(loop_budget=3),
        model_options=options or _options(),
        invoke=invoke,
    )


def _committed_document(run):
    document = Document.from_source("abc")
    substage = "grow_roots.discover_root_locators.hunt_docket_locators"
    return document.add_site_review(
        SiteReview(
            substage=substage,
            candidate_span=Span(0, 3),
            candidate_text="abc",
            outcome="declined",
            reason="Not a citation",
            ivr=run,
        )
    ).complete_substage(substage)


def test_interrupted_substage_reuses_completed_reviews_and_only_calls_pending(tmp_path):
    async def exercise():
        calls = Counter()

        async def provider(session, spec, **kwargs):
            item = spec.user_variables["citation"]
            calls[item] += 1
            if item == "three" and calls[item] == 1:
                raise RuntimeError("Interrupted before a durable answer")
            return _result(session, spec, **kwargs, value=item)

        saved = []
        with ReviewJournal(tmp_path) as journal:
            saved.append(await _call(journal, _spec("one"), invoke=provider))
            saved.append(await _call(journal, _spec("two"), invoke=provider))
            with pytest.raises(RuntimeError, match="Interrupted"):
                await _call(journal, _spec("three"), invoke=provider)
        assert len(list(tmp_path.glob("*.json"))) == 2
        with ReviewJournal(tmp_path) as resumed:
            assert await _call(resumed, _spec("one"), invoke=provider) == saved[0]
            assert await _call(resumed, _spec("two"), invoke=provider) == saved[1]
            await _call(resumed, _spec("three"), invoke=provider)
            assert resumed.reused == 2 and resumed.recorded == 1
        assert calls == {"one": 1, "two": 1, "three": 2}
        assert active_journal() is None

    asyncio.run(exercise())


def test_terminal_failed_review_is_reused_with_unchanged_usage_and_trace(tmp_path):
    async def exercise():
        async def fail(session, spec, **kwargs):
            return _result(session, spec, **kwargs, success=False)

        first = await _call(ReviewJournal(tmp_path), invoke=fail)

        async def forbidden(*args, **kwargs):
            pytest.fail("A durably completed failed review must not be called again")

        second = await _call(ReviewJournal(tmp_path), invoke=forbidden)
        assert not second.success
        assert second.model_dump(mode="json") == first.model_dump(mode="json")
        assert second.attempts[0].usage.total_tokens == 10

    asyncio.run(exercise())


def test_committed_native_ivr_is_excluded_and_duplicate_specs_are_consumed_once(tmp_path):
    async def exercise():
        calls = 0

        async def provider(session, spec, **kwargs):
            nonlocal calls
            calls += 1
            return _result(session, spec, **kwargs, value=str(calls))

        original = ReviewJournal(tmp_path)
        first = await _call(original, invoke=provider)
        second = await _call(original, invoke=provider)
        assert calls == 2  # Equal new calls are distinct occurrences, not cache hits.
        document = _committed_document(first)
        native_bytes = document.model_dump_json()
        restored = Document.model_validate_json(native_bytes)
        resumed = ReviewJournal(tmp_path, restored)
        assert await _call(resumed, invoke=provider) == second
        third = await _call(resumed, invoke=provider)
        assert third.output == '{"answer": "3"}'
        assert calls == 3 and resumed.reused == 1
        assert restored.model_dump_json() == native_bytes
        assert len(list(tmp_path.glob("*.json"))) == 3

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "change", ["context", "variables", "schema", "model", "budget", "options", "validator"]
)
def test_changed_review_specification_is_never_reused(tmp_path, change):
    async def exercise():
        await _call(ReviewJournal(tmp_path))
        spec, session, strategy, options = _spec(), _session(), MultiTurnStrategy(loop_budget=3), _options()
        if change == "context":
            spec = replace(spec, grounding_context={"source": "Different text"})
        elif change == "variables":
            spec = replace(spec, user_variables={"citation": "Different citation"})
        elif change == "schema":
            spec = replace(spec, output_format=OtherAnswer)
        elif change == "model":
            session = _session("other-model")
        elif change == "budget":
            strategy = MultiTurnStrategy(loop_budget=1)
        elif change == "options":
            options = {**options, "temperature": 0.5}
        else:
            spec = replace(spec, requirements=(req("Ground the quotation.", validation_fn=_other_validate),))
        calls = 0

        async def provider(session, spec, **kwargs):
            nonlocal calls
            calls += 1
            return _result(session, spec, **kwargs)

        resumed = ReviewJournal(tmp_path)
        await _call(resumed, spec, session=session, strategy=strategy, options=options, invoke=provider)
        assert calls == 1 and resumed.reused == 0

    asyncio.run(exercise())


def test_task_local_journals_and_concurrent_entries_do_not_mix(tmp_path):
    async def exercise():
        first, second = ReviewJournal(tmp_path / "first"), ReviewJournal(tmp_path / "second")
        seen = []

        async def work(journal, item):
            with journal:

                async def provider(session, spec, **kwargs):
                    assert active_journal() is journal
                    await asyncio.sleep(0.005)
                    assert active_journal() is journal
                    seen.append((journal.directory.name, item))
                    return _result(session, spec, **kwargs, value=item)

                return await _call(journal, _spec(item), invoke=provider)

        await asyncio.gather(work(first, "one"), work(second, "two"))
        assert sorted(seen) == [("first", "one"), ("second", "two")]
        assert active_journal() is None
        with first:
            results = await asyncio.gather(*(_call(first, _spec(str(i))) for i in range(5)))
        assert len(results) == 5 and first.recorded == 6
        entries = list((tmp_path / "first").glob("*.json"))
        assert len(entries) == 6 and len({p.name for p in entries}) == 6
        resumed = ReviewJournal(tmp_path / "first")

        async def forbidden(*args, **kwargs):
            pytest.fail("Concurrent completed entries must be reusable")

        await asyncio.gather(*(_call(resumed, _spec(str(i)), invoke=forbidden) for i in range(5)))
        assert resumed.reused == 5

    asyncio.run(exercise())


def test_corrupt_entry_raises_before_any_review_can_run(tmp_path):
    asyncio.run(_call(ReviewJournal(tmp_path)))
    entry = next(tmp_path.glob("*.json"))
    value = json.loads(entry.read_text())
    value["ivr"]["success"] = False
    entry.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="result changed"):
        ReviewJournal(tmp_path)


def test_failed_atomic_write_preserves_prior_review_and_cleans_temporary_file(tmp_path, monkeypatch):
    async def exercise():
        first = await _call(ReviewJournal(tmp_path))
        before = (tmp_path / "000001.json").read_bytes()
        native_replace = Path.replace

        def fail_second(self, destination):
            if Path(destination).name == "000002.json":
                raise OSError("Cannot save review")
            return native_replace(self, destination)

        monkeypatch.setattr(Path, "replace", fail_second)
        journal = ReviewJournal(tmp_path)
        with pytest.raises(OSError, match="Cannot save review"):
            await _call(journal, _spec("two"))
        assert (tmp_path / "000001.json").read_bytes() == before
        assert [p.name for p in tmp_path.iterdir()] == ["000001.json"]
        assert await _call(ReviewJournal(tmp_path)) == first

    asyncio.run(exercise())


def test_context_is_restored_after_exception_and_nested_context_is_rejected(tmp_path):
    first, second = ReviewJournal(tmp_path / "first"), ReviewJournal(tmp_path / "second")
    with pytest.raises(RuntimeError, match="interrupted"):
        with first:
            assert active_journal() is first
            with pytest.raises(RuntimeError, match="already active"):
                with second:
                    pytest.fail("Nested journal entry must be rejected")
            assert active_journal() is first
            raise RuntimeError("interrupted")
    assert active_journal() is None


def test_validator_captured_dataclass_and_pydantic_values_are_part_of_exact_match(tmp_path):
    @dataclass
    class Context:
        answer: Answer
        span: Span

    def specification(answer):
        context = Context(Answer(answer=answer), Span(0, 3))
        return replace(
            _spec(),
            requirements=(req("Ground the quotation.", validation_fn=lambda ctx: context.answer),),
        )

    async def exercise():
        first = await _call(ReviewJournal(tmp_path), specification("first"))

        async def forbidden(*args, **kwargs):
            pytest.fail("Equivalent validator context should reuse the recorded result")

        resumed = ReviewJournal(tmp_path)
        assert await _call(resumed, specification("first"), invoke=forbidden) == first
        changed = ReviewJournal(tmp_path)
        await _call(changed, specification("second"))
        assert changed.reused == 0 and changed.recorded == 1

    asyncio.run(exercise())


def test_unrepresentable_validator_state_fails_before_provider(tmp_path):
    opaque = object()
    spec = replace(
        _spec(),
        requirements=(req("Ground the quotation.", validation_fn=lambda ctx: opaque),),
    )

    async def forbidden(*args, **kwargs):
        pytest.fail("Unsafe validator identity must be rejected before a request")

    with pytest.raises(TypeError, match="unsupported state"):
        asyncio.run(_call(ReviewJournal(tmp_path), spec, invoke=forbidden))
    assert list(tmp_path.iterdir()) == []


def test_actual_domain_validator_contexts_roundtrip_for_reuse(tmp_path):
    from mellea_lrc.extraction.hunt_docket_locators.candidates import DocketSiteCandidate
    from mellea_lrc.extraction.leaf_attribution_review.reviewer import LeafReviewContext
    from mellea_lrc.providers.courtlistener import CourtListenerCluster
    from mellea_lrc.validation.review_unique_reporter_roots_llm.reviewer import ReporterUniqueReviewContext

    contexts = [
        DocketSiteCandidate(Span(0, 3), Span(0, 3), "abc", "abc", "abc"),
        LeafReviewContext("abc", "id", "abc", (), (), "abc"),
        ReporterUniqueReviewContext(
            source="abc",
            locator="1 U.S. 2",
            case_name_window="abc",
            case_name_offset=0,
            following_window="",
            following_offset=3,
            current_case_name=None,
            current_court=None,
            current_date=None,
            current_case_name_normalized=None,
            has_case_name=False,
            has_court=False,
            has_date=False,
            candidate=CourtListenerCluster(id="7", date_filed="2000-01-01"),
            docket=None,
        ),
    ]

    def spec_for(context, index):
        return replace(
            _spec(str(index)),
            requirements=(req("Ground the quotation.", validation_fn=lambda ctx: context),),
        )

    async def exercise():
        journal = ReviewJournal(tmp_path)
        saved = [await _call(journal, spec_for(context, i)) for i, context in enumerate(contexts)]
        resumed = ReviewJournal(tmp_path)

        async def forbidden(*args, **kwargs):
            pytest.fail("Equivalent domain context must not repeat provider work")

        for i, context in enumerate(contexts):
            assert await _call(resumed, spec_for(replace(context), i), invoke=forbidden) == saved[i]
        assert resumed.reused == 3

    asyncio.run(exercise())


def test_public_ivr_wrapper_records_and_reuses_terminal_failed_result(tmp_path, monkeypatch):
    from mellea_lrc.llm import ivr

    async def exercise():
        calls = 0

        async def provider(session, spec, **kwargs):
            nonlocal calls
            calls += 1
            return _result(session, spec, **kwargs, success=False)

        monkeypatch.setattr(ivr, "_run_instruct_ivr", provider)
        with ReviewJournal(tmp_path):
            result = await ivr.run_instruct_ivr(
                _session(), _spec(), strategy=MultiTurnStrategy(loop_budget=3), model_options=_options()
            )
        with ReviewJournal(tmp_path) as resumed:
            repeated = await ivr.run_instruct_ivr(
                _session(), _spec(), strategy=MultiTurnStrategy(loop_budget=3), model_options=_options()
            )
            assert resumed.reused == 1
        assert calls == 1 and not repeated.success
        assert repeated.model_dump_json() == result.model_dump_json()

    asyncio.run(exercise())


def test_before_review_guard_runs_before_new_call_and_saved_result_reuse(tmp_path):
    async def exercise():
        guard_calls = 0
        provider_calls = 0
        changed = True

        def guard():
            nonlocal guard_calls
            guard_calls += 1
            if changed:
                raise RuntimeError("Run configuration changed")

        async def provider(session, spec, **kwargs):
            nonlocal provider_calls
            provider_calls += 1
            return _result(session, spec, **kwargs)

        journal = ReviewJournal(tmp_path, before_review=guard)
        with pytest.raises(RuntimeError, match="configuration changed"):
            await _call(journal, invoke=provider)
        assert guard_calls == 1 and provider_calls == 0
        assert list(tmp_path.iterdir()) == []
        changed = False
        first = await _call(journal, invoke=provider)
        assert guard_calls == 2 and provider_calls == 1
        resumed = ReviewJournal(tmp_path, before_review=guard)
        changed = True
        with pytest.raises(RuntimeError, match="configuration changed"):
            await _call(resumed, invoke=provider)
        assert guard_calls == 3 and resumed.reused == 0 and provider_calls == 1
        changed = False
        assert await _call(resumed, invoke=provider) == first
        assert guard_calls == 4 and resumed.reused == 1 and provider_calls == 1
        assert len(list(tmp_path.glob("*.json"))) == 1

    asyncio.run(exercise())
