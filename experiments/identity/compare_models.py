"""Replay a fixed cohort of saved identity reviews with named .env profiles.

No retrieval or baseline model calls occur. Gold labels enter scoring only,
using the existing validation evaluator's annotation alignment rules.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from pydantic import BaseModel

from evaluations.validate_roots import FIELDS, _align, _gold_roots, _label
from mellea_lrc.llm.profiles import LlmProfile
from mellea_lrc.model import Document
from mellea_lrc.validation.docket_root_lookup_courtlistener_llm_review.reviewer import (
    DocketLookupReviewContext,
    IvrDocketLookupReviewer,
)
from mellea_lrc.validation.docket_root_lookup_govinfo_llm_review.reviewer import (
    GovInfoDocketReviewContext,
    IvrGovInfoDocketReviewer,
)
from mellea_lrc.validation.locator_body_llm_judgment.reviewer import (
    BodyCorroborationContext,
    IvrBodyCorroborationReviewer,
)
from mellea_lrc.validation.reporter_root_lookup_ambiguous_llm_judgment.reviewer import (
    IvrReporterAmbiguousReviewer,
    ReporterAmbiguousReviewContext,
)
from mellea_lrc.validation.reporter_root_lookup_unique_llm_judgment.reviewer import (
    IvrReporterUniqueReviewer,
    ReporterUniqueReviewContext,
)


@dataclass(frozen=True)
class Route:
    before: str
    review: str
    field: str
    context: type
    reviewer: type


ROUTES = {
    "unique": Route(
        "validate_roots.reporter_lookup.ambiguous_rule_judgment",
        "validate_roots.reporter_lookup.unique_llm_judgment",
        "reporter_unique_review",
        ReporterUniqueReviewContext,
        IvrReporterUniqueReviewer,
    ),
    "ambiguous": Route(
        "validate_roots.reporter_lookup.unique_llm_judgment",
        "validate_roots.reporter_lookup.ambiguous_llm_judgment",
        "reporter_ambiguous_review",
        ReporterAmbiguousReviewContext,
        IvrReporterAmbiguousReviewer,
    ),
    "docket_cl": Route(
        "validate_roots.docket_lookup.courtlistener_retrieval",
        "validate_roots.docket_lookup.courtlistener_review",
        "docket_lookup_review",
        DocketLookupReviewContext,
        IvrDocketLookupReviewer,
    ),
    "docket_govinfo": Route(
        "validate_roots.docket_lookup.govinfo_retrieval",
        "validate_roots.docket_lookup.govinfo_review",
        "govinfo_docket_review",
        GovInfoDocketReviewContext,
        IvrGovInfoDocketReviewer,
    ),
    "body": Route(
        "validate_roots.locator_body_corroboration.govinfo_opinion_retrieval",
        "validate_roots.locator_body_corroboration.llm_judgment",
        "body_reviews",
        BodyCorroborationContext,
        IvrBodyCorroborationReviewer,
    ),
}


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=lambda item: item.model_dump(mode="json") if isinstance(item, BaseModel) else str(item),
        )
        + "\n"
    )
    temporary.replace(path)


def fields_for(decision: object, route: str, root: object) -> dict[str, str] | None:
    if decision is None:
        return None
    if route == "body":
        if decision.source is None:
            return None
        return {
            field: _label(
                getattr(decision.comparisons, field).result,
                source_present=getattr(decision.filing, field) is not None,
            )
            for field in FIELDS
        }
    if route != "unique" and decision.selected_candidate_index is None:
        return None
    return {
        field: _label(
            getattr(decision, field).result,
            source_present=bool(getattr(root, field)) or getattr(decision, field).propose_replacement,
        )
        for field in FIELDS
    }


def render(destination: Path) -> None:
    cohort = json.loads((destination / "cohort.json").read_text())
    results = [json.loads(path.read_text()) for path in sorted((destination / "outcomes").glob("*.json"))]
    groups = defaultdict(list)
    for result in results:
        if "native_fields" in result:
            result["fields"] = result["native_fields"]
        groups[result["profile"]["name"]].append(result)
    baselines = {}
    for result in results:
        key = (result["case"]["route"], result["case"]["citation_id"], result["case"]["document"])
        baselines[key] = {
            **result,
            "fields": result["baseline_fields"],
            "decision": result["baseline"]["decision"],
            "run": result["baseline"]["ivr"],
        }
    groups["saved_luna"] = list(baselines.values())
    lines = [
        "# Identity model comparison",
        "",
        cohort["selection_note"],
        "",
        "Saved pre-review checkpoints provide the filing context and candidates. No new retrieval or "
        "Luna calls occur. Existing schemas, prompts, grounding, and IVR repair budgets are unchanged. "
        "Gold labels enter scoring only. This selected diagnostic cohort is not a corpus recall estimate.",
        "",
        "Field precision counts issued comparisons against the existing validation annotation labels. "
        "A refusal or failed review does not issue field comparisons; acceptance is reported separately.",
        "",
        "| Profile | Accepted reviews | Compared reviews | First attempt | Repairs | Case name | Court | Date |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    if (destination / "findings.md").exists():
        lines.insert(2, "[Reviewed disagreements and interpretation](findings.md)")
        lines.insert(3, "")
    summary = {}
    for name, items in sorted(groups.items()):
        counts = {field: [0, 0] for field in FIELDS}
        for item in items:
            if item.get("fields") is not None:
                for field in FIELDS:
                    counts[field][1] += 1
                    counts[field][0] += int(item["fields"][field] == item["gold_fields"][field])
        accepted = sum(item["decision"] is not None for item in items)
        compared = sum(item["fields"] is not None for item in items)
        first = sum(
            item["decision"] is not None and len(item["run"]["attempts"]) == 1
            for item in items
            if item["run"]
        )
        repairs = sum(max(0, len(item["run"]["attempts"]) - 1) for item in items if item["run"])
        cells = [
            f"{correct}/{total} ({correct / total:.1%})" if total else "—"
            for correct, total in counts.values()
        ]
        lines.append(
            f"| {name} | {accepted}/{len(items)} | {compared}/{len(items)} | {first}/{len(items)} | {repairs} | "
            + " | ".join(cells)
            + " |"
        )
        summary[name] = {
            "reviews": len(items),
            "accepted": accepted,
            "compared": compared,
            "first_attempt": first,
            "repairs": repairs,
            "fields": counts,
        }
    lines.extend(
        [
            "",
            "## Occurrence-level results",
            "",
            "| Profile | Route | Annotation | Accepted | Fields correct | Attempts | Seconds |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in results:
        correct = (
            sum(item["fields"][f] == item["gold_fields"][f] for f in FIELDS)
            if item["fields"] is not None
            else None
        )
        lines.append(
            f"| {item['profile']['name']} | {item['case']['route']} | "
            f"[{item['case'].get('annotation_id', item['case']['citation_id'])}]"
            f"(outcomes/{item['file']}) | {'yes' if item['decision'] else 'no'} | "
            f"{str(correct) + '/3' if correct is not None else 'no comparisons'} | "
            f"{len(item['run']['attempts']) if item['run'] else '—'} | {item['seconds']} |"
        )
    save(destination / "summary.json", summary)
    if results and all("native_document" in item for item in results):
        verified = sum(item["native_document"] is not None for item in results)
        lines.extend(
            [
                "",
                f"Native production review checkpoints saved and JSON-roundtrip verified: "
                f"{verified}/{len(results)}. Other citations retain their saved Luna outcomes; "
                "these documents are scoped integration checks, not full NRP corpus runs.",
            ]
        )
    (destination / "README.md").write_text("\n".join(lines) + "\n")


async def compare(cohort_path: Path, profile_names: list[str], destination: Path | None) -> Path:
    profiles = [LlmProfile.from_env(name) for name in profile_names]
    cohort = json.loads(await asyncio.to_thread(cohort_path.read_text))
    destination = destination or Path(__file__).parent / "results" / datetime.now(UTC).strftime(
        "%Y-%m-%dT%H-%M-%SZ"
    )
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "cohort.json").exists() and json.loads(
        (destination / "cohort.json").read_text()
    ) != cohort:
        raise ValueError("Cannot resume with a different cohort")
    save(destination / "cohort.json", cohort)
    saved_profiles = [asdict(p) for p in profiles]
    if (destination / "profiles.json").exists() and json.loads(
        (destination / "profiles.json").read_text()
    ) != saved_profiles:
        raise ValueError("Cannot resume with changed profiles")
    save(destination / "profiles.json", saved_profiles)
    print(destination.resolve(), flush=True)
    semaphore = asyncio.Semaphore(3)
    documents = {}
    for case in cohort["cases"]:
        path = Path(case["document"])
        if str(path) not in documents:
            documents[str(path)] = Document.model_validate_json(await asyncio.to_thread(path.read_text))

    prepared = []
    for index, case in enumerate(cohort["cases"]):
        route = ROUTES[case["route"]]
        document = documents[case["document"]]
        checkpoint = document.get_substage(route.before)
        root = next(root for root in checkpoint.roots if root.id == case["citation_id"])
        context = route.context.from_document(checkpoint, root)
        after = document.get_substage(route.review)
        baseline_root = next(item for item in after.roots if item.id == case["citation_id"])
        baseline = getattr(baseline_root, route.field)
        if case["route"] == "body":
            baseline = baseline[-1]
        if baseline.ivr is None:
            raise ValueError("Cohort must contain actual saved baseline model calls")
        # Check all local inputs before dispatching any generation request.
        gold = _gold_roots(checkpoint)
        aligned = _align(checkpoint.roots, gold)
        root_index = next(i for i, item in enumerate(checkpoint.roots) if item.id == root.id)
        if root_index not in aligned:
            raise ValueError(f"No root gold for cohort occurrence: {case}")
        prepared.append((route, checkpoint, root, context, baseline_root, baseline))
        save(
            destination / "inputs" / f"{index:02d}.json",
            {
                "case": case,
                "checkpoint": route.before,
                "context": asdict(context),
                "source_sha256": hashlib.sha256(document.text.encode()).hexdigest(),
                "baseline": baseline.model_dump(mode="json"),
            },
        )

    async def probe(index: int, case: dict, profile: LlmProfile) -> None:
        filename = f"{index:02d}-{case['route']}-{profile.name}.json"
        output = destination / "outcomes" / filename
        if output.exists():
            return
        route, checkpoint, root, context, baseline_root, baseline = prepared[index]
        async with semaphore:
            started = monotonic()
            result = {"file": filename, "case": case, "profile": asdict(profile)}
            try:
                outcome = await route.reviewer.from_profile(profile)(context)
                result.update(
                    decision=outcome.decision.model_dump(mode="json") if outcome.decision else None,
                    run=outcome.run.model_dump(mode="json") if outcome.run else None,
                    failure=outcome.failure_reason,
                    fields=fields_for(outcome.decision, case["route"], root),
                )
                if outcome.run:
                    result["same_baseline_prompt"] = {
                        key: getattr(outcome.run, key) == getattr(baseline.ivr, key)
                        for key in ("prefix", "instruction", "user_variables", "output_schema")
                    }
            except Exception as error:
                message = str(error).replace(profile.resolve().api_key, "[redacted]")
                result.update(
                    decision=None, run=None, failure=f"{type(error).__name__}: {message}", fields=None
                )
            result["seconds"] = round(monotonic() - started, 2)
        # Gold is never passed to the reviewer or its validation requirements.
        gold = _gold_roots(checkpoint)
        roots = checkpoint.roots
        aligned = _align(roots, gold)
        root_index = next(i for i, item in enumerate(roots) if item.id == root.id)
        target = gold[aligned[root_index]]
        result["gold_fields"] = target.labels
        result["gold_identity"] = target.identity
        result["baseline_fields"] = fields_for(baseline.decision, case["route"], baseline_root)
        result["baseline"] = baseline.model_dump(mode="json")
        save(output, result)
        print(
            json.dumps(
                {
                    "case": index,
                    "route": case["route"],
                    "profile": profile.name,
                    "accepted": result["decision"] is not None,
                    "seconds": result["seconds"],
                    "failure": result["failure"],
                }
            ),
            flush=True,
        )
        render(destination)

    await asyncio.gather(
        *(probe(index, case, profile) for index, case in enumerate(cohort["cases"]) for profile in profiles)
    )
    from experiments.identity.replay import replay_saved_outcomes

    await replay_saved_outcomes(destination)
    render(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--profiles", nargs="+")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--render", type=Path)
    parser.add_argument("--materialize", type=Path, help="Apply saved answers offline, with no model calls")
    args = parser.parse_args()
    if args.materialize:
        from experiments.identity.replay import replay_saved_outcomes

        asyncio.run(replay_saved_outcomes(args.materialize))
        render(args.materialize)
    elif args.render:
        render(args.render)
    else:
        if not args.cohort or not args.profiles:
            parser.error("--cohort and --profiles are required for model calls")
        asyncio.run(compare(args.cohort, args.profiles, args.destination))


if __name__ == "__main__":
    main()
