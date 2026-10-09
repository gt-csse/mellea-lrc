"""Opinion retrieval keeps each subopinion and preserves exact substage recovery."""

import asyncio

import pytest

from mellea_lrc.api import (
    Document,
    grow_roots,
    reporter_root_lookup_cluster_retrieval,
    reporter_root_opinion_retrieval,
)
from mellea_lrc.model.citations.judgments import IdentityVerdict
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterExactAmbiguityOutcome,
    ReporterExactAmbiguityResolution,
)
from mellea_lrc.model.citations.reporter_opinion import (
    OpinionRetrievalOutcome,
    ReporterRootOpinionRetrieval,
    ReporterRootOpinionSource,
)
from mellea_lrc.providers.courtlistener import CourtListenerCitationLookup, CourtListenerError
from mellea_lrc.providers.courtlistener.models import CourtListenerCluster
from mellea_lrc.validation.reporter_root_opinion_retrieval import SUBSTAGE


class Client:
    def __init__(self, clusters, opinions):
        self.clusters = clusters
        self.opinions = opinions
        self.calls = []

    def lookup_citation(self, *args):
        return CourtListenerCitationLookup.model_validate(
            {"citation": "550 U.S. 544", "status": 200, "clusters": self.clusters}
        )

    def get_opinion(self, opinion_id):
        self.calls.append(opinion_id)
        result = self.opinions[opinion_id]
        if isinstance(result, Exception):
            raise result
        return result


def opinion(identifier, cluster="1", **changes):
    return {
        "id": int(identifier),
        "cluster": f"https://www.courtlistener.com/api/rest/v4/clusters/{cluster}/",
        "type": "020lead",
        "html_with_citations": '<p><span class="page-label">556</span>Text.</p>',
        **changes,
    }


def admitted(client, *, index=None):
    document = asyncio.run(grow_roots(Document.from_source("Twombly, 550 U.S. 544 (2007).")))
    document = reporter_root_lookup_cluster_retrieval(document, client=client)
    citation = document.roots[0].record("test_identity")
    if index is not None:
        citation = citation.with_reporter_exact_ambiguity_resolution(
            ReporterExactAmbiguityResolution(
                node_id=citation.nodes[-1].id,
                outcome=ReporterExactAmbiguityOutcome.UNIQUE_RULE_MATCH,
                passing_candidate_indices=(index,),
                selected_candidate_index=index,
            )
        )
    citation = citation.with_identity_judgment(IdentityVerdict.CORRECT_IDENTITY)
    return document.replace_citation(citation).complete_substage("test_identity")


def test_preserves_all_opinion_types_raw_html_and_native_checkpoint():
    client = Client(
        [
            {
                "id": 1,
                "sub_opinions": [
                    "https://www.courtlistener.com/api/rest/v4/opinions/20/",
                    21,
                    20,
                    22,
                ],
            }
        ],
        {
            "20": opinion("20", type="040dissent"),
            "21": opinion("21"),
            "22": opinion("22", type="010combined", ordering_key=None, unusual_field={"saved": True}),
        },
    )
    before = admitted(client)
    after = reporter_root_opinion_retrieval(before, client=client)
    saved = Document.model_validate_json(after.model_dump_json())
    result = saved.roots[0].reporter_root_opinion_retrieval
    assert client.calls == ["20", "21", "22"]
    assert result.sub_opinion_ids == ("20", "21", "20", "22")
    assert [item.response["type"] for item in result.opinions] == ["040dissent", "020lead", "010combined"]
    assert result.opinions[2].response["unusual_field"] == {"saved": True}
    assert result.opinions[0].response["html_with_citations"] == opinion("20")["html_with_citations"]
    bound_source = saved.roots[0].reporter_root_opinion_source
    assert bound_source.cluster == before.roots[0].reporter_exact_lookup.response.clusters[0]
    assert bound_source.node_id == result.node_id
    assert saved == after and saved.get_substage("test_identity") == before
    assert saved.roots[0].identity_judgments == before.roots[0].identity_judgments
    assert saved.roots[0].pin_cite is None
    with pytest.raises(ValueError, match="already completed"):
        reporter_root_opinion_retrieval(after, client=client)


def test_ambiguous_selection_fetches_the_selected_cluster_only():
    client = Client(
        [{"id": 1, "sub_opinions": [20]}, {"id": 2, "sub_opinions": [21]}], {"21": opinion("21", cluster="2")}
    )
    result = reporter_root_opinion_retrieval(admitted(client, index=1), client=client)
    assert client.calls == ["21"]
    assert result.roots[0].reporter_root_opinion_source.cluster_id == "2"
    assert result.roots[0].reporter_root_opinion_retrieval.cluster_id == "2"


def test_bound_original_source_retrieves_without_identity_lookup_and_rewinds():
    document = asyncio.run(grow_roots(Document.from_source("Twombly, 550 U.S. 544 (2007).")))
    root = document.roots[0].record("bind_original_source")
    root = root.with_identity_judgment(IdentityVerdict.CORRECT_IDENTITY)
    root = root.with_reporter_root_opinion_source(
        ReporterRootOpinionSource(
            node_id=root.nodes[-1].id,
            cluster=CourtListenerCluster.model_validate(
                {"id": 2, "sub_opinions": [21], "caseName": "Twombly", "unknown_metadata": {"saved": True}}
            ),
        )
    )
    before = document.replace_citation(root).complete_substage("bind_original_source")
    client = Client([], {"21": opinion("21", cluster="2")})

    after = reporter_root_opinion_retrieval(before, client=client)
    saved = Document.model_validate_json(after.model_dump_json())

    assert client.calls == ["21"]
    assert saved.roots[0].reporter_exact_lookup is None
    assert saved.roots[0].reporter_root_opinion_source == before.roots[0].reporter_root_opinion_source
    assert saved.roots[0].reporter_root_opinion_source.cluster.raw_json["unknown_metadata"] == {"saved": True}
    assert saved.get_substage("bind_original_source") == before
    assert saved.roots[0].reporter_root_opinion_retrieval.cluster_id == "2"


@pytest.mark.parametrize("verdict", [None, IdentityVerdict.WRONG_IDENTITY, IdentityVerdict.UNDETERMINED])
def test_bound_source_does_not_bypass_default_correct_identity_policy(verdict):
    document = asyncio.run(grow_roots(Document.from_source("Twombly, 550 U.S. 544 (2007).")))
    root = document.roots[0].record("bind_original_source")
    if verdict is not None:
        root = root.with_identity_judgment(verdict)
    root = root.with_reporter_root_opinion_source(
        ReporterRootOpinionSource(
            node_id=root.nodes[-1].id,
            cluster=CourtListenerCluster.model_validate({"id": 1, "sub_opinions": [20]}),
        )
    )
    before = document.replace_citation(root).complete_substage("bind_original_source")
    client = Client([], {})

    after = reporter_root_opinion_retrieval(before, client=client)

    assert client.calls == []
    assert after.roots[0].reporter_root_opinion_retrieval is None
    assert after.roots[0].reporter_root_opinion_source == before.roots[0].reporter_root_opinion_source


def test_bound_original_source_takes_precedence_over_lookup_selection():
    client = Client([{"id": 1, "sub_opinions": [20]}], {"21": opinion("21", cluster="2")})
    document = admitted(client)
    root = document.roots[0].record("bind_original_source")
    root = root.with_reporter_root_opinion_source(
        ReporterRootOpinionSource(
            node_id=root.nodes[-1].id,
            cluster=CourtListenerCluster.model_validate({"id": 2, "sub_opinions": [21]}),
        )
    )
    document = document.replace_citation(root).complete_substage("bind_original_source")

    after = reporter_root_opinion_retrieval(document, client=client)

    assert client.calls == ["21"]
    assert after.roots[0].reporter_root_opinion_retrieval.cluster_id == "2"


@pytest.mark.parametrize(
    "cluster_id, sub_opinion_ids, message",
    [("2", (), "bound source cluster"), ("1", (), "bound source's subopinions")],
)
def test_retrieval_cannot_claim_another_bound_source(cluster_id, sub_opinion_ids, message):
    document = asyncio.run(grow_roots(Document.from_source("Twombly, 550 U.S. 544 (2007).")))
    root = document.roots[0].record("bind_original_source")
    root = root.with_reporter_root_opinion_source(
        ReporterRootOpinionSource(
            node_id=root.nodes[-1].id,
            cluster=CourtListenerCluster.model_validate({"id": 1, "sub_opinions": [20]}),
        )
    )
    document = document.replace_citation(root).complete_substage("bind_original_source")
    root = document.roots[0].record(SUBSTAGE)

    with pytest.raises(ValueError, match=message):
        root.with_reporter_root_opinion_retrieval(
            ReporterRootOpinionRetrieval(
                node_id=root.nodes[-1].id,
                cluster_id=cluster_id,
                sub_opinion_ids=sub_opinion_ids,
                opinions=(),
            )
        )


def test_original_source_binding_is_single_assignment_with_node_provenance():
    document = asyncio.run(grow_roots(Document.from_source("Twombly, 550 U.S. 544 (2007).")))
    root = document.roots[0].record("bind_original_source")
    source = ReporterRootOpinionSource(
        node_id=root.nodes[-1].id,
        cluster=CourtListenerCluster.model_validate({"id": 1, "sub_opinions": []}),
    )
    with pytest.raises(ValueError, match="current decision node"):
        root.with_reporter_root_opinion_source(source.model_copy(update={"node_id": root.nodes[0].id}))
    root = root.with_reporter_root_opinion_source(source)
    with pytest.raises(ValueError, match="already bound"):
        root.with_reporter_root_opinion_source(source)


def test_real_missing_and_empty_opinions_are_retained():
    client = Client(
        [{"id": 1, "sub_opinions": [20, 21, 22]}],
        {
            "20": None,
            "21": opinion("21", html_with_citations=""),
            "22": opinion("22"),
        },
    )
    result = reporter_root_opinion_retrieval(admitted(client), client=client)
    assert [item.outcome for item in result.roots[0].reporter_root_opinion_retrieval.opinions] == [
        OpinionRetrievalOutcome.NOT_FOUND,
        OpinionRetrievalOutcome.EMPTY_TEXT,
        OpinionRetrievalOutcome.RETRIEVED,
    ]


@pytest.mark.parametrize(
    "response, message", [(opinion("21"), "ID differs"), (opinion("20", cluster="2"), "selected cluster")]
)
def test_misassociated_provider_response_raises(response, message):
    client = Client([{"id": 1, "sub_opinions": [20]}], {"20": response})
    before = admitted(client)
    with pytest.raises(ValueError, match=message):
        reporter_root_opinion_retrieval(before, client=client)
    assert SUBSTAGE not in before.substage_runs


def test_provider_error_is_not_a_retrieval_miss():
    client = Client(
        [{"id": 1, "sub_opinions": [20]}],
        {
            "20": CourtListenerError("quota", failure_type="http_error", upstream_status_code=429),
        },
    )
    with pytest.raises(CourtListenerError, match="quota"):
        reporter_root_opinion_retrieval(admitted(client), client=client)


def test_body_admission_without_original_cluster_does_not_fetch_corroborator():
    client = Client([], {})
    result = reporter_root_opinion_retrieval(admitted(client), client=client)
    assert client.calls == [] and result.roots[0].reporter_root_opinion_retrieval is None
    assert result.substage_runs[-1] == SUBSTAGE
