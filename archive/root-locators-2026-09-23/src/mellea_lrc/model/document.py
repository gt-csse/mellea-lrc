"""The shared document state passed between extraction and validation stages."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import PrivateAttr, SerializerFunctionWrapHandler, model_serializer, model_validator

from mellea_lrc.model.case_names import CaseName
from mellea_lrc.model.citations import CanonicalCitation, is_full_citation, is_leaf
from mellea_lrc.model.documents import SourceMetadata
from mellea_lrc.model.extraction_metadata import ExtractionMetadata
from mellea_lrc.model.findings import Finding
from mellea_lrc.model.locators import (
    Locator,
    colocation_layer,
    locator_layer,
)
from mellea_lrc.model.pin_cites import PinCite
from mellea_lrc.model.preprocessed import PreprocessedDocument
from mellea_lrc.model.record import WITHDRAWN_HEAD_ID, CitationRecord, Node
from mellea_lrc.model.spans import Span

if TYPE_CHECKING:
    from mellea_lrc.extraction.rules import ExtractionRules


@dataclass(frozen=True, slots=True)
class DocumentCreated:
    """The source and document-level reading at checkpoint creation."""

    kind: Literal["create"]
    source_metadata: SourceMetadata
    unread_case_names: tuple[Span, ...]


@dataclass(frozen=True, slots=True)
class SourceMetadataUpdated:
    """A later source-provenance value, retaining the preceding value in history."""

    kind: Literal["source_metadata"]
    after: SourceMetadata


@dataclass(frozen=True, slots=True)
class UnreadCaseNamesUpdated:
    """A later projection of names still unread in the document."""

    kind: Literal["unread_case_names"]
    after: tuple[Span, ...]


DocumentEvent = DocumentCreated | SourceMetadataUpdated | UnreadCaseNamesUpdated


def _is_prefix(earlier: Sequence[object], later: Sequence[object]) -> bool:
    return tuple(earlier) == tuple(later[: len(earlier)])


class Document(PreprocessedDocument):
    """One document and everything the pipeline has read in it.

    Each public stage returns an independent snapshot with the new evidence and
    decisions. Extraction adds roots and leaves; validation adds nodes and
    withdraws citations that reach nothing. Earlier stage results remain valid
    in memory. See `docs/Document.md`.

    What is here and not on a citation is here because it cannot be on one.
    `text` is the coordinate space every span indexes, and per-citation copies
    of it could disagree. `findings` are true of the document and of no citation
    in it. `passes` is what has run, which is the only way a consumer can tell a
    document whose leaves have not grown from one that writes no short forms.
    """

    schema_version: Literal[21] = 21
    artifact_type: Literal["document"] = "document"

    citations: tuple[CitationRecord, ...]
    nodes: tuple[Node, ...] = ()
    """Document-level trace nodes for readings that concern no citation.

    A rejected site is evidence about a span but is not itself a citation.
    Its node therefore belongs here, where a :class:`Finding` can point to it
    without inventing a citation record merely to carry a trace.
    """
    unread_case_names: tuple[Span, ...] = ()
    """Text naming a case that no citation covers.

    Read last, from the document with every citation blanked, so it holds what
    nothing else read: a case cited with no locator, which a reporter-driven
    tokenizer cannot see at all, and a case whose citation *was* read but whose
    name was not reached. See
    :mod:`mellea_lrc.extraction.reading.unread_names`.
    """
    findings: tuple[Finding, ...] = ()
    """What a pass learned that belongs to no citation.

    A leaf it could not grow, a site it hunted and rejected. Beside the
    citations rather than among them, because a record is a citation and one
    standing for a citation the document does not hold would be a hole in the
    invariant. See :mod:`mellea_lrc.model.findings`.
    """
    passes: tuple[str, ...] = ()
    """Which passes have run over this document, in order.

    `("root_formation",)` means complete identifiers have been assigned roots.
    `("root_formation", "leaf_growth")` includes the filing's short forms. A
    consumer that does not check these passes cannot tell an unfinished stage
    from one that found no short forms.
    """
    extraction_metadata: ExtractionMetadata
    document_events: tuple[DocumentEvent, ...]
    """Append-only history for document-level values that legitimately change."""

    _origin_snapshot: tuple[object, ...] = PrivateAttr()

    @model_validator(mode="before")
    @classmethod
    def _require_native_history(cls, data: object) -> object:
        # A native checkpoint has its explicit version marker. An old or
        # partial payload cannot silently become a newly created document.
        if (
            isinstance(data, Mapping)
            and ("schema_version" in data or "artifact_type" in data)
            and "document_events" not in data
        ):
            raise ValueError("A serialized Document must contain document_events")
        return data

    def __init__(self, **data: object) -> None:
        # Direct Python construction creates a new document. Native reloads
        # carry their original event stream, which is validated below.
        if (
            "document_events" not in data
            and "source_metadata" in data
            and "schema_version" not in data
            and "artifact_type" not in data
        ):
            data["document_events"] = (
                DocumentCreated(
                    "create",
                    data["source_metadata"],  # type: ignore[arg-type]
                    tuple(data.get("unread_case_names", ())),  # type: ignore[arg-type]
                ),
            )
        super().__init__(**data)
        self._origin_snapshot = self._comparison_state()

    def _comparison_state(self) -> tuple[object, ...]:
        """Keep an independent baseline even when a low-level reader mutates records."""
        return deepcopy(
            (
                self.source_metadata,
                self.unread_case_names,
                self.text,
                self.preprocessing_metadata,
                self.index_spans,
                self.extraction_metadata,
                self.citations,
                self.nodes,
                self.findings,
                self.passes,
                self.document_events,
            )
        )

    def __eq__(self, other: object) -> bool:
        """Compare persisted state; the private transition baseline is not data."""
        if not isinstance(other, Document) or type(self) is not type(other):
            return NotImplemented
        return all(getattr(self, name) == getattr(other, name) for name in type(self).model_fields)

    @model_serializer(mode="wrap")
    def _serialize_validated_state(self, handler: SerializerFunctionWrapHandler):
        """Refuse a dump if nested mutable records bypassed the operation API.

        Pydantic validates nested dataclasses when constructing or restoring a
        Document, but a caller can still mutate a citation record afterward.
        Rechecking before native serialization prevents writing a checkpoint
        that the same model would refuse to reload.
        """
        for citation in self.citations:
            citation.__post_init__()
        self._validate_document()
        self.assert_cumulative_successor(self)
        return handler(self)

    @property
    def active_citations(self) -> tuple[CitationRecord, ...]:
        """Records admitted for downstream work; withdrawn candidates stay in citations."""
        return tuple(item for item in self.citations if not item.withdrawn)

    def snapshot(self) -> "Document":
        """Return independent citation and evidence state for another stage."""
        return self.model_copy(deep=True)

    def evolve(self, **changes: object) -> "Document":
        """Return a validated checkpoint retaining every prior durable reading."""
        values = {name: getattr(self, name) for name in type(self).model_fields}
        values.update(changes)
        history = list(self.document_events)
        source = values["source_metadata"]
        unread = tuple(values["unread_case_names"])
        if source != self.source_metadata:
            history.append(SourceMetadataUpdated("source_metadata", source))  # type: ignore[arg-type]
        if unread != self.unread_case_names:
            history.append(UnreadCaseNamesUpdated("unread_case_names", unread))
        values["document_events"] = tuple(history)
        result = type(self)(**values)
        self.assert_cumulative_successor(result)
        return result

    def assert_cumulative_successor(self, later: "Document") -> None:
        """Require a later checkpoint to preserve both saved and live history."""
        if not isinstance(later, Document):
            raise TypeError("A Document stage must return a Document")
        self._assert_cumulative_state(self._origin_snapshot, later)
        # Low-level readers may append operations to a stage-local working
        # record before calling evolve. Those additions are already durable
        # readings and cannot disappear merely because they were made after
        # this Document instance was constructed.
        self._assert_cumulative_state(self._comparison_state(), later)

    @staticmethod
    def _assert_cumulative_state(earlier: tuple[object, ...], later: "Document") -> None:
        (
            old_source,
            old_unread,
            old_text,
            old_preprocessing,
            old_index_spans,
            old_extraction,
            old_citations,
            old_nodes,
            old_findings,
            old_passes,
            old_events,
        ) = earlier
        if (
            old_text != later.text
            or old_preprocessing != later.preprocessing_metadata
            or old_index_spans != later.index_spans
            or old_extraction != later.extraction_metadata
        ):
            raise ValueError("A stage cannot change the source text or extraction coordinate system")
        if not _is_prefix(old_events, later.document_events):
            raise ValueError("A stage cannot rewrite document-level event history")
        if not _is_prefix(old_nodes, later.nodes):
            raise ValueError("A stage cannot remove or rewrite document nodes")
        if not _is_prefix(old_findings, later.findings):
            raise ValueError("A stage cannot remove or rewrite document findings")
        if not _is_prefix(old_passes, later.passes):
            raise ValueError("A stage cannot remove or rewrite completed passes")
        previous = {record.citation_id: record for record in old_citations}
        current = {record.citation_id: record for record in later.citations}
        if not previous.keys() <= current.keys():
            raise ValueError("A stage cannot remove a created citation")
        for citation_id, record in previous.items():
            successor = current[citation_id]
            if not _is_prefix(record.operations, successor.operations):
                raise ValueError(f"A stage cannot rewrite citation {citation_id!r} operations")
            if not _is_prefix(record.trace, successor.trace):
                raise ValueError(f"A stage cannot rewrite citation {citation_id!r} trace")
        # The event stream must retain the prior document-level readings even
        # when the current projections change in this stage.
        if old_source != later.source_metadata and not any(
            isinstance(event, SourceMetadataUpdated) and event.after == later.source_metadata
            for event in later.document_events[len(old_events) :]
        ):
            raise ValueError("A source metadata change needs a document event")
        if old_unread != later.unread_case_names and not any(
            isinstance(event, UnreadCaseNamesUpdated) and event.after == later.unread_case_names
            for event in later.document_events[len(old_events) :]
        ):
            raise ValueError("An unread-name change needs a document event")

    @classmethod
    def from_source(
        cls,
        source: Path | str,
        *,
        rules: "ExtractionRules | None" = None,
    ) -> "Document":
        """Preprocess content or a source path, then create its locator document.

        A string is document content. Pass a :class:`~pathlib.Path` to load a
        file through the appropriate preprocessing backend. ``rules`` selects
        the extraction profile and is recorded on the resulting document.
        """
        from mellea_lrc.preprocessing import preprocess

        return cls.from_preprocessed(preprocess(source), rules=rules)

    @classmethod
    def from_plain_text(
        cls,
        text: str,
        *,
        source_path: str | None = None,
        rules: "ExtractionRules | None" = None,
    ) -> "Document":
        """Create an empty locator-stage document from already-extracted text.

        This is the convenient public constructor for text callers. Callers
        with a layout-aware source preprocess it first, then use
        :meth:`from_preprocessed` so the preprocessing provenance remains
        explicit.
        """
        from mellea_lrc.extraction.locator_stages import start_locator_document
        from mellea_lrc.preprocessing.plain_text import preprocess_plain_text_from_string

        return start_locator_document(
            preprocess_plain_text_from_string(text, source_path=source_path), rules=rules
        )

    @classmethod
    def from_preprocessed(
        cls,
        document: PreprocessedDocument,
        *,
        rules: "ExtractionRules | None" = None,
    ) -> "Document":
        """Create an empty locator-stage document from preprocessing output."""
        from mellea_lrc.extraction.locator_stages import start_locator_document

        result = start_locator_document(document, rules=rules)
        if not isinstance(result, cls):
            msg = f"Locator initialization produced {type(result).__name__}, not {cls.__name__}"
            raise TypeError(msg)
        return result

    @property
    def full_citations(self) -> tuple[CitationRecord, ...]:
        """Return only self-contained bibliographic citations."""
        return tuple(item for item in self.citations if is_full_citation(item.fields))

    @property
    def locators(self) -> tuple[Locator, ...]:
        """Every complete reporter or docket locator occurrence, including repeats.

        This reading is independent of root assignment, case context, and
        citation occurrence grouping. The locator text is sliced
        from this document so its span and text cannot drift apart.
        """
        return locator_layer(self.text, self.citations)

    @property
    def colocations(self) -> tuple[tuple[str, ...], ...]:
        """Co-located case locator occurrences, represented by citation ids.

        One citation remains a locator but does not form a one-member
        colocation. Groups include repeated occurrences as well as first-seen
        roots, which keeps grouping independent from root attribution.
        """
        return colocation_layer(self.citations)

    @model_validator(mode="after")
    def _validate_document(self) -> "Document":
        if self.schema_version != 21 or self.artifact_type != "document":
            raise ValueError("Unsupported Document schema or artifact type")
        if not self.document_events or not isinstance(self.document_events[0], DocumentCreated):
            raise ValueError("A Document must begin with a creation event")
        source = self.document_events[0].source_metadata
        unread = self.document_events[0].unread_case_names
        for event in self.document_events[1:]:
            if isinstance(event, DocumentCreated):
                raise ValueError("A Document cannot be created twice")
            if isinstance(event, SourceMetadataUpdated):
                if source == event.after:
                    raise ValueError("A source metadata update must change its value")
                source = event.after
            elif isinstance(event, UnreadCaseNamesUpdated):
                if unread == event.after:
                    raise ValueError("An unread-name update must change its value")
                unread = event.after
        if source != self.source_metadata or unread != self.unread_case_names:
            raise ValueError("Document-level state disagrees with its event history")
        citation_ids = [item.citation_id for item in self.citations]
        if any(not citation_id for citation_id in citation_ids):
            msg = "Extracted citation identifiers must not be empty"
            raise ValueError(msg)
        if WITHDRAWN_HEAD_ID in citation_ids:
            raise ValueError("The withdrawn head identifier is reserved for graph attachment")
        if len(citation_ids) != len(set(citation_ids)):
            msg = "Extracted citation identifiers must be unique within a document"
            raise ValueError(msg)

        known_ids = set(citation_ids)
        by_id = {item.citation_id: item for item in self.citations}
        document_node_ids = [node.node_id for node in self.nodes]
        if len(document_node_ids) != len(set(document_node_ids)):
            msg = "Document-level node identifiers must be unique"
            raise ValueError(msg)
        citation_node_ids = [node.node_id for item in self.citations for node in item.trace]
        if set(document_node_ids) & set(citation_node_ids):
            msg = "A node identifier cannot belong to both a citation and the document"
            raise ValueError(msg)
        known_node_ids = set(document_node_ids) | set(citation_node_ids)
        for item in self.citations:
            if not item.has_complete_history:
                msg = (
                    f"Citation {item.citation_id!r} must have a CREATE-backed operation "
                    "history before entering a Document"
                )
                raise ValueError(msg)
            if item.root_id is None and is_leaf(item.fields):
                msg = f"Leaf citation {item.citation_id!r} cannot enter a Document without a root"
                raise ValueError(msg)
            if item.root_id is not None and item.root_id != WITHDRAWN_HEAD_ID:
                target = by_id.get(item.root_id)
                if target is None or not target.is_root:
                    raise ValueError(
                        f"Citation {item.citation_id!r} points to a missing or non-root citation"
                    )
                if target.withdrawn and not item.withdrawn:
                    raise ValueError(f"Active citation {item.citation_id!r} points to a withdrawn root")
            if item.full_span.end > len(self.text):
                msg = f"Citation {item.citation_id!r} span exceeds document text"
                raise ValueError(msg)
            if item.locator_span.end > len(self.text):
                msg = f"Citation {item.citation_id!r} locator span exceeds document text"
                raise ValueError(msg)
            if item.locator_span.start < item.full_span.start or item.locator_span.end > item.full_span.end:
                msg = f"Citation {item.citation_id!r} locator span must be within its full span"
                raise ValueError(msg)
            if item.resolves_to is not None and (
                item.resolves_to not in known_ids or item.resolves_to == item.citation_id
            ):
                msg = f"Citation {item.citation_id!r} has invalid resolves_to={item.resolves_to!r}"
                raise ValueError(msg)
        for finding in self.findings:
            if finding.node_id is not None and finding.node_id not in known_node_ids:
                msg = f"Finding references unknown node {finding.node_id!r}"
                raise ValueError(msg)
        return self
