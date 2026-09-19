"""Extraction result types."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from mellea_lrc.core.case_names import CaseName
from mellea_lrc.core.citations import is_full_citation
from mellea_lrc.core.findings import Finding
from mellea_lrc.core.pin_cites import PinCitePages
from mellea_lrc.core.record import CitationRecord, Node
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.reading.relaxation import Relaxation
from mellea_lrc.extraction.structure.locator_layers import (
    Locator,
    colocation_layer,
    locator_layer,
)
from mellea_lrc.preprocessing.types import PreprocessedDocument

if TYPE_CHECKING:
    from mellea_lrc.extraction.rules import ExtractionRules


class ExtractionBackend(str, Enum):
    """Engine that produced the extracted citations."""

    EYECITE = "eyecite"
    MELLEA = "mellea"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class ExtractionMetadata:
    """Provenance for the extraction stage."""

    backend: ExtractionBackend = ExtractionBackend.EYECITE
    backend_version: str | None = None
    relaxation: Relaxation = Relaxation.FULL
    """Which tokenizer read the text.

    Two levels disagree about whether a given citation is there at all, so a
    document that does not say which one ran cannot be compared with another.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class Document(PreprocessedDocument):
    """One document and everything the pipeline has read in it.

    Not the output of a stage. Every stage appends to this object -- extraction
    adds the roots and then the leaves, validation adds nodes to them and
    withdraws what reaches nothing -- and each hands on the same document with
    more written on it rather than an artifact of its own kind wrapping the one
    before. See `docs/Document.md`.

    What is here and not on a citation is here because it cannot be on one.
    `text` is the coordinate space every span indexes, and per-citation copies
    of it could disagree. `findings` are true of the document and of no citation
    in it. `passes` is what has run, which is the only way a consumer can tell a
    document whose leaves have not grown from one that writes no short forms.
    """

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
    invariant. See :mod:`mellea_lrc.core.findings`.
    """
    passes: tuple[str, ...] = ()
    """Which passes have run over this document, in order.

    `("roots",)` is the first growth and is **not a reading of the document's
    citations** -- it holds what states a complete identifier and no leaf of any
    kind. `("roots", "leaves")` is the whole of what the filing writes. A
    consumer that does not check this cannot tell the two apart.
    """
    extraction_metadata: ExtractionMetadata

    @property
    def active_citations(self) -> tuple[CitationRecord, ...]:
        """Records admitted for downstream work; withdrawn candidates stay in citations."""
        return tuple(item for item in self.citations if not item.withdrawn)

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

    def serialize(self) -> dict[str, object]:
        """Return the complete JSON-ready document checkpoint.

        This is the outer serialization boundary. Stage code passes a
        ``Document`` directly; callers saving or handing it to another process
        use this method instead of importing serializer internals.
        """
        from mellea_lrc.serialization.document import serialize_document

        return serialize_document(self)

    @classmethod
    def from_serialized(cls, payload: Mapping[str, object]) -> "Document":
        """Recover a document checkpoint produced by :meth:`serialize`.

        Python reserves ``from``, so the paired constructor is named
        ``from_serialized`` rather than ``Document.from(...)``.
        """
        from mellea_lrc.serialization.document import deserialize_document

        document = deserialize_document(payload)
        if not isinstance(document, cls):
            msg = f"Serialized document produced {type(document).__name__}, not {cls.__name__}"
            raise TypeError(msg)
        return document

    @property
    def full_citations(self) -> tuple[CitationRecord, ...]:
        """Return only self-contained bibliographic citations."""
        return tuple(item for item in self.citations if is_full_citation(item.stated))

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

    def __post_init__(self) -> None:
        PreprocessedDocument.__post_init__(self)
        citation_ids = [item.citation_id for item in self.citations]
        if any(not citation_id for citation_id in citation_ids):
            msg = "Extracted citation identifiers must not be empty"
            raise ValueError(msg)
        if len(citation_ids) != len(set(citation_ids)):
            msg = "Extracted citation identifiers must be unique within a document"
            raise ValueError(msg)

        known_ids = set(citation_ids)
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

    def observe(self, node: Node) -> Node:
        """Add one document-level reading to the trace and return it."""
        if node.node_id in {seen.node_id for item in self.citations for seen in item.trace}:
            msg = f"Document-level node identifier {node.node_id!r} already belongs to a citation"
            raise ValueError(msg)
        if node.node_id not in {seen.node_id for seen in self.nodes}:
            object.__setattr__(self, "nodes", (*self.nodes, node))
        return node
