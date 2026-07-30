"use client";

import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";

type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

type ValidationNode = {
  node_type: string;
  node_id: string;
  status: string;
  outcome: string;
  depends_on: string[];
  [key: string]: JsonValue;
};

type CandidateSummaryRow = {
  provenance: string;
  candidate_index: number;
  assessment_node_id: string;
  outcome: string;
  extracted_citation: string | null;
  extracted_case_name: string | null;
  retrieved_case_name: string | null;
  case_name_outcome: string;
  case_name_evidence: string;
  extracted_year: string | null;
  retrieved_year: string | null;
  year_outcome: string;
  extracted_court_id: string | null;
  retrieved_court_id: string | null;
  court_outcome: string;
  docket_id: string | null;
  opinion_url?: string | null;
  pinpoint?: CitationSummaryPinpoint | null;
};

type CitationSummary = ValidationNode & {
  candidates: CandidateSummaryRow[];
  overall_outcome?: string | null;
  pinpoint_requires_review?: boolean | null;
};

type TextSpan = { start: number; end: number };

type CitationSummaryPinpoint = {
  node_id: string;
  status: string;
  outcome: string;
  reporter_citation: string | null;
  pin_cite: string | null;
  opinion_id: string | null;
  opinion_type: string | null;
  reporter_page_text: string | null;
  citing_context_span: TextSpan;
  citation_span: TextSpan | null;
  proposition: string | null;
  proposition_span: TextSpan | null;
  reasoning: string | null;
  evidence_quote: string | null;
  evidence_span: TextSpan | null;
  evidence_match_method: string | null;
  evidence_match_score: number | null;
  status_message: string | null;
  outcome_message: string | null;
  error: string | null;
};

type CompletePinpointEvidence = CitationSummaryPinpoint & {
  reporter_page_text: string;
  proposition: string;
  proposition_span: TextSpan;
  evidence_quote: string;
  evidence_span: TextSpan;
};

type EvidenceHighlight = {
  kind: "citation" | "proposition" | "evidence";
  label: string;
  span: TextSpan;
};

const EVIDENCE_EXCERPT_CHARS = 720;

type ValidatedCitation = {
  citation_id: string;
  nodes: ValidationNode[];
  aggregation?: CitationSummary | null;
};

type ValidatedDocumentPayload = {
  schema_version: 2;
  artifact_type: "validated_document";
  source: {
    text: string;
    citations: Array<{
      citation_id: string;
      span: { start: number; end: number };
      matched_text: string;
      citation: { citation_type: string; [key: string]: JsonValue };
    }>;
  };
  citations: ValidatedCitation[];
};

const STATUS_CLASS: Record<string, string> = {
  succeeded: "status-succeeded",
  skipped: "status-skipped",
  failed: "status-failed"
};

export default function Page() {
  const [document, setDocument] = useState<ValidatedDocumentPayload | null>(null);
  const [selectedCitationId, setSelectedCitationId] = useState<string | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeView, setActiveView] = useState<"graph" | "bibliography" | "document">("bibliography");

  const selectedCitation = document?.citations.find(
    (citation) => citation.citation_id === selectedCitationId
  );
  const selectedSourceCitation = document?.source.citations.find(
    (citation) => citation.citation_id === selectedCitationId
  );
  const selectedNode = selectedCitation?.nodes.find((node) => node.node_id === selectedNodeId);
  const graph = useMemo(
    () => buildGraph(selectedCitation?.nodes ?? []),
    [selectedCitation?.nodes]
  );

  function selectCitation(citationId: string) {
    const citation = document?.citations.find((item) => item.citation_id === citationId);
    setSelectedCitationId(citationId);
    setSelectedNodeId(citation?.nodes[0]?.node_id ?? null);
  }

  function loadFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const payload = parseValidatedDocument(reader.result);
        setDocument(payload);
        setSelectedCitationId(payload.citations[0]?.citation_id ?? null);
        setSelectedNodeId(payload.citations[0]?.nodes[0]?.node_id ?? null);
        setError(null);
      } catch (reason) {
        setDocument(null);
        setSelectedCitationId(null);
        setSelectedNodeId(null);
        setError(reason instanceof Error ? reason.message : "Could not read this JSON file.");
      }
    };
    reader.readAsText(file);
  }

  return (
    <main id="main-content" className="app-shell">
      <header className="header">
        <div>
          <p className="eyebrow">Citation validation workspace</p>
          <h1>mellea-LRC</h1>
          <p className="subtitle">Review a serialized citation-validation result without rerunning the pipeline.</p>
        </div>
        <label className="load-control">
          <span>Open result JSON</span>
          <input type="file" accept="application/json,.json" onChange={loadFile} />
        </label>
      </header>

      {error ? <p className="error" role="alert">{error}</p> : null}

      {!document ? (
        <section className="empty-state" aria-labelledby="empty-title">
          <h2 id="empty-title">No result loaded</h2>
          <p>Open JSON produced by <code>serialize_validated_document</code>. This workspace never runs validation or calls an API.</p>
        </section>
      ) : (
        <>
        <div className="view-tabs" role="tablist" aria-label="Artifact view">
          <button role="tab" aria-selected={activeView === "bibliography"} className={activeView === "bibliography" ? "active" : ""} onClick={() => setActiveView("bibliography")}>Citation Summary</button>
          <button role="tab" aria-selected={activeView === "document"} className={activeView === "document" ? "active" : ""} onClick={() => setActiveView("document")}>Document</button>
          <button role="tab" aria-selected={activeView === "graph"} className={activeView === "graph" ? "active" : ""} onClick={() => setActiveView("graph")}>Validation Viz</button>
        </div>
        <section className={`workspace${activeView !== "graph" ? " document-workspace" : ""}`} aria-label="Validation artifact">
          <aside className="citation-list" aria-label="Extracted citations">
            <div className="panel-heading">
              <h2>Extracted citations</h2>
              <span>{document.citations.length}</span>
            </div>
            {document.citations.map((citation) => {
              const source = document.source.citations.find((item) => item.citation_id === citation.citation_id);
              const selected = citation.citation_id === selectedCitationId;
              const overallOutcome = citation.aggregation?.overall_outcome;
              const pinpointRequiresReview =
                overallOutcome === "match" && citation.aggregation?.pinpoint_requires_review === true;
              return (
                <button
                  key={citation.citation_id}
                  className={`citation-button${selected ? " selected" : ""}`}
                  onClick={() => selectCitation(citation.citation_id)}
                  aria-pressed={selected}
                >
                  <strong>{source?.matched_text ?? citation.citation_id}</strong>
                  <span className="citation-button-meta">
                    <span>{citation.nodes.length} node{citation.nodes.length === 1 ? "" : "s"}</span>
                    {overallOutcome ? (
                      <span className="citation-summary-indicator">
                        {pinpointRequiresReview ? (
                          <span
                            className="pinpoint-review-flag"
                            role="img"
                            aria-label="Pinpoint citation requires review"
                            title="Pinpoint citation did not support this match"
                          >
                            !
                          </span>
                        ) : null}
                        <span className={`citation-outcome outcome outcome-${overallOutcome}`}>
                          {formatOutcome(overallOutcome)}
                        </span>
                      </span>
                    ) : null}
                  </span>
                </button>
              );
            })}
          </aside>

          <section className="graph-panel" aria-labelledby="graph-title">
            <div className="panel-heading">
              <div>
                <h2 id="graph-title">{activeView === "graph" ? "Validation progression" : activeView === "bibliography" ? "Citation Summary" : "Source document"}</h2>
                <p>{activeView === "graph" ? selectedSourceCitation?.matched_text ?? "Select a citation" : activeView === "bibliography" ? "Candidate assessment record" : "Citation locations in the source text"}</p>
              </div>
              <span className="artifact-version">schema {document.schema_version}</span>
            </div>
            {activeView === "graph" ? <StaticGraph
              graph={graph}
              selectedNodeId={selectedNodeId}
              onSelectNode={setSelectedNodeId}
            /> : activeView === "bibliography" ? <Bibliography aggregation={selectedCitation?.aggregation} documentText={document.source.text} /> : <DocumentViewer text={document.source.text} citations={document.source.citations} selectedCitationId={selectedCitationId} onSelectCitation={(citationId) => { selectCitation(citationId); }} />}
          </section>

          {activeView === "graph" ? <aside className="details" aria-live="polite">
            <div className="panel-heading"><h2>Validation node details</h2></div>
            {selectedNode ? (
              <>
                <p className={`status ${STATUS_CLASS[selectedNode.status] ?? ""}`}>{selectedNode.status} · {selectedNode.outcome}</p>
                <dl>
                  {Object.entries(selectedNode)
                    .filter(([key]) => !["node_type", "status", "outcome"].includes(key))
                    .map(([key, value]) => (
                      <div key={key}>
                        <dt>{key}</dt>
                        <dd><code>{formatValue(value)}</code></dd>
                      </div>
                    ))}
                </dl>
              </>
            ) : <p className="muted">Select a node to inspect its serialized fields.</p>}
          </aside> : null}
        </section>
        </>
      )}
    </main>
  );
}

function Bibliography({ aggregation, documentText }: { aggregation: CitationSummary | null | undefined; documentText: string }) {
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);
  const [openEvidence, setOpenEvidence] = useState<CompletePinpointEvidence | null>(null);
  useEffect(() => {
    setSelectedCandidateId(aggregation?.candidates[0]?.assessment_node_id ?? null);
    setOpenEvidence(null);
  }, [aggregation?.node_id]);

  if (!aggregation) {
    return <section className="bibliography-empty"><h3>No citation summary available</h3><p>This citation has not reached a route with a candidate assessment summary.</p></section>;
  }
  const candidates = aggregation.candidates ?? [];
  const selectedCandidate = candidates.find(
    (candidate) => candidate.assessment_node_id === selectedCandidateId
  ) ?? candidates[0];
  const selectedOpinionUrl = selectedCandidate ? opinionUrl(selectedCandidate) : null;
  const pinpoint = selectedCandidate?.pinpoint ?? null;
  const completePinpoint = isCompletePinpointEvidence(pinpoint) ? pinpoint : null;
  const presentedPinpoint = completePinpoint ?? (
    pinpoint?.outcome === "inconclusive" ? pinpoint : null
  );

  return <section className="bibliography" aria-label="Bibliographic comparison">
    {selectedCandidate ? <>
      <div className="candidate-tabs" role="tablist" aria-label="Candidate assessments">
        {candidates.map((candidate) => {
          const selected = candidate.assessment_node_id === selectedCandidate.assessment_node_id;
          return <button
            key={candidate.assessment_node_id}
            id={candidateTabId(candidate.assessment_node_id)}
            role="tab"
            aria-selected={selected}
            aria-controls="candidate-detail"
            className={`candidate-tab candidate-tab-${candidateConclusionClass(candidate.outcome)}${selected ? " active" : ""}`}
            onClick={() => setSelectedCandidateId(candidate.assessment_node_id)}
          >
            <span className="candidate-tab-outcome">{formatOutcome(candidate.outcome)}</span>
            <strong>{candidate.retrieved_case_name ?? `Candidate ${candidate.candidate_index}`}</strong>
          </button>;
        })}
      </div>
      <article
        id="candidate-detail"
        className="candidate-detail"
        role="tabpanel"
        aria-labelledby={candidateTabId(selectedCandidate.assessment_node_id)}
      >
        <header className="candidate-detail-header">
          <div>
            <p className="candidate-kicker">{provenanceLabel(selectedCandidate.provenance)} candidate</p>
            <h3 className="candidate-title">
              {selectedCandidate.retrieved_case_name ?? "Unnamed candidate"}
              {selectedOpinionUrl ? <a
                className="opinion-link"
                href={selectedOpinionUrl}
                target="_blank"
                rel="noreferrer"
              >
                View opinion <span aria-hidden="true">↗</span>
              </a> : null}
            </h3>
          </div>
          <span className={`outcome outcome-${selectedCandidate.outcome}`}>{selectedCandidate.outcome}</span>
        </header>
        <div className="candidate-fields">
          <CandidateField label="Case name">
            <Comparison extracted={selectedCandidate.extracted_case_name} retrieved={selectedCandidate.retrieved_case_name} outcome={selectedCandidate.case_name_outcome} evidence={selectedCandidate.case_name_evidence} />
          </CandidateField>
          <CandidateField label="Year">
            <Comparison extracted={selectedCandidate.extracted_year} retrieved={selectedCandidate.retrieved_year} outcome={selectedCandidate.year_outcome} />
          </CandidateField>
          <CandidateField label="Court">
            <Comparison extracted={selectedCandidate.extracted_court_id} retrieved={selectedCandidate.retrieved_court_id} outcome={selectedCandidate.court_outcome} />
          </CandidateField>
        </div>
        {presentedPinpoint ? <section className={`pinpoint-summary pinpoint-summary-${presentedPinpoint.outcome}`} aria-labelledby="pinpoint-title">
          <div className="pinpoint-conclusion">
            <p className="candidate-kicker">Pinpoint citation</p>
            <h3 id="pinpoint-title">{pinpointConclusion(presentedPinpoint.outcome)}</h3>
          </div>
          <dl className="pinpoint-preview">
            <div><dt>Citing proposition</dt><dd>{presentedPinpoint.proposition ?? "Not found here"}</dd></div>
            <div><dt>Opinion evidence</dt><dd>{presentedPinpoint.evidence_quote ?? "Not found here"}</dd></div>
            <div><dt>Assessment</dt><dd>{presentedPinpoint.reasoning ?? "Not found here"}</dd></div>
          </dl>
          {completePinpoint ? <button className="evidence-button" onClick={() => setOpenEvidence(completePinpoint)}>
            Compare highlighted sources
          </button> : null}
        </section> : <section className="pinpoint-unavailable">
          <strong>No pinpoint comparison available</strong>
          <p>{pinpoint?.outcome_message ?? pinpoint?.status_message ?? "This candidate did not produce both a grounded citing proposition and reporter-page evidence."}</p>
        </section>}
      </article>
    </> : <div className="candidate-empty"><p>No candidate assessments were serialized for this summary.</p></div>}
    {openEvidence ? <EvidenceModal evidence={openEvidence} documentText={documentText} onClose={() => setOpenEvidence(null)} /> : null}
  </section>;
}

function CandidateField({ label, children }: { label: string; children: React.ReactNode }) {
  return <section className="candidate-field"><h4>{label}</h4>{children}</section>;
}

function Comparison({ extracted, retrieved, outcome, evidence }: { extracted: string | null; retrieved: string | null; outcome: string; evidence?: string }) {
  return <div className="comparison">
    <dl>
      <div><dt>Extracted</dt><dd>{formatValue(extracted)}</dd></div>
      <div><dt>Retrieved</dt><dd>{formatValue(retrieved)}</dd></div>
    </dl>
    <small><span className={`outcome outcome-${outcome}`}>{outcome}</span>{evidence ? ` · ${evidence.replaceAll("_", " ")}` : ""}</small>
  </div>;
}

function EvidenceModal({ evidence, documentText, onClose }: { evidence: CompletePinpointEvidence; documentText: string; onClose: () => void }) {
  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [onClose]);

  return <div className="modal-backdrop" onMouseDown={(event) => {
    if (event.currentTarget === event.target) onClose();
  }}>
    <section className="evidence-modal" role="dialog" aria-modal="true" aria-labelledby="evidence-modal-title">
      <header className="evidence-modal-header">
        <div>
          <p className="candidate-kicker">Pinpoint evidence</p>
          <h2 id="evidence-modal-title">Citing document and cited opinion</h2>
        </div>
        <button className="modal-close" onClick={onClose} autoFocus>Close</button>
      </header>
      <div className="evidence-sources">
        <EvidenceSource
          label="Citing document"
          description="A focused excerpt around the citation and attributed proposition"
          text={documentText}
          contextSpan={evidence.citing_context_span}
          focusKind="proposition"
          highlights={[
            ...(evidence.citation_span ? [{ kind: "citation" as const, label: "Citation", span: evidence.citation_span }] : []),
            { kind: "proposition", label: "Attributed proposition", span: evidence.proposition_span }
          ]}
        />
        <EvidenceSource
          label="Cited opinion"
          description="The passage recovered from the cited reporter page"
          text={evidence.reporter_page_text}
          focusKind="evidence"
          highlights={[{ kind: "evidence", label: "Cited reporter-page evidence", span: evidence.evidence_span }]}
        />
      </div>
    </section>
  </div>;
}

function EvidenceSource({ label, description, text, highlights, focusKind, contextSpan }: { label: string; description: string; text: string; highlights: EvidenceHighlight[]; focusKind: EvidenceHighlight["kind"]; contextSpan?: TextSpan }) {
  const textPane = useRef<HTMLDivElement>(null);
  const focusHighlight = useRef<HTMLElement>(null);
  const focusSpan = highlights.find((highlight) => highlight.kind === focusKind)?.span ?? highlights[0]?.span;
  useEffect(() => {
    const pane = textPane.current;
    const markedText = focusHighlight.current;
    if (!pane || !markedText) return;
    const paneBounds = pane.getBoundingClientRect();
    const highlightBounds = markedText.getBoundingClientRect();
    pane.scrollTop +=
      highlightBounds.top -
      paneBounds.top -
      pane.clientHeight / 2 +
      highlightBounds.height / 2;
  }, [focusSpan?.start, focusSpan?.end]);
  const validHighlights = [...highlights]
    .filter((highlight) => validSpan(highlight.span, text.length))
    .sort((left, right) => left.span.start - right.span.start || left.span.end - right.span.end);
  const firstHighlight = validHighlights[0]?.span;
  const lastHighlight = validHighlights.at(-1)?.span;
  const availableContext = validSpan(contextSpan, text.length)
    ? contextSpan
    : {
        start: Math.max(0, (firstHighlight?.start ?? 0) - 500),
        end: Math.min(text.length, (lastHighlight?.end ?? text.length) + 500)
      };
  const context = balancedEvidenceExcerpt(availableContext, validHighlights);
  return <article className="evidence-source">
    <header>
      <h3>{label}</h3><p>{description}</p>
      <ul className="evidence-legend" aria-label={`${label} highlight legend`}>
        {validHighlights.map((highlight) => <li key={highlight.kind}>
          <span className={`evidence-legend-swatch evidence-highlight-${highlight.kind}`} aria-hidden="true" />
          {highlight.label}
        </li>)}
      </ul>
    </header>
    <div className="evidence-text" ref={textPane}>
      {context.start > 0 ? <span className="excerpt-boundary" aria-hidden="true">… </span> : null}
      {renderEvidenceExcerpt(text, context, validHighlights, focusKind, focusHighlight)}
      {context.end < text.length ? <span className="excerpt-boundary" aria-hidden="true"> …</span> : null}
    </div>
  </article>;
}

function renderEvidenceExcerpt(text: string, context: TextSpan, highlights: EvidenceHighlight[], focusKind: EvidenceHighlight["kind"], focusHighlight: React.RefObject<HTMLElement | null>) {
  let cursor = context.start;
  const fragments: React.ReactNode[] = [];
  const ordered = highlights
    .map((highlight) => ({ ...highlight, start: Math.max(context.start, highlight.span.start), end: Math.min(context.end, highlight.span.end) }))
    .filter((highlight) => highlight.end > highlight.start)
    .sort((left, right) => left.start - right.start || left.end - right.end);
  for (const highlight of ordered) {
    if (highlight.start < cursor) continue;
    if (highlight.start > cursor) fragments.push(text.slice(cursor, highlight.start));
    fragments.push(<mark key={`${highlight.kind}-${highlight.start}-${highlight.end}`} className={`evidence-highlight evidence-highlight-${highlight.kind}`} ref={highlight.kind === focusKind ? focusHighlight : undefined}>{text.slice(highlight.start, highlight.end)}</mark>);
    cursor = highlight.end;
  }
  if (cursor < context.end) fragments.push(text.slice(cursor, context.end));
  return fragments;
}

function balancedEvidenceExcerpt(availableContext: TextSpan, highlights: EvidenceHighlight[]): TextSpan {
  const first = highlights[0]?.span;
  const last = highlights.at(-1)?.span;
  if (!first || !last || availableContext.end - availableContext.start <= EVIDENCE_EXCERPT_CHARS) {
    return availableContext;
  }
  const focusStart = Math.max(availableContext.start, first.start);
  const focusEnd = Math.min(availableContext.end, last.end);
  const excerptLength = Math.max(EVIDENCE_EXCERPT_CHARS, focusEnd - focusStart);
  const availableEnd = availableContext.end - excerptLength;
  const desiredStart = focusStart - Math.floor((excerptLength - (focusEnd - focusStart)) / 2);
  const start = Math.max(availableContext.start, Math.min(desiredStart, availableEnd));
  return { start, end: start + excerptLength };
}

function provenanceLabel(provenance: string) {
  if (provenance === "recap" || provenance === "recap_search") return "RECAP";
  return "Opinion";
}

function candidateTabId(assessmentNodeId: string) {
  return `candidate-tab-${assessmentNodeId.replaceAll(/[^a-zA-Z0-9_-]/g, "-")}`;
}

function candidateConclusionClass(outcome: string) {
  if (outcome === "match") return "match";
  if (outcome === "mismatch") return "mismatch";
  return "possible-match";
}

function formatOutcome(outcome: string) {
  return outcome.replaceAll("_", " ");
}

function pinpointConclusion(outcome: string) {
  if (outcome === "supports") return "The cited evidence supports the proposition.";
  return "The cited evidence is inconclusive for the proposition.";
}

function opinionUrl(candidate: CandidateSummaryRow) {
  return candidate.provenance === "opinion" ? candidate.opinion_url ?? null : null;
}

function isCompletePinpointEvidence(
  pinpoint: CitationSummaryPinpoint | null
): pinpoint is CompletePinpointEvidence {
  return !!(
    pinpoint?.reporter_page_text &&
    pinpoint.proposition &&
    pinpoint.proposition_span &&
    pinpoint.evidence_quote &&
    pinpoint.evidence_span
  );
}

function validSpan(span: TextSpan | undefined, textLength: number): span is TextSpan {
  return !!span && span.start >= 0 && span.end >= span.start && span.end <= textLength;
}

function DocumentViewer({ text, citations, selectedCitationId, onSelectCitation }: { text: string; citations: ValidatedDocumentPayload["source"]["citations"]; selectedCitationId: string | null; onSelectCitation: (citationId: string) => void }) {
  const citationElements = useRef(new Map<string, HTMLButtonElement>());
  useEffect(() => {
    citationElements.current.get(selectedCitationId ?? "")?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [selectedCitationId]);
  const ordered = [...citations].sort((left, right) => left.span.start - right.span.start);
  let cursor = 0;
  return <article className="document-viewer" aria-label="Extracted document text">{ordered.flatMap((citation) => {
    const before = text.slice(cursor, citation.span.start);
    const content = text.slice(citation.span.start, citation.span.end);
    cursor = citation.span.end;
    return [before, <button key={citation.citation_id} ref={(element) => { if (element) citationElements.current.set(citation.citation_id, element); }} className={`citation-mark${citation.citation_id === selectedCitationId ? " selected" : ""}`} onClick={() => onSelectCitation(citation.citation_id)}>{content}</button>];
  }).concat(text.slice(cursor))}</article>;
}

function parseValidatedDocument(value: string | ArrayBuffer | null): ValidatedDocumentPayload {
  if (typeof value !== "string") throw new Error("The selected file was not readable as text.");
  const payload: unknown = JSON.parse(value);
  if (!isValidatedDocument(payload)) {
    throw new Error("Expected a schema 2 validated_document serialization artifact.");
  }
  return payload;
}

function isValidatedDocument(value: unknown): value is ValidatedDocumentPayload {
  if (!value || typeof value !== "object") return false;
  const payload = value as Record<string, unknown>;
  return payload.schema_version === 2 && payload.artifact_type === "validated_document" && Array.isArray(payload.citations) && !!payload.source;
}

type PositionedNode = { node: ValidationNode; x: number; y: number };
type StaticGraphLayout = { nodes: PositionedNode[]; edges: Array<{ source: PositionedNode; target: PositionedNode }>; width: number; height: number };

const GRAPH_NODE_WIDTH = 200;
const GRAPH_NODE_HEIGHT = 76;
const GRAPH_COLUMN_GAP = 24;
const GRAPH_ROW_GAP = 56;
const GRAPH_PADDING = 24;

function buildGraph(validationNodes: ValidationNode[]): StaticGraphLayout {
  const levelById = new Map<string, number>();
  const byId = new Map(validationNodes.map((node) => [node.node_id, node]));
  const levelFor = (node: ValidationNode): number => {
    const known = levelById.get(node.node_id);
    if (known !== undefined) return known;
    const dependencies = node.depends_on
      .map((id) => byId.get(id))
      .filter((item): item is ValidationNode => item !== undefined);
    const level = dependencies.length ? Math.max(...dependencies.map(levelFor)) + 1 : 0;
    levelById.set(node.node_id, level);
    return level;
  };
  validationNodes.forEach(levelFor);
  const nodesByLevel = new Map<number, ValidationNode[]>();
  validationNodes.forEach((node) => {
    const level = levelById.get(node.node_id) ?? 0;
    nodesByLevel.set(level, [...(nodesByLevel.get(level) ?? []), node]);
  });
  const widestLevel = Math.max(1, ...[...nodesByLevel.values()].map((nodes) => nodes.length));
  const width =
    widestLevel * GRAPH_NODE_WIDTH +
    (widestLevel - 1) * GRAPH_COLUMN_GAP +
    GRAPH_PADDING * 2;
  const positions = new Map<string, PositionedNode>();
  [...nodesByLevel.entries()].forEach(([level, nodes]) => {
    const rowWidth =
      nodes.length * GRAPH_NODE_WIDTH + (nodes.length - 1) * GRAPH_COLUMN_GAP;
    const initialX = (width - rowWidth) / 2;
    nodes.forEach((node, index) => {
      positions.set(node.node_id, {
        node,
        x: initialX + index * (GRAPH_NODE_WIDTH + GRAPH_COLUMN_GAP),
        y: GRAPH_PADDING + level * (GRAPH_NODE_HEIGHT + GRAPH_ROW_GAP),
      });
    });
  });
  const nodes = validationNodes.map((node) => positions.get(node.node_id) as PositionedNode);
  const edges = nodes.flatMap((target) => target.node.depends_on.flatMap((dependency) => {
    const source = positions.get(dependency);
    return source ? [{ source, target }] : [];
  }));
  return {
    nodes,
    edges,
    width,
    height:
      GRAPH_PADDING * 2 +
      (Math.max(0, ...levelById.values()) + 1) * GRAPH_NODE_HEIGHT +
      Math.max(0, ...levelById.values()) * GRAPH_ROW_GAP,
  };
}

function StaticGraph({ graph, selectedNodeId, onSelectNode }: { graph: StaticGraphLayout; selectedNodeId: string | null; onSelectNode: (nodeId: string) => void }) {
  return (
    <div className="graph static-graph" aria-label="Validation node graph">
      <div className="static-graph-inner" style={{ width: graph.width, height: graph.height }}>
        <svg className="graph-edges" viewBox={`0 0 ${graph.width} ${graph.height}`} aria-hidden="true">
          <defs><marker id="graph-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" /></marker></defs>
          {graph.edges.map(({ source, target }) => {
            const sourceX = source.x + GRAPH_NODE_WIDTH / 2;
            const sourceY = source.y + GRAPH_NODE_HEIGHT;
            const targetX = target.x + GRAPH_NODE_WIDTH / 2;
            const targetY = target.y;
            return <path key={`${source.node.node_id}-${target.node.node_id}`} d={`M ${sourceX} ${sourceY} C ${sourceX} ${sourceY + 34}, ${targetX} ${targetY - 34}, ${targetX} ${targetY}`} markerEnd="url(#graph-arrow)" />;
          })}
        </svg>
        {graph.nodes.map(({ node, x, y }) => (
          <button key={node.node_id} className={`graph-node ${STATUS_CLASS[node.status] ?? ""}${node.node_id === selectedNodeId ? " selected" : ""}`} style={{ left: x, top: y }} onClick={() => onSelectNode(node.node_id)} aria-pressed={node.node_id === selectedNodeId}>
            <strong>{node.node_type}</strong><span>{node.outcome}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function formatValue(value: JsonValue): string {
  return typeof value === "string" ? value : JSON.stringify(value);
}
