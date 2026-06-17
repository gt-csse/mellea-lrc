"use client";

import {
  AlertCircle,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  CheckCircle2,
  FileText,
  Loader2,
  Search,
  Upload
} from "lucide-react";
import { ChangeEvent, DragEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";

type ValidationPayload = {
  citation_id: string;
  locator: string | null;
  status: string;
  source: string;
  message: string;
  case_names: string[];
  lookup_status: number | null;
  lookup_cache: string | null;
  lookup_key: string | null;
  error_message: string | null;
  limit_detail: Record<string, unknown> | null;
  clusters: CourtListenerCluster[];
};

type CourtListenerCluster = Record<string, unknown>;

type ReviewCitation = {
  id: string;
  start: number;
  end: number;
  matched_text: string;
  kind: string;
  fields: Record<string, string | number | boolean>;
  resolves_to: string | null;
  validation: ValidationPayload | null;
};

type ReviewResult = {
  document: {
    text: string;
    source_path: string | null;
    source_format: string;
    backend: string;
  };
  citations: ReviewCitation[];
  stats: Record<string, number>;
};

type CitationStatus = "found" | "ambiguous" | "not_found" | "not_checked" | "other";
type CitationFilter = "all" | "found" | "ambiguous" | "not_found" | "all_citations";
type RenderCitation = ReviewCitation & {
  highlightStart: number;
  highlightEnd: number;
};

const exampleText = "Brown v. Board, 347 U.S. 483 (1954). See also Roe v. Wade, 410 U.S. 113.";
const citationFilters: Array<{ label: string; value: CitationFilter }> = [
  { label: "All", value: "all" },
  { label: "Found", value: "found" },
  { label: "Ambiguous", value: "ambiguous" },
  { label: "Not found", value: "not_found" },
  { label: "All citations", value: "all_citations" }
];

export default function Home() {
  const [text, setText] = useState(exampleText);
  const [file, setFile] = useState<File | null>(null);
  const [validate, setValidate] = useState(true);
  const [result, setResult] = useState<ReviewResult | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isInputCollapsed, setIsInputCollapsed] = useState(false);
  const [citationFilter, setCitationFilter] = useState<CitationFilter>("all");
  const [clusterIndexes, setClusterIndexes] = useState<Record<string, number>>({});
  const [isDetailsExpanded, setIsDetailsExpanded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const documentPaneRef = useRef<HTMLElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const primaryCitations = useMemo(
    () =>
      result
        ? visibleCitationSpans(
            result.document.text,
            result.citations,
            selectedId,
            false
          )
        : [],
    [result, selectedId]
  );

  const allCitations = useMemo(
    () =>
      result
        ? visibleCitationSpans(result.document.text, result.citations, selectedId, true)
        : [],
    [result, selectedId]
  );

  const renderCitations = useMemo(
    () => (citationFilter === "all_citations" ? allCitations : primaryCitations),
    [allCitations, citationFilter, primaryCitations]
  );

  const filteredCitations = useMemo(
    () =>
      renderCitations.filter((citation) => {
        if (citationFilter === "all_citations") {
          return true;
        }
        if (!isFullCaseCitation(citation)) {
          return false;
        }
        return citationFilter === "all" || citationStatus(citation) === citationFilter;
      }),
    [citationFilter, renderCitations]
  );

  const fullCaseCitations = useMemo(
    () => primaryCitations.filter(isFullCaseCitation),
    [primaryCitations]
  );
  const statusCounts = useMemo(() => citationStatusCounts(fullCaseCitations), [fullCaseCitations]);

  const selectedCitation = useMemo(
    () => result?.citations.find((citation) => citation.id === selectedId) ?? null,
    [result, selectedId]
  );
  const selectedClusterIndex = selectedCitation ? clusterIndexes[selectedCitation.id] ?? 0 : 0;

  useEffect(() => {
    if (!result) {
      return;
    }
    if (selectedId && filteredCitations.some((citation) => citation.id === selectedId)) {
      return;
    }
    setSelectedId(filteredCitations[0]?.id ?? null);
  }, [filteredCitations, result, selectedId]);

  useEffect(() => {
    if (!selectedId) {
      return;
    }

    const pane = documentPaneRef.current;
    const target = document.getElementById(`citation-${selectedId}`);
    if (!pane || !target) {
      return;
    }

    const paneRect = pane.getBoundingClientRect();
    const targetRect = target.getBoundingClientRect();
    const centeredOffset =
      targetRect.top - paneRect.top - pane.clientHeight / 2 + targetRect.height / 2;

    pane.scrollTo({
      top: pane.scrollTop + centeredOffset,
      behavior: "auto"
    });
  }, [selectedId]);

  async function runReview() {
    setIsLoading(true);
    setError(null);

    try {
      const response = file ? await reviewDocument(file, validate) : await reviewText(text, validate);
      setResult(response);
      setSelectedId(response.citations[0]?.id ?? null);
      setCitationFilter("all");
      setClusterIndexes({});
      setIsInputCollapsed(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Review failed");
    } finally {
      setIsLoading(false);
    }
  }

  function selectCitation(citationId: string) {
    setSelectedId(citationId);
  }

  function selectCourtListenerCandidate(citationId: string, clusterIndex: number) {
    setClusterIndexes((current) => ({
      ...current,
      [citationId]: clusterIndex
    }));
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    setFile(event.target.files?.[0] ?? null);
    setIsInputCollapsed(false);
  }

  function onDrop(event: DragEvent<HTMLButtonElement>) {
    event.preventDefault();
    setFile(event.dataTransfer.files?.[0] ?? null);
    setIsInputCollapsed(false);
  }

  const citationCount = result?.stats.citation_spans ?? 0;
  const foundCount = result?.stats.found;
  const sourceLabel = file?.name ?? result?.document.source_path ?? `${text.trim().length || 0} chars`;

  return (
    <main className={`app-shell${isDetailsExpanded ? " details-expanded" : ""}`}>
      <header className="topbar">
        <div>
          <p className="eyebrow">Mellea LRC</p>
          <h1>Citation review</h1>
        </div>
        <div className="metric-strip" aria-label="Review statistics">
          <Metric label="Citations" value={citationCount} />
          <Metric label="Found" value={foundCount ?? "-"} />
          <Metric label="Chars" value={result?.stats.chars ?? "-"} />
        </div>
      </header>

      <section
        className={`ingest-panel${isInputCollapsed ? " collapsed" : ""}`}
        aria-label="Document input"
      >
        <div className="ingest-header">
          <div>
            <p className="eyebrow">Input</p>
            <strong>{sourceLabel}</strong>
          </div>
          <button
            className="icon-action"
            type="button"
            onClick={() => setIsInputCollapsed((value) => !value)}
            aria-label={isInputCollapsed ? "Expand input panel" : "Collapse input panel"}
          >
            {isInputCollapsed ? <ChevronDown size={18} /> : <ChevronUp size={18} />}
          </button>
        </div>

        {isInputCollapsed ? null : (
          <>
            <label className="text-input">
              <span>Plain text</span>
              <textarea
                value={text}
                onChange={(event) => {
                  setText(event.target.value);
                  if (file) {
                    setFile(null);
                  }
                }}
                spellCheck={false}
              />
            </label>

            <button
              className={`drop-zone${file ? " has-file" : ""}`}
              type="button"
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(event) => event.preventDefault()}
              onDrop={onDrop}
            >
              <Upload size={20} aria-hidden="true" />
              <span>{file ? file.name : "Drop document"}</span>
              <small>PDF, DOCX, TXT, HTML, MD</small>
            </button>
          </>
        )}
        <input ref={fileInputRef} className="hidden-input" type="file" onChange={onFileChange} />

        <div className="ingest-actions">
          <label className="validation-toggle">
            <input
              checked={validate}
              onChange={(event) => setValidate(event.target.checked)}
              type="checkbox"
            />
            <span>Validate</span>
          </label>

          <button
            className="primary-action"
            disabled={isLoading || (!file && !text.trim())}
            onClick={runReview}
            type="button"
          >
            {isLoading ? <Loader2 className="spin" size={18} /> : <Search size={18} />}
            <span>{isLoading ? "Checking" : "Review"}</span>
          </button>
        </div>
      </section>

      {error ? (
        <div className="error-banner" role="alert">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      ) : null}

      <section className="workspace" aria-label="Citation review workspace">
        <article ref={documentPaneRef} className="document-pane" aria-label="Document viewer">
          {result ? (
            <div className="document-text">
              {renderHighlightedDocument(
                result.document.text,
                filteredCitations,
                selectedId,
                selectCitation
              )}
            </div>
          ) : (
            <div className="empty-state">
              <FileText size={28} aria-hidden="true" />
              <span>Run a review to inspect citation spans.</span>
            </div>
          )}
        </article>

        <aside className="citation-rail" aria-label="Extracted citations">
          <div className="rail-header">
            <div>
              <h2>Extracted citations</h2>
              <p>{filteredCitations.length} shown</p>
            </div>
            <span>{renderCitations.length}</span>
          </div>
          <div className="rail-controls" aria-label="Citation display controls">
            <label className="status-filter">
              <span>Scope</span>
              <select
                value={citationFilter}
                onChange={(event) => setCitationFilter(event.target.value as CitationFilter)}
                aria-label="Citation scope filter"
              >
              {citationFilters.map((filter) => (
                <option key={filter.value} value={filter.value}>
                  {filterLabel(
                    filter,
                    statusCounts,
                    fullCaseCitations.length,
                    allCitations.length
                  )}
                </option>
              ))}
              </select>
            </label>
          </div>
          <div className="citation-list">
            {filteredCitations.length ? (
              filteredCitations.map((citation) => (
                <button
                  className={`citation-row${citation.id === selectedId ? " selected" : ""}`}
                  key={citation.id}
                  onClick={() => selectCitation(citation.id)}
                  type="button"
                >
                  <span className="citation-row-main">{citation.matched_text}</span>
                  <span className="citation-row-meta">
                    {citation.kind}
                    {citation.validation ? ` · ${citation.validation.status}` : ""}
                  </span>
                </button>
              ))
            ) : (
              <p className="rail-empty">No citations match this view.</p>
            )}
          </div>
        </aside>
      </section>

      <section
        className={`details-panel${isDetailsExpanded ? " expanded" : ""}`}
        aria-label="Citation details"
      >
        {selectedCitation ? (
          <>
            <div className="details-heading">
              <div>
                <p className="eyebrow">Selected citation</p>
                <h2>{selectedCitation.matched_text}</h2>
              </div>
              <div className="details-actions">
                <StatusBadge validation={selectedCitation.validation} />
                <button
                  className="icon-action"
                  type="button"
                  onClick={() => setIsDetailsExpanded((value) => !value)}
                  aria-expanded={isDetailsExpanded}
                  aria-label={
                    isDetailsExpanded
                      ? "Collapse citation checking panel"
                      : "Expand citation checking panel"
                  }
                >
                  {isDetailsExpanded ? <ChevronDown size={18} /> : <ChevronUp size={18} />}
                </button>
              </div>
            </div>
            <div className="details-grid">
              <BibliographicComparison
                citation={selectedCitation}
                clusterIndex={selectedClusterIndex}
                onClusterChange={selectCourtListenerCandidate}
              />
              <ValidationDetails validation={selectedCitation.validation} />
            </div>
          </>
        ) : (
          <div className="details-empty">Select a citation to inspect fields and validation.</div>
        )}
      </section>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StatusBadge({ validation }: { validation: ValidationPayload | null }) {
  if (!validation) {
    return <span className="status-badge neutral">Not checked</span>;
  }

  const isFound = validation.status === "found";
  return (
    <span className={`status-badge ${isFound ? "found" : "attention"}`}>
      {isFound ? <CheckCircle2 size={15} /> : <AlertCircle size={15} />}
      {validation.status.replaceAll("_", " ")}
    </span>
  );
}

function BibliographicComparison({
  citation,
  clusterIndex,
  onClusterChange
}: {
  citation: ReviewCitation;
  clusterIndex: number;
  onClusterChange: (citationId: string, clusterIndex: number) => void;
}) {
  const clusters = comparableCourtListenerClusters(citation.validation);
  const safeClusterIndex = clusters.length
    ? Math.min(Math.max(clusterIndex, 0), clusters.length - 1)
    : 0;
  const cluster = clusters[safeClusterIndex] ?? null;
  const rows = bibliographicRows(citation, cluster);
  const hasMultipleClusters = clusters.length > 1;

  function changeCluster(direction: -1 | 1) {
    if (!hasMultipleClusters) {
      return;
    }
    const nextIndex = (safeClusterIndex + direction + clusters.length) % clusters.length;
    onClusterChange(citation.id, nextIndex);
  }

  return (
    <div className="detail-group comparison-group">
      <div className="detail-title-row">
        <h3>Bibliographic comparison</h3>
        <span>
          {clusters.length
            ? `${clusters.length} CourtListener candidate${clusters.length === 1 ? "" : "s"}`
            : "No CourtListener candidate"}
        </span>
      </div>
      <div className="comparison-table" role="table" aria-label="Bibliographic field comparison">
        <div className="comparison-header" role="row">
          <span role="columnheader">Field</span>
          <span role="columnheader">Extracted</span>
          <span className="courtlistener-column-header" role="columnheader">
            <span>CourtListener</span>
            {hasMultipleClusters ? (
              <span className="case-switcher" aria-label="CourtListener candidate selector">
                <button
                  aria-label="Show previous CourtListener candidate"
                  type="button"
                  onClick={() => changeCluster(-1)}
                >
                  <ChevronLeft size={14} aria-hidden="true" />
                </button>
                <strong>
                  {safeClusterIndex + 1}/{clusters.length}
                </strong>
                <button
                  aria-label="Show next CourtListener candidate"
                  type="button"
                  onClick={() => changeCluster(1)}
                >
                  <ChevronRight size={14} aria-hidden="true" />
                </button>
              </span>
            ) : null}
          </span>
        </div>
        {rows.map((row) => (
          <div className="comparison-row" key={row.label} role="row">
            <span role="cell">{row.label}</span>
            <span role="cell">{formatValue(row.extracted)}</span>
            <span role="cell">{formatValue(row.courtListener)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ValidationDetails({ validation }: { validation: ValidationPayload | null }) {
  if (!validation) {
    return (
      <div className="detail-group">
        <h3>Validation status</h3>
        <dl>
          <div>
            <dt>Status</dt>
            <dd>Not requested</dd>
          </div>
          <div>
            <dt>Message</dt>
            <dd>Run with validation enabled to query CourtListener.</dd>
          </div>
        </dl>
      </div>
    );
  }

  return (
    <div className="detail-group">
      <h3>Validation status</h3>
      <dl>
        <div>
          <dt>Status</dt>
          <dd>{validation.status.replaceAll("_", " ")}</dd>
        </div>
        <div>
          <dt>Message</dt>
          <dd>{validation.message}</dd>
        </div>
        <div>
          <dt>Source</dt>
          <dd>{validation.source}</dd>
        </div>
        {validation.locator ? (
          <div>
            <dt>Locator</dt>
            <dd>{validation.locator}</dd>
          </div>
        ) : null}
        {validation.case_names.length ? (
          <div>
            <dt>Case names</dt>
            <dd>{validation.case_names.join(", ")}</dd>
          </div>
        ) : null}
        <div>
          <dt>Lookup status</dt>
          <dd>{formatValue(validation.lookup_status)}</dd>
        </div>
        {validation.lookup_cache ? (
          <div>
            <dt>Cache</dt>
            <dd>{validation.lookup_cache}</dd>
          </div>
        ) : null}
        {validation.lookup_key ? (
          <div>
            <dt>Lookup key</dt>
            <dd>{validation.lookup_key}</dd>
          </div>
        ) : null}
        {validation.error_message ? (
          <div>
            <dt>Error</dt>
            <dd>{validation.error_message}</dd>
          </div>
        ) : null}
        {validation.limit_detail ? (
          <div>
            <dt>Limit detail</dt>
            <dd>{JSON.stringify(validation.limit_detail)}</dd>
          </div>
        ) : null}
        <div>
          <dt>Clusters</dt>
          <dd>{validation.clusters.length}</dd>
        </div>
      </dl>
    </div>
  );
}

function bibliographicRows(citation: ReviewCitation, cluster: CourtListenerCluster | null) {
  const citationLocator = citation.validation?.locator ?? citation.matched_text;
  const extractedLocatorParts = splitLocator(citationLocator);
  const courtListenerLocator = cluster ? citation.validation?.locator : null;
  const courtListenerLocatorParts = splitLocator(courtListenerLocator);
  const extractedCaseName = caseNameFromFields(citation.fields) || citation.matched_text;
  const courtListenerCaseName = readString(cluster, ["case_name", "caseName"]);
  const courtListenerCourt = readString(cluster, ["court", "court_id", "courtId"]);
  const courtListenerDate = readString(cluster, ["date_filed", "dateFiled"]);
  const courtListenerUrl = readString(cluster, ["absolute_url", "absoluteUrl", "resource_uri"]);
  const knownFieldKeys = new Set([
    "plaintiff",
    "defendant",
    "volume",
    "reporter",
    "page",
    "year",
    "court"
  ]);
  const extraFieldRows = Object.entries(citation.fields)
    .filter(([key]) => !knownFieldKeys.has(key))
    .map(([key, value]) => ({
      label: key.replaceAll("_", " "),
      extracted: value,
      courtListener: null
    }));

  return [
    {
      label: "Case name",
      extracted: extractedCaseName,
      courtListener: courtListenerCaseName
    },
    {
      label: "Locator",
      extracted: citationLocator,
      courtListener: courtListenerLocator
    },
    {
      label: "Volume",
      extracted: citation.fields.volume ?? extractedLocatorParts.volume,
      courtListener: courtListenerLocatorParts.volume
    },
    {
      label: "Reporter",
      extracted: citation.fields.reporter ?? extractedLocatorParts.reporter,
      courtListener: courtListenerLocatorParts.reporter
    },
    {
      label: "Page",
      extracted: citation.fields.page ?? extractedLocatorParts.page,
      courtListener: courtListenerLocatorParts.page
    },
    {
      label: "Year",
      extracted: citation.fields.year,
      courtListener: courtListenerDate?.slice(0, 4)
    },
    {
      label: "Court",
      extracted: citation.fields.court,
      courtListener: courtListenerCourt
    },
    {
      label: "URL",
      extracted: null,
      courtListener: courtListenerUrl
    },
    ...extraFieldRows
  ];
}

function caseNameFromFields(fields: Record<string, unknown>) {
  const plaintiff = stringField(fields.plaintiff);
  const defendant = stringField(fields.defendant);
  if (plaintiff && defendant) {
    return `${plaintiff} v. ${defendant}`;
  }
  return plaintiff || defendant || null;
}

function splitLocator(locator: string | null | undefined) {
  const parts = locator?.trim().split(/\s+/) ?? [];
  if (parts.length < 3) {
    return { volume: null, reporter: null, page: null };
  }
  return {
    volume: parts[0],
    reporter: parts.slice(1, -1).join(" "),
    page: parts.at(-1) ?? null
  };
}

function readString(record: CourtListenerCluster | null, keys: string[]) {
  if (!record) {
    return null;
  }
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value.trim()) {
      return value;
    }
    if (typeof value === "number") {
      return String(value);
    }
  }
  return null;
}

function stringField(value: unknown) {
  return typeof value === "string" && value.trim() ? value : null;
}

function hasDisplayValue(value: unknown) {
  return value !== null && value !== undefined && value !== "";
}

function formatValue(value: unknown) {
  if (!hasDisplayValue(value)) {
    return "-";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function renderHighlightedDocument(
  text: string,
  citations: RenderCitation[],
  selectedId: string | null,
  onSelect: (citationId: string) => void
) {
  const nodes: ReactNode[] = [];
  const sorted = [...citations].sort((left, right) => left.highlightStart - right.highlightStart);
  let cursor = 0;

  sorted.forEach((citation) => {
    if (
      citation.highlightStart < cursor ||
      citation.highlightEnd > text.length ||
      citation.highlightStart >= citation.highlightEnd
    ) {
      return;
    }

    if (citation.highlightStart > cursor) {
      nodes.push(<span key={`text-${cursor}`}>{text.slice(cursor, citation.highlightStart)}</span>);
    }

    nodes.push(
      ...renderCitationMark(
        text.slice(citation.highlightStart, citation.highlightEnd),
        citation,
        citation.id === selectedId,
        onSelect
      )
    );
    cursor = citation.highlightEnd;
  });

  if (cursor < text.length) {
    nodes.push(<span key={`text-${cursor}`}>{text.slice(cursor)}</span>);
  }

  return nodes;
}

function renderCitationMark(
  citationText: string,
  citation: RenderCitation,
  isSelected: boolean,
  onSelect: (citationId: string) => void
) {
  const chunks = citationText.split(/([^\S\r\n]*[\r\n]+[^\S\r\n]*)/);
  let firstTextChunk = true;

  return chunks.map((chunk, index) => {
    if (!chunk) {
      return null;
    }
    if (/^[^\S\r\n]*[\r\n]+[^\S\r\n]*$/.test(chunk)) {
      return <span key={`${citation.id}-space-${index}`}>{chunk}</span>;
    }

    const id = firstTextChunk ? `citation-${citation.id}` : undefined;
    firstTextChunk = false;

    return (
      <button
        className={`citation-mark${isSelected ? " active" : ""}`}
        id={id}
        key={`${citation.id}-mark-${index}`}
        onClick={() => onSelect(citation.id)}
        type="button"
      >
        {chunk}
      </button>
    );
  });
}

function visibleCitationSpans(
  text: string,
  citations: ReviewCitation[],
  selectedId: string | null,
  includeAll: boolean
) {
  const candidates = citations.map((citation) => renderCitation(text, citation)).filter(isRenderCitation);
  if (includeAll) {
    return candidates.sort((left, right) => left.highlightStart - right.highlightStart);
  }

  const selected = candidates.find((citation) => citation.id === selectedId);
  const sorted = candidates.sort(
    (left, right) =>
      left.highlightStart - right.highlightStart ||
      right.highlightEnd -
        right.highlightStart -
        (left.highlightEnd - left.highlightStart)
  );
  const visible: RenderCitation[] = [];

  sorted.forEach((citation) => {
    if (selected && citation.id !== selected.id && renderSpansOverlap(citation, selected)) {
      return;
    }
    if (!visible.some((visibleCitation) => renderSpansOverlap(visibleCitation, citation))) {
      visible.push(citation);
    }
  });

  return visible;
}

function renderCitation(text: string, citation: ReviewCitation): RenderCitation | null {
  if (!isValidSpan(text, citation)) {
    return null;
  }

  const { start: highlightStart, end: highlightEnd } = trimPaintedSpan(
    text,
    citation.start,
    citation.end
  );

  if (highlightStart >= highlightEnd) {
    return null;
  }

  return {
    ...citation,
    highlightStart,
    highlightEnd
  };
}

function trimPaintedSpan(text: string, start: number, end: number) {
  let highlightStart = start;
  let highlightEnd = end;

  while (highlightStart < highlightEnd && /\s/.test(text[highlightStart] ?? "")) {
    highlightStart += 1;
  }
  while (highlightEnd > highlightStart && /\s/.test(text[highlightEnd - 1] ?? "")) {
    highlightEnd -= 1;
  }

  return { start: highlightStart, end: highlightEnd };
}

function isRenderCitation(citation: RenderCitation | null): citation is RenderCitation {
  return citation !== null;
}

function isValidSpan(text: string, citation: Pick<ReviewCitation, "start" | "end">) {
  return citation.start >= 0 && citation.end <= text.length && citation.start < citation.end;
}

function renderSpansOverlap(left: RenderCitation, right: RenderCitation) {
  return left.highlightStart < right.highlightEnd && right.highlightStart < left.highlightEnd;
}

function comparableCourtListenerClusters(validation: ValidationPayload | null) {
  if (!validation || !hasCourtListenerCandidate(validation)) {
    return [];
  }
  return validation.clusters;
}

function hasCourtListenerCandidate(validation: ValidationPayload) {
  return (validation.status === "found" || citationStatusFromValidation(validation) === "ambiguous") &&
    validation.clusters.length > 0;
}

function citationStatus(citation: ReviewCitation): CitationStatus {
  return citation.validation ? citationStatusFromValidation(citation.validation) : "not_checked";
}

function isFullCaseCitation(citation: ReviewCitation) {
  return citation.kind.toLowerCase() === "fullcasecitation";
}

function citationStatusFromValidation(validation: ValidationPayload): CitationStatus {
  const normalized = validation.status.toLowerCase().replaceAll("-", "_");
  if (normalized === "found") {
    return validation.clusters.length > 1 ? "ambiguous" : "found";
  }
  if (normalized.includes("ambiguous")) {
    return "ambiguous";
  }
  if (normalized.includes("not_found") || normalized.includes("not found")) {
    return "not_found";
  }
  return "other";
}

function citationStatusCounts(citations: ReviewCitation[]) {
  return citations.reduce(
    (counts, citation) => {
      const status = citationStatus(citation);
      counts[status] += 1;
      return counts;
    },
    { found: 0, ambiguous: 0, not_found: 0, not_checked: 0, other: 0 } satisfies Record<
      CitationStatus,
      number
    >
  );
}

function filterLabel(
  filter: { label: string; value: CitationFilter },
  counts: Record<CitationStatus, number>,
  fullCaseTotal: number,
  allCitationTotal: number
) {
  return `${filter.label} (${filterCount(filter.value, counts, fullCaseTotal, allCitationTotal)})`;
}

function filterCount(
  filter: CitationFilter,
  counts: Record<CitationStatus, number>,
  fullCaseTotal: number,
  allCitationTotal: number
) {
  if (filter === "all") {
    return fullCaseTotal;
  }
  if (filter === "all_citations") {
    return allCitationTotal;
  }
  return counts[filter];
}

async function reviewText(text: string, validate: boolean): Promise<ReviewResult> {
  const response = await fetch("/api/e2e/review-text", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text, validate })
  });
  return parseReviewResponse(response);
}

async function reviewDocument(file: File, validate: boolean): Promise<ReviewResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("validate", String(validate));

  const response = await fetch("/api/e2e/review-document", {
    method: "POST",
    body: form
  });
  return parseReviewResponse(response);
}

async function parseReviewResponse(response: Response): Promise<ReviewResult> {
  if (response.ok) {
    return (await response.json()) as ReviewResult;
  }

  let message = `Request failed with ${response.status}`;
  try {
    const payload = (await response.json()) as { detail?: string };
    if (payload.detail) {
      message = payload.detail;
    }
  } catch {
    // The status code above is enough when the backend returns non-JSON errors.
  }
  throw new Error(message);
}
