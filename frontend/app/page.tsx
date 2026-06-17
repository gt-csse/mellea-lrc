"use client";

import {
  AlertCircle,
  CheckCircle2,
  FileText,
  Loader2,
  Search,
  Upload
} from "lucide-react";
import { ChangeEvent, DragEvent, ReactNode, useMemo, useRef, useState } from "react";

type ValidationPayload = {
  citation_id: string;
  locator: string | null;
  status: string;
  source: string;
  message: string;
  case_names: string[];
};

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

const exampleText = "Brown v. Board, 347 U.S. 483 (1954). See also Roe v. Wade, 410 U.S. 113.";

export default function Home() {
  const [text, setText] = useState(exampleText);
  const [file, setFile] = useState<File | null>(null);
  const [validate, setValidate] = useState(true);
  const [result, setResult] = useState<ReviewResult | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const selectedCitation = useMemo(
    () => result?.citations.find((citation) => citation.id === selectedId) ?? null,
    [result, selectedId]
  );

  async function runReview() {
    setIsLoading(true);
    setError(null);

    try {
      const response = file ? await reviewDocument(file, validate) : await reviewText(text, validate);
      setResult(response);
      setSelectedId(response.citations[0]?.id ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Review failed");
    } finally {
      setIsLoading(false);
    }
  }

  function selectCitation(citationId: string) {
    setSelectedId(citationId);
    window.requestAnimationFrame(() => {
      document.getElementById(`citation-${citationId}`)?.scrollIntoView({
        behavior: "smooth",
        block: "center"
      });
    });
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    setFile(event.target.files?.[0] ?? null);
  }

  function onDrop(event: DragEvent<HTMLButtonElement>) {
    event.preventDefault();
    setFile(event.dataTransfer.files?.[0] ?? null);
  }

  const citationCount = result?.stats.citation_spans ?? 0;
  const foundCount = result?.stats.found;

  return (
    <main className="app-shell">
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

      <section className="ingest-panel" aria-label="Document input">
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
        <input ref={fileInputRef} className="hidden-input" type="file" onChange={onFileChange} />

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
      </section>

      {error ? (
        <div className="error-banner" role="alert">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      ) : null}

      <section className="workspace" aria-label="Citation review workspace">
        <article className="document-pane" aria-label="Document viewer">
          {result ? (
            <div className="document-text">
              {renderHighlightedDocument(result.document.text, result.citations, selectedId, selectCitation)}
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
            <h2>Extracted citations</h2>
            <span>{citationCount}</span>
          </div>
          <div className="citation-list">
            {result?.citations.length ? (
              result.citations.map((citation) => (
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
              <p className="rail-empty">No citations yet.</p>
            )}
          </div>
        </aside>
      </section>

      <section className="details-panel" aria-label="Citation details">
        {selectedCitation ? (
          <>
            <div className="details-heading">
              <div>
                <p className="eyebrow">Selected citation</p>
                <h2>{selectedCitation.matched_text}</h2>
              </div>
              <StatusBadge validation={selectedCitation.validation} />
            </div>
            <div className="details-grid">
              <DetailGroup title="Bibliographic fields" data={selectedCitation.fields} />
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

function DetailGroup({ title, data }: { title: string; data: Record<string, unknown> }) {
  const entries = Object.entries(data);

  return (
    <div className="detail-group">
      <h3>{title}</h3>
      {entries.length ? (
        <dl>
          {entries.map(([key, value]) => (
            <div key={key}>
              <dt>{key.replaceAll("_", " ")}</dt>
              <dd>{String(value)}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="muted">No fields extracted.</p>
      )}
    </div>
  );
}

function ValidationDetails({ validation }: { validation: ValidationPayload | null }) {
  if (!validation) {
    return (
      <div className="detail-group">
        <h3>Validation</h3>
        <p className="muted">Validation was not requested.</p>
      </div>
    );
  }

  return (
    <div className="detail-group">
      <h3>Validation</h3>
      <dl>
        <div>
          <dt>Status</dt>
          <dd>{validation.status.replaceAll("_", " ")}</dd>
        </div>
        <div>
          <dt>Message</dt>
          <dd>{validation.message}</dd>
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
      </dl>
    </div>
  );
}

function renderHighlightedDocument(
  text: string,
  citations: ReviewCitation[],
  selectedId: string | null,
  onSelect: (citationId: string) => void
) {
  const nodes: ReactNode[] = [];
  const sorted = [...citations].sort((left, right) => left.start - right.start);
  let cursor = 0;

  sorted.forEach((citation) => {
    if (citation.start < cursor || citation.end > text.length || citation.start >= citation.end) {
      return;
    }

    if (citation.start > cursor) {
      nodes.push(<span key={`text-${cursor}`}>{text.slice(cursor, citation.start)}</span>);
    }

    nodes.push(
      <button
        className={`citation-mark${citation.id === selectedId ? " active" : ""}`}
        id={`citation-${citation.id}`}
        key={citation.id}
        onClick={() => onSelect(citation.id)}
        type="button"
      >
        {text.slice(citation.start, citation.end)}
      </button>
    );
    cursor = citation.end;
  });

  if (cursor < text.length) {
    nodes.push(<span key={`text-${cursor}`}>{text.slice(cursor)}</span>);
  }

  return nodes;
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
