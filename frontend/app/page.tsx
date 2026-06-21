"use client";

import {
  AlertCircle,
  Brain,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  FileText,
  Loader2,
  RotateCcw,
  Search,
  ShieldCheck,
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

type AssessmentPayload = {
  citation_id: string;
  status: string;
  message: string;
  case_assess: CaseNameAssessmentPayload | null;
  year_assess: YearAssessmentPayload | null;
};

type CaseNameAssessmentPayload = {
  citation_id: string;
  status: string;
  extracted_case_name: string | null;
  courtlistener_case_name: string | null;
  message: string;
};

type TextSpan = {
  start: number;
  end: number;
};

type YearAssessmentPayload = {
  citation_id: string;
  status: string;
  extracted_year: string | null;
  courtlistener_year: string | null;
  message: string;
};

type CourtListenerCluster = Record<string, unknown>;

type ReviewValidation = {
  validations: ValidationPayload[];
  counts: {
    total: number;
    found: number;
  };
};

type ReviewAssessment = {
  assessments: AssessmentPayload[];
  modified_citations: ModifiedExtractedCitationPayload[];
  reassessments: AssessmentPayload[];
  counts: Record<string, number>;
};

type ModifiedExtractedCitationPayload = {
  citation_id: string;
  span: TextSpan | null;
  matched_text: string | null;
  plaintiff: string | null;
  defendant: string | null;
  case_name: string | null;
  extracted_case_name: string | null;
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
  assessment: AssessmentPayload | null;
};

type ReviewResult = {
  document: {
    text: string;
    source_path: string | null;
    source_format: string;
    backend: string;
  };
  citations: ReviewCitation[];
  validation: ReviewValidation | null;
  assessment: ReviewAssessment | null;
  stats: Record<string, number>;
};

type WorkflowStage = "input" | "preprocessed" | "extracted" | "validated" | "assessed";
type LoadingStage = WorkflowStage | "snapshot";
type SnapshotReviewResult = {
  stage: Exclude<WorkflowStage, "input">;
  result: ReviewResult;
};
type CitationStatus = "found" | "ambiguous" | "not_found" | "not_checked" | "other";
type AssessmentStatus =
  | "exact_match"
  | "match"
  | "irregular_form"
  | "different_case"
  | "not_assessed";
type AssessmentFilter = Exclude<AssessmentStatus, "not_assessed">;
type CitationFilter =
  | "all"
  | "found"
  | "ambiguous"
  | "not_found"
  | AssessmentFilter
  | "with_history"
  | "all_citations";
type ComparisonMatchType = "perfect" | "exact" | "semantic" | "warning" | "error" | "unchecked";
type BibliographicRow = {
  label: string;
  extracted: unknown;
  courtListener: unknown;
  matchType: ComparisonMatchType;
};
type ExtractedCitationAttempt = {
  label: string;
  fields: Record<string, unknown>;
  locator: string | null;
  citation: ReviewCitation;
  reassessment: AssessmentPayload | null;
  span: TextSpan | null;
};
type ReextractOverlay = {
  citationId: string;
  span: TextSpan;
};
type RenderCitationOverlay = {
  originalStart: number;
  originalEnd: number;
  reextractStart: number;
  reextractEnd: number;
};
type RenderCitation = ReviewCitation & {
  highlightStart: number;
  highlightEnd: number;
  overlay?: RenderCitationOverlay;
};
type WorkflowStageControl = {
  label: string;
  canEditInput: boolean;
  canLoadSnapshot: boolean;
  canExtract: boolean;
  canValidate: boolean;
  canAssess: boolean;
  canReset: boolean;
};

const VALIDATION_REQUEST_INTERVAL_MS = 1500;
const exampleText = "Brown v. Board, 347 U.S. 483 (1954). See also Roe v. Wade, 410 U.S. 113.";
const citationFilters: Array<{ label: string; value: CitationFilter }> = [
  { label: "All", value: "all" },
  { label: "Found", value: "found" },
  { label: "Ambiguous", value: "ambiguous" },
  { label: "Not found", value: "not_found" },
  { label: "All citations", value: "all_citations" }
];
const assessmentFilters: Array<{ label: string; value: CitationFilter }> = [
  { label: "Exact match", value: "exact_match" },
  { label: "Match", value: "match" },
  { label: "Irregular form", value: "irregular_form" },
  { label: "Different case", value: "different_case" },
  { label: "With history", value: "with_history" }
];
const workflowStageControls: Record<WorkflowStage, WorkflowStageControl> = {
  input: {
    label: "Input",
    canEditInput: true,
    canLoadSnapshot: true,
    canExtract: true,
    canValidate: false,
    canAssess: false,
    canReset: false
  },
  preprocessed: {
    label: "Preprocessed",
    canEditInput: false,
    canLoadSnapshot: false,
    canExtract: true,
    canValidate: false,
    canAssess: false,
    canReset: true
  },
  extracted: {
    label: "Extracted",
    canEditInput: false,
    canLoadSnapshot: false,
    canExtract: false,
    canValidate: true,
    canAssess: false,
    canReset: true
  },
  validated: {
    label: "Validated",
    canEditInput: false,
    canLoadSnapshot: false,
    canExtract: false,
    canValidate: false,
    canAssess: true,
    canReset: true
  },
  assessed: {
    label: "Assessed",
    canEditInput: false,
    canLoadSnapshot: false,
    canExtract: false,
    canValidate: false,
    canAssess: false,
    canReset: true
  }
};

export default function Home() {
  const [text, setText] = useState(exampleText);
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<ReviewResult | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState<LoadingStage | null>(null);
  const [validationProgress, setValidationProgress] = useState<{ completed: number; total: number } | null>(
    null
  );
  const [workflowStage, setWorkflowStage] = useState<WorkflowStage>("input");
  const [isInputCollapsed, setIsInputCollapsed] = useState(false);
  const [citationFilter, setCitationFilter] = useState<CitationFilter>("all");
  const [clusterIndexes, setClusterIndexes] = useState<Record<string, number>>({});
  const [extractedAttemptIndexes, setExtractedAttemptIndexes] = useState<Record<string, number>>({});
  const [isDetailsExpanded, setIsDetailsExpanded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const documentPaneRef = useRef<HTMLElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const snapshotInputRef = useRef<HTMLInputElement | null>(null);
  const validationRunIdRef = useRef(0);
  const stageControl = workflowStageControls[workflowStage];
  const isTaskLocked = !stageControl.canEditInput;
  const hasExtractionInput =
    workflowStage === "preprocessed"
      ? Boolean(result?.document.text.trim())
      : Boolean(file || text.trim());
  const canRunExtraction = stageControl.canExtract && hasExtractionInput;

  const selectedCitation = useMemo(
    () => result?.citations.find((citation) => citation.id === selectedId) ?? null,
    [result, selectedId]
  );

  const selectedExtractedAttempts = useMemo(
    () => (selectedCitation ? extractedCitationAttempts(selectedCitation, result?.assessment ?? null) : []),
    [selectedCitation, result]
  );
  const defaultExtractedAttemptIndex =
    selectedExtractedAttempts.length > 1 ? selectedExtractedAttempts.length - 1 : 0;
  const selectedExtractedAttemptIndex = selectedCitation
    ? extractedAttemptIndexes[selectedCitation.id] ?? defaultExtractedAttemptIndex
    : 0;
  const safeSelectedExtractedAttemptIndex = selectedExtractedAttempts.length
    ? Math.min(Math.max(selectedExtractedAttemptIndex, 0), selectedExtractedAttempts.length - 1)
    : 0;
  const selectedExtractedAttempt = selectedExtractedAttempts[safeSelectedExtractedAttemptIndex] ?? null;

  // A re-extracted attempt carries its own span; overlay it on top of the
  // original full-span highlight (rather than replacing it) so both are visible.
  const reextractOverlay = useMemo<ReextractOverlay | null>(() => {
    if (!selectedCitation || !selectedExtractedAttempt?.span) {
      return null;
    }
    return { citationId: selectedCitation.id, span: selectedExtractedAttempt.span };
  }, [selectedCitation, selectedExtractedAttempt]);

  const primaryCitations = useMemo(
    () =>
      result
        ? visibleCitationSpans(
            result.document.text,
            result.citations,
            selectedId,
            false,
            reextractOverlay
          )
        : [],
    [result, selectedId, reextractOverlay]
  );

  const allCitations = useMemo(
    () =>
      result
        ? visibleCitationSpans(
            result.document.text,
            result.citations,
            selectedId,
            true,
            reextractOverlay
          )
        : [],
    [result, selectedId, reextractOverlay]
  );

  const renderCitations = useMemo(
    () => (citationFilter === "all_citations" ? allCitations : primaryCitations),
    [allCitations, citationFilter, primaryCitations]
  );

  const availableFilters = useMemo(
    () =>
      workflowStage === "assessed"
        ? [...citationFilters.slice(0, -1), ...assessmentFilters, citationFilters.at(-1)!]
        : citationFilters,
    [workflowStage]
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
        if (citationFilter === "with_history") {
          return citationHasHistory(citation, result?.assessment ?? null);
        }
        if (isAssessmentFilter(citationFilter)) {
          return (
            assessmentStatusFromPayload(effectiveAssessment(citation, result?.assessment ?? null)) ===
            citationFilter
          );
        }
        return citationFilter === "all" || citationStatus(citation) === citationFilter;
      }),
    [citationFilter, renderCitations, result]
  );

  const fullCaseCitations = useMemo(
    () => primaryCitations.filter(isFullCaseCitation),
    [primaryCitations]
  );
  const statusCounts = useMemo(() => citationStatusCounts(fullCaseCitations), [fullCaseCitations]);
  const assessmentCounts = useMemo(
    () => assessmentStatusCounts(fullCaseCitations, result?.assessment ?? null),
    [fullCaseCitations, result]
  );
  const withHistoryCount = useMemo(
    () =>
      fullCaseCitations.filter((citation) =>
        citationHasHistory(citation, result?.assessment ?? null)
      ).length,
    [fullCaseCitations, result]
  );

  const selectedClusterIndex = selectedCitation ? clusterIndexes[selectedCitation.id] ?? 0 : 0;

  useEffect(() => {
    if (!result) {
      return;
    }
    if (!availableFilters.some((filter) => filter.value === citationFilter)) {
      setCitationFilter("all");
      return;
    }
    if (selectedId && filteredCitations.some((citation) => citation.id === selectedId)) {
      return;
    }
    setSelectedId(filteredCitations[0]?.id ?? null);
  }, [availableFilters, citationFilter, filteredCitations, result, selectedId]);

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

  async function runExtraction() {
    validationRunIdRef.current += 1;
    setIsLoading(true);
    setLoadingStage("extracted");
    setValidationProgress(null);
    setError(null);

    try {
      const sourceText = workflowStage === "preprocessed" && result ? result.document.text : text;
      const response = file ? await extractDocument(file) : await extractText(sourceText);
      setResult(response);
      setSelectedId(response.citations[0]?.id ?? null);
      setCitationFilter("all");
      setClusterIndexes({});
      setExtractedAttemptIndexes({});
      setWorkflowStage("extracted");
      setIsInputCollapsed(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Extraction failed");
    } finally {
      setIsLoading(false);
      setLoadingStage(null);
    }
  }

  async function runValidation() {
    if (!result) {
      return;
    }
    const citationsToValidate = result.citations.filter(isFullCaseCitation);
    if (!citationsToValidate.length) {
      setResult(withValidationStats(result));
      setWorkflowStage("validated");
      setCitationFilter("all");
      return;
    }

    const runId = validationRunIdRef.current + 1;
    validationRunIdRef.current = runId;
    setIsLoading(true);
    setLoadingStage("validated");
    setValidationProgress({ completed: 0, total: citationsToValidate.length });
    setError(null);

    try {
      const failures: string[] = [];
      await Promise.all(
        citationsToValidate.map(async (citation, index) => {
          if (index > 0) {
            await wait(VALIDATION_REQUEST_INTERVAL_MS * index);
          }
          if (validationRunIdRef.current !== runId) {
            return;
          }

          try {
            const validation = await validateReviewCitation(citation);
            if (validationRunIdRef.current !== runId) {
              return;
            }
            setResult((current) => (current ? mergeCitationValidation(current, validation) : current));
          } catch (err) {
            failures.push(err instanceof Error ? err.message : "Validation failed");
          } finally {
            if (validationRunIdRef.current === runId) {
              setValidationProgress((current) =>
                current ? { ...current, completed: Math.min(current.completed + 1, current.total) } : current
              );
            }
          }
        })
      );

      if (validationRunIdRef.current !== runId) {
        return;
      }
      setResult((current) => (current ? withValidationStats(current) : current));
      setCitationFilter("all");
      if (failures.length) {
        setError(`Validation finished with ${failures.length} failed request${failures.length === 1 ? "" : "s"}.`);
        setWorkflowStage("extracted");
      } else {
        setWorkflowStage("validated");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Validation failed");
    } finally {
      if (validationRunIdRef.current === runId) {
        setIsLoading(false);
        setLoadingStage(null);
        setValidationProgress(null);
      }
    }
  }

  async function runAssessment() {
    if (!result) {
      return;
    }
    setIsLoading(true);
    setLoadingStage("assessed");
    setError(null);

    try {
      const response = await assessReview(result);
      setResult(response);
      setWorkflowStage("assessed");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Assessment failed");
    } finally {
      setIsLoading(false);
      setLoadingStage(null);
    }
  }

  async function loadSnapshot(file: File) {
    validationRunIdRef.current += 1;
    setIsLoading(true);
    setLoadingStage("snapshot");
    setValidationProgress(null);
    setError(null);

    try {
      const response = await loadSnapshotReview(file);
      setResult(response.result);
      setText(response.result.document.text);
      setFile(null);
      setSelectedId(response.result.citations[0]?.id ?? null);
      setCitationFilter("all");
      setClusterIndexes({});
      setExtractedAttemptIndexes({});
      setWorkflowStage(response.stage);
      setIsInputCollapsed(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Snapshot load failed");
    } finally {
      setIsLoading(false);
      setLoadingStage(null);
    }
  }

  function resetTask() {
    validationRunIdRef.current += 1;
    setResult(null);
    setSelectedId(null);
    setCitationFilter("all");
    setClusterIndexes({});
    setExtractedAttemptIndexes({});
    setWorkflowStage("input");
    setIsInputCollapsed(false);
    setIsDetailsExpanded(false);
    setValidationProgress(null);
    setError(null);
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

  function selectExtractedAttempt(citationId: string, attemptIndex: number) {
    setExtractedAttemptIndexes((current) => ({
      ...current,
      [citationId]: attemptIndex
    }));
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    if (isTaskLocked) {
      return;
    }
    setFile(event.target.files?.[0] ?? null);
    setIsInputCollapsed(false);
  }

  function onSnapshotChange(event: ChangeEvent<HTMLInputElement>) {
    const snapshot = event.target.files?.[0];
    event.target.value = "";
    if (!snapshot || isTaskLocked) {
      return;
    }
    void loadSnapshot(snapshot);
  }

  function onDrop(event: DragEvent<HTMLButtonElement>) {
    event.preventDefault();
    if (isTaskLocked) {
      return;
    }
    setFile(event.dataTransfer.files?.[0] ?? null);
    setIsInputCollapsed(false);
  }

  const citationCount = result?.stats.citation_spans ?? 0;
  const foundCount = result?.stats.found;
  const sourceLabel = file?.name ?? result?.document.source_path ?? `${text.trim().length || 0} chars`;
  const validationLabel =
    loadingStage === "validated" && validationProgress
      ? `Validating ${validationProgress.completed}/${validationProgress.total}`
      : "Validate";

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
          <Metric label="Boundary" value={stageControl.label} />
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
            {isTaskLocked ? <span className="lock-note">Locked task</span> : null}
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
                disabled={isTaskLocked}
                onChange={(event) => {
                  if (isTaskLocked) {
                    return;
                  }
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
              disabled={isTaskLocked}
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
        <input
          ref={snapshotInputRef}
          className="hidden-input"
          type="file"
          accept="application/json,.json"
          onChange={onSnapshotChange}
        />

        <div className="ingest-actions">
          <button
            className="primary-action"
            disabled={isLoading || !canRunExtraction}
            onClick={runExtraction}
            type="button"
          >
            {loadingStage === "extracted" ? <Loader2 className="spin" size={18} /> : <Search size={18} />}
            <span>{loadingStage === "extracted" ? "Extracting" : "Extract"}</span>
          </button>
          <button
            className="secondary-action"
            disabled={isLoading || !result || !stageControl.canValidate}
            onClick={runValidation}
            type="button"
          >
            {loadingStage === "validated" ? (
              <Loader2 className="spin" size={18} />
            ) : (
              <ShieldCheck size={18} />
            )}
            <span>{validationLabel}</span>
          </button>
          <button
            className="secondary-action"
            disabled={isLoading || !result || !stageControl.canAssess}
            onClick={runAssessment}
            type="button"
          >
            {loadingStage === "assessed" ? <Loader2 className="spin" size={18} /> : <Brain size={18} />}
            <span>{loadingStage === "assessed" ? "Assessing" : "Assess"}</span>
          </button>
          <button
            className="secondary-action"
            disabled={isLoading || !stageControl.canLoadSnapshot}
            onClick={() => snapshotInputRef.current?.click()}
            type="button"
          >
            {loadingStage === "snapshot" ? <Loader2 className="spin" size={18} /> : <FileText size={18} />}
            <span>{loadingStage === "snapshot" ? "Loading" : "Load snapshot"}</span>
          </button>
          <button
            className="icon-action"
            disabled={isLoading || !stageControl.canReset}
            onClick={resetTask}
            type="button"
            aria-label="Reset current task"
          >
            <RotateCcw size={18} />
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
              <span>Run extraction to inspect citation spans.</span>
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
              {availableFilters.map((filter) => (
                <option key={filter.value} value={filter.value}>
                  {filterLabel(
                    filter,
                    statusCounts,
                    assessmentCounts,
                    fullCaseCitations.length,
                    allCitations.length,
                    withHistoryCount
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
                  <CitationTags citation={citation} assessment={result?.assessment ?? null} />
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
                <CitationTags
                  citation={selectedCitation}
                  assessment={result?.assessment ?? null}
                  variant="details"
                />
              </div>
              <div className="details-actions">
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
                extractedAttempts={selectedExtractedAttempts}
                extractedAttemptIndex={selectedExtractedAttemptIndex}
                clusterIndex={selectedClusterIndex}
                onClusterChange={selectCourtListenerCandidate}
                onExtractedAttemptChange={selectExtractedAttempt}
              />
              <ValidationDetails validation={selectedCitation.validation} />
              <AssessmentDetails assessment={selectedCitation.assessment} />
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

function CitationTags({
  citation,
  assessment = null,
  variant = "rail"
}: {
  citation: ReviewCitation;
  assessment?: ReviewAssessment | null;
  variant?: "rail" | "details";
}) {
  // The event-level status rolls up to the re-extraction result when one exists,
  // so a recovered citation reads as its final (re-extracted) verdict.
  const effective = effectiveAssessment(citation, assessment);
  return (
    <span className={`citation-tags ${variant}`} aria-label="Citation labels">
      <span className="citation-tag kind">{citation.kind}</span>
      {citation.validation ? (
        <span className={`citation-tag validation ${citationStatus(citation)}`}>
          {formatStatusLabel(citation.validation.status)}
        </span>
      ) : null}
      {effective ? (
        <span className={`citation-tag assessment ${assessmentStatusFromPayload(effective)}`}>
          {formatAssessmentLabel(effective.status)}
        </span>
      ) : null}
    </span>
  );
}

function BibliographicComparison({
  citation,
  extractedAttempts,
  clusterIndex,
  extractedAttemptIndex,
  onClusterChange,
  onExtractedAttemptChange
}: {
  citation: ReviewCitation;
  extractedAttempts: ExtractedCitationAttempt[];
  clusterIndex: number;
  extractedAttemptIndex: number;
  onClusterChange: (citationId: string, clusterIndex: number) => void;
  onExtractedAttemptChange: (citationId: string, attemptIndex: number) => void;
}) {
  const clusters = comparableCourtListenerClusters(citation.validation);
  const safeClusterIndex = clusters.length
    ? Math.min(Math.max(clusterIndex, 0), clusters.length - 1)
    : 0;
  const cluster = clusters[safeClusterIndex] ?? null;
  const safeExtractedAttemptIndex = extractedAttempts.length
    ? Math.min(Math.max(extractedAttemptIndex, 0), extractedAttempts.length - 1)
    : 0;
  const extractedAttempt = extractedAttempts[safeExtractedAttemptIndex] ?? extractedAttempts[0];
  const rows = bibliographicRows(extractedAttempt, cluster);
  const hasMultipleClusters = clusters.length > 1;
  const hasMultipleExtractedAttempts = extractedAttempts.length > 1;

  function changeCluster(direction: -1 | 1) {
    if (!hasMultipleClusters) {
      return;
    }
    const nextIndex = (safeClusterIndex + direction + clusters.length) % clusters.length;
    onClusterChange(citation.id, nextIndex);
  }

  function changeExtractedAttempt(direction: -1 | 1) {
    if (!hasMultipleExtractedAttempts) {
      return;
    }
    const nextIndex =
      (safeExtractedAttemptIndex + direction + extractedAttempts.length) % extractedAttempts.length;
    onExtractedAttemptChange(citation.id, nextIndex);
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
          <span className="courtlistener-column-header" role="columnheader">
            <span>{extractedAttempt?.label ?? "Extracted"}</span>
            {hasMultipleExtractedAttempts ? (
              <span className="case-switcher" aria-label="Extracted citation history selector">
                <button
                  aria-label="Show previous extracted citation"
                  type="button"
                  onClick={() => changeExtractedAttempt(-1)}
                >
                  <ChevronLeft size={14} aria-hidden="true" />
                </button>
                <strong>
                  {safeExtractedAttemptIndex + 1}/{extractedAttempts.length}
                </strong>
                <button
                  aria-label="Show next extracted citation"
                  type="button"
                  onClick={() => changeExtractedAttempt(1)}
                >
                  <ChevronRight size={14} aria-hidden="true" />
                </button>
              </span>
            ) : null}
          </span>
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
          <div className={`comparison-row ${row.matchType}`} key={row.label} role="row">
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

function AssessmentDetails({ assessment }: { assessment: AssessmentPayload | null }) {
  if (!assessment) {
    return (
      <div className="detail-group">
        <h3>Citation assessment</h3>
        <dl>
          <div>
            <dt>Status</dt>
            <dd>Not requested</dd>
          </div>
          <div>
            <dt>Message</dt>
            <dd>Run assessment after validation to check case-name extraction.</dd>
          </div>
        </dl>
      </div>
    );
  }

  return (
    <div className="detail-group">
      <h3>Citation assessment</h3>
      <dl>
        <div>
          <dt>Status</dt>
          <dd>{assessment.status.replaceAll("_", " ")}</dd>
        </div>
        <div>
          <dt>Message</dt>
          <dd>{assessment.message}</dd>
        </div>
        {assessment.case_assess ? <CaseNameAssessmentDetails assessment={assessment.case_assess} /> : null}
        {assessment.year_assess ? <YearAssessmentDetails assessment={assessment.year_assess} /> : null}
      </dl>
    </div>
  );
}

function CaseNameAssessmentDetails({ assessment }: { assessment: CaseNameAssessmentPayload }) {
  return (
    <>
      <div>
        <dt>Case name</dt>
        <dd>{assessment.status.replaceAll("_", " ")}</dd>
      </div>
      <div>
        <dt>Case message</dt>
        <dd>{assessment.message}</dd>
      </div>
      <div>
        <dt>Extracted case</dt>
        <dd>{formatValue(assessment.extracted_case_name)}</dd>
      </div>
      <div>
        <dt>CourtListener case</dt>
        <dd>{formatValue(assessment.courtlistener_case_name)}</dd>
      </div>
    </>
  );
}

function YearAssessmentDetails({ assessment }: { assessment: YearAssessmentPayload }) {
  return (
    <>
      <div>
        <dt>Year</dt>
        <dd>{assessment.status.replaceAll("_", " ")}</dd>
      </div>
      <div>
        <dt>Year message</dt>
        <dd>{assessment.message}</dd>
      </div>
      <div>
        <dt>Extracted year</dt>
        <dd>{formatValue(assessment.extracted_year)}</dd>
      </div>
      <div>
        <dt>CourtListener year</dt>
        <dd>{formatValue(assessment.courtlistener_year)}</dd>
      </div>
    </>
  );
}

function extractedCitationAttempts(
  citation: ReviewCitation,
  assessment: ReviewAssessment | null
): ExtractedCitationAttempt[] {
  const original: ExtractedCitationAttempt = {
    label: "Extracted",
    fields: citation.fields,
    locator: citation.validation?.locator ?? citation.matched_text,
    citation,
    reassessment: null,
    span: null
  };
  const modified = assessment?.modified_citations
    .filter((item) => item.citation_id === citation.id)
    .map((item, index, items) => ({
      label: items.length > 1 ? `Re-extracted ${index + 1}` : "Re-extracted",
      fields: {
        ...citation.fields,
        plaintiff: item.plaintiff ?? undefined,
        defendant: item.defendant ?? undefined,
        case_name: item.case_name ?? item.extracted_case_name ?? undefined
      },
      locator: citation.validation?.locator ?? citation.matched_text,
      citation,
      reassessment:
        assessment.reassessments.find((candidate) => candidate.citation_id === citation.id) ?? null,
      span: item.span ?? null
    }));

  return [original, ...(modified ?? [])];
}

function bibliographicRows(
  extractedAttempt: ExtractedCitationAttempt,
  cluster: CourtListenerCluster | null
): BibliographicRow[] {
  const { citation, fields } = extractedAttempt;
  const canColorRows = Boolean(citation.assessment);
  const citationLocator = extractedAttempt.locator ?? citation.validation?.locator ?? citation.matched_text;
  const extractedLocatorParts = splitLocator(citationLocator);
  const courtListenerLocator = cluster ? citation.validation?.locator : null;
  const courtListenerLocatorParts = splitLocator(courtListenerLocator);
  const extractedCaseName = caseNameFromFields(fields) || extractedAttempt.locator || citation.matched_text;
  const courtListenerCaseName = readString(cluster, ["case_name", "caseName"]);
  const courtListenerCourt = readString(cluster, ["court", "court_id", "courtId"]);
  const courtListenerDate = readString(cluster, ["date_filed", "dateFiled"]);
  const courtListenerUrl = readString(cluster, ["absolute_url", "absoluteUrl", "resource_uri"]);

  return [
    {
      label: "Plaintiff",
      extracted: fields.plaintiff ?? null,
      courtListener: null,
      matchType: "unchecked"
    },
    {
      label: "Defendant",
      extracted: fields.defendant ?? null,
      courtListener: null,
      matchType: "unchecked"
    },
    {
      label: "Case name",
      extracted: extractedCaseName,
      courtListener: courtListenerCaseName,
      matchType: canColorRows
        ? caseNameRowMatchType(
            extractedAttempt.reassessment?.case_assess ?? citation.assessment?.case_assess,
            extractedCaseName,
            courtListenerCaseName
          )
        : "unchecked"
    },
    {
      label: "Locator",
      extracted: citationLocator,
      courtListener: courtListenerLocator,
      matchType: canColorRows ? "perfect" : "unchecked"
    },
    {
      label: "Volume",
      extracted: fields.volume ?? extractedLocatorParts.volume,
      courtListener: courtListenerLocatorParts.volume,
      matchType: canColorRows ? "perfect" : "unchecked"
    },
    {
      label: "Reporter",
      extracted: fields.reporter ?? extractedLocatorParts.reporter,
      courtListener: courtListenerLocatorParts.reporter,
      matchType: canColorRows ? "perfect" : "unchecked"
    },
    {
      label: "Page",
      extracted: fields.page ?? extractedLocatorParts.page,
      courtListener: courtListenerLocatorParts.page,
      matchType: canColorRows ? "perfect" : "unchecked"
    },
    {
      label: "Year",
      extracted: fields.year,
      courtListener: courtListenerDate?.slice(0, 4),
      matchType: canColorRows
        ? yearRowMatchType(
            extractedAttempt.reassessment?.year_assess ?? citation.assessment?.year_assess,
            fields.year,
            courtListenerDate?.slice(0, 4)
          )
        : "unchecked"
    },
    {
      label: "Court",
      extracted: fields.court,
      courtListener: courtListenerCourt,
      matchType: "unchecked"
    },
    {
      label: "URL",
      extracted: null,
      courtListener: courtListenerUrl,
      matchType: "unchecked"
    }
  ];
}

function caseNameRowMatchType(
  assessment: CaseNameAssessmentPayload | null | undefined,
  extractedCaseName: string | null,
  courtListenerCaseName: string | null
): ComparisonMatchType {
  if (assessment) {
    const status = assessment.status.toLowerCase().replaceAll("-", "_");
    if (status === "exact_match") {
      return "exact";
    }
    if (status === "match" || status === "semantic_match") {
      return "semantic";
    }
    if (status === "irregular_form") {
      return "warning";
    }
    if (status === "different_case" || status === "extraction_error") {
      return "error";
    }
  }
  return directRowMatchType(extractedCaseName, courtListenerCaseName);
}

function yearRowMatchType(
  assessment: YearAssessmentPayload | null | undefined,
  extracted: unknown,
  courtListener: unknown
): ComparisonMatchType {
  const status = assessment?.status.toLowerCase().replaceAll("-", "_");
  if (status === "exact_match") {
    return "exact";
  }
  if (status === "mismatch") {
    return "error";
  }
  if (status === "missing") {
    return "unchecked";
  }
  return directRowMatchType(extracted, courtListener);
}

function directRowMatchType(extracted: unknown, courtListener: unknown): ComparisonMatchType {
  if (!hasDisplayValue(extracted) || !hasDisplayValue(courtListener)) {
    return "unchecked";
  }
  return String(extracted) === String(courtListener) ? "exact" : "error";
}

function caseNameFromFields(fields: Record<string, unknown>) {
  const caseName = stringField(fields.case_name);
  if (caseName) {
    return caseName;
  }
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

function formatStatusLabel(value: string) {
  return value.replaceAll("_", " ");
}

function formatAssessmentLabel(value: string) {
  return value.replaceAll("_", " ");
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

type AnchorClaim = { claimed: boolean };

function renderCitationMark(
  citationText: string,
  citation: RenderCitation,
  isSelected: boolean,
  onSelect: (citationId: string) => void
) {
  const anchor: AnchorClaim = { claimed: false };
  if (citation.overlay) {
    return renderOverlayCitationMark(citationText, citation, citation.overlay, onSelect, anchor);
  }
  return renderMarkChunks(
    citationText,
    citation.id,
    `citation-mark${isSelected ? " active" : ""}`,
    onSelect,
    "mark",
    anchor
  );
}

// Paint the original full-span highlight and, layered on top of it, the
// re-extracted span in a distinct color. The two spans can overlap or extend
// past each other, so split the union into segments classed by membership.
function renderOverlayCitationMark(
  citationText: string,
  citation: RenderCitation,
  overlay: RenderCitationOverlay,
  onSelect: (citationId: string) => void,
  anchor: AnchorClaim
) {
  const base = citation.highlightStart;
  const { originalStart, originalEnd, reextractStart, reextractEnd } = overlay;
  const bounds = Array.from(
    new Set(
      [
        citation.highlightStart,
        citation.highlightEnd,
        originalStart,
        originalEnd,
        reextractStart,
        reextractEnd
      ].filter((value) => value >= citation.highlightStart && value <= citation.highlightEnd)
    )
  ).sort((left, right) => left - right);

  const nodes: ReactNode[] = [];
  for (let index = 0; index < bounds.length - 1; index += 1) {
    const segStart = bounds[index];
    const segEnd = bounds[index + 1];
    if (segStart >= segEnd) {
      continue;
    }
    const segText = citationText.slice(segStart - base, segEnd - base);
    const mid = (segStart + segEnd) / 2;
    const inOriginal = mid >= originalStart && mid < originalEnd;
    const inReextract = mid >= reextractStart && mid < reextractEnd;
    if (!inOriginal && !inReextract) {
      nodes.push(<span key={`${citation.id}-gap-${segStart}`}>{segText}</span>);
      continue;
    }
    const classes = ["citation-mark", "active"];
    if (inReextract) {
      classes.push("reextracted");
    }
    if (inReextract && !inOriginal) {
      classes.push("reextract-only");
    }
    nodes.push(
      ...renderMarkChunks(segText, citation.id, classes.join(" "), onSelect, `seg-${segStart}`, anchor)
    );
  }
  return nodes;
}

function renderMarkChunks(
  text: string,
  citationId: string,
  className: string,
  onSelect: (citationId: string) => void,
  keyPrefix: string,
  anchor: AnchorClaim
) {
  const chunks = text.split(/([^\S\r\n]*[\r\n]+[^\S\r\n]*)/);

  return chunks.map((chunk, index) => {
    if (!chunk) {
      return null;
    }
    if (/^[^\S\r\n]*[\r\n]+[^\S\r\n]*$/.test(chunk)) {
      return <span key={`${citationId}-${keyPrefix}-space-${index}`}>{chunk}</span>;
    }

    let id: string | undefined;
    if (!anchor.claimed) {
      id = `citation-${citationId}`;
      anchor.claimed = true;
    }

    return (
      <button
        className={className}
        id={id}
        key={`${citationId}-${keyPrefix}-mark-${index}`}
        onClick={() => onSelect(citationId)}
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
  includeAll: boolean,
  overlay: ReextractOverlay | null = null
) {
  const candidates = citations
    .map((citation) =>
      renderCitation(text, citation, overlay?.citationId === citation.id ? overlay : null)
    )
    .filter(isRenderCitation);
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

function renderCitation(
  text: string,
  citation: ReviewCitation,
  overlay: ReextractOverlay | null = null
): RenderCitation | null {
  const span = preferredCitationSpan(citation);

  if (!isValidSpan(text, span)) {
    return null;
  }

  const original = trimPaintedSpan(text, span.start, span.end);

  if (original.start >= original.end) {
    return null;
  }

  let highlightStart = original.start;
  let highlightEnd = original.end;
  let renderOverlay: RenderCitationOverlay | undefined;

  if (overlay && isValidSpan(text, overlay.span)) {
    const reextract = trimPaintedSpan(text, overlay.span.start, overlay.span.end);
    if (reextract.start < reextract.end) {
      highlightStart = Math.min(highlightStart, reextract.start);
      highlightEnd = Math.max(highlightEnd, reextract.end);
      renderOverlay = {
        originalStart: original.start,
        originalEnd: original.end,
        reextractStart: reextract.start,
        reextractEnd: reextract.end
      };
    }
  }

  return {
    ...citation,
    highlightStart,
    highlightEnd,
    overlay: renderOverlay
  };
}

function preferredCitationSpan(citation: ReviewCitation): TextSpan {
  return citation;
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

// A citation "has history" when a grounded re-extraction produced an additional
// extracted attempt for it (the source of the extraction-history switcher).
function citationHasHistory(
  citation: ReviewCitation,
  assessment: ReviewAssessment | null
): boolean {
  return Boolean(
    assessment?.modified_citations.some((modified) => modified.citation_id === citation.id)
  );
}

// The effective (event-level) assessment for a citation: the re-extraction
// reassessment when one exists, otherwise the original first-pass assessment.
function effectiveAssessment(
  citation: ReviewCitation,
  assessment: ReviewAssessment | null
): AssessmentPayload | null {
  const reassessment = assessment?.reassessments.find(
    (candidate) => candidate.citation_id === citation.id
  );
  return reassessment ?? citation.assessment;
}

function assessmentStatusFromPayload(assessment: AssessmentPayload | null): AssessmentStatus {
  if (!assessment) {
    return "not_assessed";
  }
  const normalized = assessment.status.toLowerCase().replaceAll("-", "_");
  if (normalized === "exact_match") {
    return "exact_match";
  }
  // Accept the legacy "semantic_match" alias for the current "match" verdict.
  if (normalized === "match" || normalized === "semantic_match") {
    return "match";
  }
  if (normalized === "irregular_form") {
    return "irregular_form";
  }
  // Map the legacy catch-all "extraction_error" onto "different_case".
  if (normalized === "different_case" || normalized === "extraction_error") {
    return "different_case";
  }
  return "not_assessed";
}

function isAssessmentFilter(filter: CitationFilter): filter is AssessmentFilter {
  return (
    filter === "exact_match" ||
    filter === "match" ||
    filter === "irregular_form" ||
    filter === "different_case"
  );
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

function mergeCitationValidation(result: ReviewResult, validation: ValidationPayload): ReviewResult {
  return withValidationStats({
    ...result,
    citations: result.citations.map((citation) =>
      citation.id === validation.citation_id
        ? { ...citation, validation, assessment: null }
        : citation
    )
  });
}

function withValidationStats(result: ReviewResult): ReviewResult {
  const validations = result.citations
    .map((citation) => citation.validation)
    .filter(isValidationPayload);
  const validated = validations.filter((validation) => validation.status !== "skipped").length;
  const found = validations.filter((validation) => validation.status === "found").length;

  return {
    ...result,
    validation: {
      validations,
      counts: {
        total: validations.length,
        found
      }
    },
    stats: {
      ...result.stats,
      validated,
      found
    }
  };
}

function isValidationPayload(value: ValidationPayload | null): value is ValidationPayload {
  return value !== null;
}

function assessmentStatusCounts(
  citations: ReviewCitation[],
  assessment: ReviewAssessment | null
) {
  return citations.reduce(
    (counts, citation) => {
      const status = assessmentStatusFromPayload(effectiveAssessment(citation, assessment));
      counts[status] += 1;
      return counts;
    },
    {
      exact_match: 0,
      match: 0,
      irregular_form: 0,
      different_case: 0,
      not_assessed: 0
    } satisfies Record<AssessmentStatus, number>
  );
}

function filterLabel(
  filter: { label: string; value: CitationFilter },
  counts: Record<CitationStatus, number>,
  assessmentCounts: Record<AssessmentStatus, number>,
  fullCaseTotal: number,
  allCitationTotal: number,
  withHistoryTotal: number
) {
  return `${filter.label} (${filterCount(
    filter.value,
    counts,
    assessmentCounts,
    fullCaseTotal,
    allCitationTotal,
    withHistoryTotal
  )})`;
}

function filterCount(
  filter: CitationFilter,
  counts: Record<CitationStatus, number>,
  assessmentCounts: Record<AssessmentStatus, number>,
  fullCaseTotal: number,
  allCitationTotal: number,
  withHistoryTotal: number
) {
  if (filter === "all") {
    return fullCaseTotal;
  }
  if (filter === "all_citations") {
    return allCitationTotal;
  }
  if (filter === "with_history") {
    return withHistoryTotal;
  }
  if (isAssessmentFilter(filter)) {
    return assessmentCounts[filter];
  }
  return counts[filter];
}

async function extractText(text: string): Promise<ReviewResult> {
  const response = await fetch("/api/e2e/extract-text", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text })
  });
  return parseReviewResponse(response);
}

async function extractDocument(file: File): Promise<ReviewResult> {
  const form = new FormData();
  form.append("file", file);

  const response = await fetch("/api/e2e/extract-document", {
    method: "POST",
    body: form
  });
  return parseReviewResponse(response);
}

async function validateReviewCitation(citation: ReviewCitation): Promise<ValidationPayload> {
  const response = await fetch("/api/e2e/validate-review-citation", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ citation })
  });
  return parseJsonResponse<ValidationPayload>(response);
}

async function assessReview(result: ReviewResult): Promise<ReviewResult> {
  const response = await fetch("/api/e2e/assess-review", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(result)
  });
  return parseReviewResponse(response);
}

async function loadSnapshotReview(file: File): Promise<SnapshotReviewResult> {
  const form = new FormData();
  form.append("file", file);

  const response = await fetch("/api/e2e/review-snapshot", {
    method: "POST",
    body: form
  });
  return parseJsonResponse<SnapshotReviewResult>(response);
}

async function parseReviewResponse(response: Response): Promise<ReviewResult> {
  return parseJsonResponse<ReviewResult>(response);
}

async function parseJsonResponse<T>(response: Response): Promise<T> {
  if (response.ok) {
    return (await response.json()) as T;
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

function wait(milliseconds: number) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}
