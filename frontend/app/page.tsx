"use client";

import {
  AlertCircle,
  Bookmark,
  BookmarkCheck,
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
import {
  ChangeEvent,
  DragEvent,
  FormEvent,
  KeyboardEvent,
  ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState
} from "react";

type CourtListenerRequestTracePayload = {
  http_status: number | null;
  cache: string | null;
  error_message: string | null;
};

type CourtResolutionTracePayload = {
  courtlistener_court_id: string | null;
  resolved_via: string;
  docket_id: string | null;
  docket_url: string | null;
  request_trace: CourtListenerRequestTracePayload | null;
};

type CaseNameSearchTracePayload = {
  status: string;
  query: string | null;
  preparation: {
    status: string;
    original_case_name: string | null;
    plaintiff: string | null;
    defendant: string | null;
    prepared_case_name: string | null;
    court: string | null;
    locator: string | null;
    source: string | null;
    llm_classification: string | null;
    llm_reason: string | null;
    error_message: string | null;
  };
  probes: Array<{
    corpus: "o" | "r";
    status: string;
    request_trace: CourtListenerRequestTracePayload;
    case_count: number | null;
    candidates: Array<{
      case_name: string | null;
      court_id: string | null;
      date_filed: string | null;
      docket_number: string | null;
      cluster_id: string | null;
      docket_id: string | null;
      absolute_url: string | null;
    }>;
  }>;
};

type RetrievedCandidatePayload = {
  candidate_id: string;
  record: CourtListenerCitationRecord;
  court_resolution: CourtResolutionTracePayload;
};

type RetrievalPayload = {
  citation_id: string;
  locator: string | null;
  status: string;
  source: string;
  case_names: string[];
  request_trace: CourtListenerRequestTracePayload;
  failure_detail: Record<string, unknown> | null;
  candidate: RetrievedCandidatePayload | null;
  candidates: RetrievedCandidatePayload[];
  candidate_search: CaseNameSearchTracePayload | null;
};

type AssessmentPayload = {
  case_name: CaseNameAssessmentRunPayload;
  court: CourtAssessmentPayload;
  year: YearAssessmentPayload;
};

type CandidateAssessmentPayload = {
  candidate_id: string;
  result: AssessmentPayload;
};

type CitationAssessmentPayload =
  | { citation_id: string; status: "waiting" }
  | { citation_id: string; status: "skipped"; reason: string; message: string }
  | { citation_id: string; status: "assessed"; candidate_id: string; result: AssessmentPayload }
  | {
      citation_id: string;
      status: "ambiguous";
      candidates: CandidateAssessmentPayload[];
      gated: boolean;
      message: string;
    }
  | { citation_id: string; status: "failed"; error: string };

type CaseNameAssessmentPayload = {
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
  status: string;
  extracted_year: string | null;
  courtlistener_year: string | null;
  message: string;
};

type CourtAssessmentPayload = {
  status: string;
  extracted_court: string | null;
  courtlistener_court_id: string | null;
  message: string;
};


type CourtListenerCitationRecord = Record<string, unknown>;

type ReviewRetrieval = {
  retrievals: RetrievalPayload[];
  counts: {
    total: number;
    found: number;
  };
};

type ReviewAssessment = {
  assessments: CitationAssessmentPayload[];
  assessment_complete: boolean;
  status_counts: Record<string, number>;
  case_name_followup_status_counts: Record<string, number>;
  case_name_counts: Record<string, number>;
  court_counts: Record<string, number>;
  year_counts: Record<string, number>;
};

type CitationNodeStepPayload = {
  step_id: string | null;
  operation: string;
  status: "succeeded" | "blocked" | "failed" | "skipped";
  depends_on: string[];
  lane: string | null;
  summary: string;
  data: Record<string, unknown>;
  error: string | null;
};

type CitationNodePayload = {
  citation_id: string;
  status: "ready" | "running" | "blocked" | "failed" | "complete";
  input: {
    citation_id: string;
    matched_text: string;
    span: TextSpan;
    citation: Record<string, unknown>;
    resolves_to: string | null;
  };
  steps: CitationNodeStepPayload[];
};

type CitationNodeGraphPayload = {
  schema_version: number;
  artifact_type: "citation_node_document";
  text: string;
  nodes: CitationNodePayload[];
};

type CitationNodeGraphItem = CitationNodeStepPayload & {
  key: string;
};

type ReextractedCaseNamePayload = {
  case_name: string;
  case_name_span: TextSpan;
};

type CaseNameFollowupPayload =
  | { status: "not_required" }
  | { status: "reextraction_failed"; error: string }
  | {
      status: "reassessed";
      reextracted_case_name: ReextractedCaseNamePayload;
      result: CaseNameAssessmentPayload;
    }
  | {
      status: "reassessment_failed";
      reextracted_case_name: ReextractedCaseNamePayload;
      error: string;
    };

type CaseNameAssessmentRunPayload = {
  initial: CaseNameAssessmentPayload;
  followup: CaseNameFollowupPayload;
};

type ReviewCitation = {
  id: string;
  start: number;
  end: number;
  matched_text: string;
  kind: string;
  fields: Record<string, string | number | boolean>;
  resolves_to: string | null;
  retrieval: RetrievalPayload | null;
  assessment: CitationAssessmentPayload | null;
};

type ReporterPayload = {
  edition_short_name: string;
  root_short_name: string;
  name: string;
  cite_type: string;
  is_scotus: boolean;
  source: string | null;
};

type ReporterInferencePayload = {
  reporter: ReporterPayload | null;
  status: "unsupported" | "missing_reporter" | "unrecognized" | "recognized";
  mlz_jurisdictions: string[];
};

type CourtsDBClassificationPayload = {
  court_id: string;
  system: string;
  jurisdiction: string | null;
  type: string | null;
};

type CourtInferencePayload = {
  extracted_court: string | null;
  status: "unsupported" | "missing_court" | "unrecognized" | "resolved";
  courts_db_classification: CourtsDBClassificationPayload | null;
};

type JurisdictionPayload = {
  reporter_inference: ReporterInferencePayload;
  court_inference: CourtInferencePayload;
};

type ReviewResult = {
  document: {
    text: string;
    source_path: string | null;
    source_format: string;
    backend: string;
  };
  citations: ReviewCitation[];
  jurisdictions: JurisdictionPayload[] | null;
  retrieval: ReviewRetrieval | null;
  assessment: ReviewAssessment | null;
  node_graph?: CitationNodeGraphPayload | null;
  stats: Record<string, number>;
};

type BookmarkStatusEntry = {
  bookmarked: boolean;
  comment: string | null;
};

type BookmarkStatusResponse = {
  statuses: Record<string, BookmarkStatusEntry>;
  paths: { json: string; text: string };
};

type BookmarkMutationResult = {
  result: "created" | "provenance_added" | "already_bookmarked";
  bookmark_id: string;
  comment: string | null;
  provenance_count: number;
  paths: { json: string; text: string };
};

type BookmarkCommentUpdateResult = {
  result: "updated" | "unchanged" | "not_found";
  bookmark_id: string;
  comment: string | null;
  paths: { json: string; text: string };
};

type WorkflowStage =
  | "input"
  | "preprocessed"
  | "extracted"
  | "node_graph"
  | "inferred"
  | "retrieved"
  | "assessed";
type LoadingStage = WorkflowStage | "snapshot";
type SnapshotReviewResult = {
  stage: Exclude<WorkflowStage, "input">;
  result: ReviewResult;
};
type CitationStatus = "found" | "ambiguous" | "not_found" | "throttled" | "not_checked";
type AssessmentStatus =
  | "exact_match"
  | "semantic_match"
  | "not_semantic_match"
  | "irregular_form"
  | "different_case"
  | "unassessable";
type ExtractionFilter = "all" | "all_citations";
type RetrievalFilter = "all" | "found" | "ambiguous" | "not_found" | "throttled";
type AssessmentFilter = "all" | AssessmentStatus | "reextraction";
type CitationFilter =
  | ExtractionFilter
  | RetrievalFilter
  | AssessmentFilter;
type FilterOption = { label: string; value: CitationFilter };
type ComparisonMatchType = "perfect" | "exact" | "semantic" | "warning" | "error" | "unchecked";
type TraceSectionId = "retrieval" | "assessment" | "jurisdiction-inference" | "comment";
type TraceOperationStatus = "complete" | "running" | "failed" | "not-run" | "unavailable";
type BibliographicRow = {
  id: string;
  label: string;
  extracted: unknown;
  courtListener: unknown;
  matchType: ComparisonMatchType;
};
type ExtractedCitationAttempt = {
  label: string;
  isReextracted: boolean;
  fields: Record<string, unknown>;
  locator: string | null;
  citation: ReviewCitation;
  reassessment: CaseNameAssessmentPayload | null;
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
  canRetrieve: boolean;
  canAssess: boolean;
  canReset: boolean;
};

const RETRIEVAL_REQUEST_INTERVAL_MS = 1500;
const MISSING_EXTRACTED_CASE_NAME_LABEL = "No extracted case name";
const exampleText = "Brown v. Board, 347 U.S. 483 (1954). See also Roe v. Wade, 410 U.S. 113.";
const extractionFilters: FilterOption[] = [
  { label: "All", value: "all" },
  { label: "All citations", value: "all_citations" }
];
const retrievalFilters: FilterOption[] = [
  { label: "All", value: "all" },
  { label: "Found", value: "found" },
  { label: "Ambiguous", value: "ambiguous" },
  { label: "Not found", value: "not_found" },
  { label: "Throttled", value: "throttled" }
];
const assessmentFilters: FilterOption[] = [
  { label: "All", value: "all" },
  { label: "Exact match", value: "exact_match" },
  { label: "Semantic match", value: "semantic_match" },
  { label: "Not semantic match", value: "not_semantic_match" },
  { label: "Irregular form", value: "irregular_form" },
  { label: "Different case", value: "different_case" },
  { label: "Unassessable", value: "unassessable" },
  { label: "Re-extraction", value: "reextraction" }
];
const workflowStageControls: Record<WorkflowStage, WorkflowStageControl> = {
  input: {
    label: "Input",
    canEditInput: true,
    canLoadSnapshot: true,
    canExtract: true,
    canRetrieve: false,
    canAssess: false,
    canReset: false
  },
  preprocessed: {
    label: "Preprocessed",
    canEditInput: false,
    canLoadSnapshot: false,
    canExtract: true,
    canRetrieve: false,
    canAssess: false,
    canReset: true
  },
  extracted: {
    label: "Extracted",
    canEditInput: false,
    canLoadSnapshot: false,
    canExtract: false,
    canRetrieve: true,
    canAssess: false,
    canReset: true
  },
  node_graph: {
    label: "Node graph",
    canEditInput: false,
    canLoadSnapshot: false,
    canExtract: false,
    canRetrieve: true,
    canAssess: false,
    canReset: true
  },
  inferred: {
    label: "Inferred",
    canEditInput: false,
    canLoadSnapshot: false,
    canExtract: false,
    canRetrieve: true,
    canAssess: false,
    canReset: true
  },
  retrieved: {
    label: "Retrieved",
    canEditInput: false,
    canLoadSnapshot: false,
    canExtract: false,
    canRetrieve: false,
    canAssess: true,
    canReset: true
  },
  assessed: {
    label: "Assessed",
    canEditInput: false,
    canLoadSnapshot: false,
    canExtract: false,
    canRetrieve: false,
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
  const [retrievalProgress, setRetrievalProgress] = useState<{ completed: number; total: number } | null>(
    null
  );
  const [workflowStage, setWorkflowStage] = useState<WorkflowStage>("input");
  const [isInputCollapsed, setIsInputCollapsed] = useState(false);
  const [citationFilter, setCitationFilter] = useState<CitationFilter>("all");
  const [clusterIndexes, setClusterIndexes] = useState<Record<string, number>>({});
  const [extractedAttemptIndexes, setExtractedAttemptIndexes] = useState<Record<string, number>>({});
  const [isDetailsExpanded, setIsDetailsExpanded] = useState(false);
  const [selectedTraceSection, setSelectedTraceSection] = useState<TraceSectionId>("retrieval");
  const [error, setError] = useState<string | null>(null);
  const [bookmarkStatus, setBookmarkStatus] = useState<string | null>(null);
  const [bookmarkEntries, setBookmarkEntries] = useState<Map<string, BookmarkStatusEntry>>(
    new Map()
  );
  const [bookmarkModal, setBookmarkModal] = useState<{
    citationId: string;
    matchedText: string;
    context: string;
    existingComment: string | null;
  } | null>(null);
  const documentPaneRef = useRef<HTMLElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const snapshotInputRef = useRef<HTMLInputElement | null>(null);
  const retrievalRunIdRef = useRef(0);
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
  const selectedCitationIndex = useMemo(
    () => result?.citations.findIndex((citation) => citation.id === selectedId) ?? -1,
    [result, selectedId]
  );
  const isBookmarkFixture = isBookmarkFixturePath(result?.document.source_path ?? null);
  const selectedCitationEntry = selectedCitation
    ? citationBookmarkEntry(selectedCitation, bookmarkEntries)
    : undefined;
  const selectedCitationIsBookmarked = selectedCitationEntry?.bookmarked ?? false;
  const selectedCitationComment = selectedCitationEntry?.comment ?? null;

  useEffect(() => {
    const citations = result?.citations ?? [];
    const documentText = result?.document.text ?? "";
    if (!citations.length) {
      setBookmarkEntries(new Map());
      return;
    }
    let cancelled = false;
    void fetchBookmarkStatuses(citations, documentText)
      .then((statuses) => {
        if (!cancelled) {
          setBookmarkEntries(new Map(Object.entries(statuses)));
        }
      })
      .catch(() => {
        if (!cancelled) {
          setBookmarkEntries(new Map());
        }
      });
    return () => {
      cancelled = true;
    };
  }, [result?.citations, result?.document.text]);

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

  const fullCaseCitations = useMemo(
    () => primaryCitations.filter(isFullCaseCitation),
    [primaryCitations]
  );
  const statusCounts = useMemo(() => citationStatusCounts(fullCaseCitations), [fullCaseCitations]);
  const assessmentCounts = useMemo(
    () => assessmentStatusCounts(fullCaseCitations, result?.assessment ?? null),
    [fullCaseCitations, result]
  );
  const reextractionCount = useMemo(
    () =>
      fullCaseCitations.filter((citation) =>
        citationHasReextraction(citation, result?.assessment ?? null)
      ).length,
    [fullCaseCitations, result]
  );
  const assessmentCandidateCount = useMemo(
    () => fullCaseCitations.filter(isAssessmentCandidateCitation).length,
    [fullCaseCitations]
  );
  const stageFullCaseTotal =
    workflowStage === "assessed" ? assessmentCandidateCount : fullCaseCitations.length;

  const availableFilters = useMemo(
    () =>
      stageFilterOptions(
        workflowStage,
        statusCounts,
        assessmentCounts,
        stageFullCaseTotal,
        allCitations.length,
        reextractionCount
      ),
    [allCitations.length, assessmentCounts, stageFullCaseTotal, statusCounts, reextractionCount, workflowStage]
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
        if (workflowStage === "assessed") {
          if (citationFilter === "reextraction") {
            return citationHasReextraction(citation, result?.assessment ?? null);
          }
          if (isAssessmentStatusFilter(citationFilter)) {
            return (
              assessmentStatusFromPayload(effectiveAssessment(citation, result?.assessment ?? null)) ===
              citationFilter
            );
          }
          return citationFilter === "all" && isAssessmentCandidateCitation(citation);
        }
        if (workflowStage === "retrieved" && isRetrievalStatusFilter(citationFilter)) {
          return citationStatus(citation) === citationFilter;
        }
        return citationFilter === "all";
      }),
    [citationFilter, renderCitations, result, workflowStage]
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
    retrievalRunIdRef.current += 1;
    setIsLoading(true);
    setLoadingStage("extracted");
    setRetrievalProgress(null);
    setError(null);
    setBookmarkStatus(null);
    setBookmarkEntries(new Map());

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

  async function runRetrieval() {
    if (!result) {
      return;
    }
    const citationsToRetrieve = result.citations.filter(isFullCaseCitation);
    if (!citationsToRetrieve.length) {
      setResult(withRetrievalStats(result));
      setWorkflowStage("retrieved");
      setCitationFilter("all");
      return;
    }

    const runId = retrievalRunIdRef.current + 1;
    retrievalRunIdRef.current = runId;
    setIsLoading(true);
    setLoadingStage("retrieved");
    setRetrievalProgress({ completed: 0, total: citationsToRetrieve.length });
    setError(null);

    try {
      const failures: string[] = [];
      await Promise.all(
        citationsToRetrieve.map(async (citation, index) => {
          if (index > 0) {
            await wait(RETRIEVAL_REQUEST_INTERVAL_MS * index);
          }
          if (retrievalRunIdRef.current !== runId) {
            return;
          }

          try {
            const retrieval = await retrieveReviewCitation(citation);
            if (retrievalRunIdRef.current !== runId) {
              return;
            }
            setResult((current) => (current ? mergeCitationRetrieval(current, retrieval) : current));
          } catch (err) {
            failures.push(err instanceof Error ? err.message : "Retrieval failed");
          } finally {
            if (retrievalRunIdRef.current === runId) {
              setRetrievalProgress((current) =>
                current ? { ...current, completed: Math.min(current.completed + 1, current.total) } : current
              );
            }
          }
        })
      );

      if (retrievalRunIdRef.current !== runId) {
        return;
      }
      setResult((current) => (current ? withRetrievalStats(current) : current));
      setCitationFilter("all");
      if (failures.length) {
        setError(`Retrieval finished with ${failures.length} failed request${failures.length === 1 ? "" : "s"}.`);
        setWorkflowStage("extracted");
      } else {
        setWorkflowStage("retrieved");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Retrieval failed");
    } finally {
      if (retrievalRunIdRef.current === runId) {
        setIsLoading(false);
        setLoadingStage(null);
        setRetrievalProgress(null);
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
    retrievalRunIdRef.current += 1;
    setIsLoading(true);
    setLoadingStage("snapshot");
    setRetrievalProgress(null);
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
    retrievalRunIdRef.current += 1;
    setResult(null);
    setSelectedId(null);
    setCitationFilter("all");
    setClusterIndexes({});
    setExtractedAttemptIndexes({});
    setWorkflowStage("input");
    setIsInputCollapsed(false);
    setIsDetailsExpanded(false);
    setRetrievalProgress(null);
    setError(null);
    setBookmarkStatus(null);
    setBookmarkEntries(new Map());
  }

  function selectCitation(citationId: string) {
    setSelectedId(citationId);
    setBookmarkStatus(null);
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

  function openBookmarkModalForSelected() {
    if (!result || !selectedCitation || isBookmarkFixture) {
      return;
    }
    const context = citationContextWindow(result.document.text, selectedCitation);
    setBookmarkModal({
      citationId: selectedCitation.id,
      matchedText: selectedCitation.matched_text,
      context,
      existingComment: selectedCitationComment
    });
  }

  function closeBookmarkModal() {
    setBookmarkModal(null);
  }

  async function submitBookmarkModal(comment: string | null) {
    if (!result || !bookmarkModal) {
      return;
    }
    const trimmed = comment?.trim() ? comment.trim() : null;
    const isUpdate = bookmarkModal.existingComment !== null;
    setBookmarkStatus("Saving");
    try {
      if (isUpdate) {
        const saved = await updateBookmarkCommentAction({
          citation: {
            matched_text: bookmarkModal.matchedText,
            context: bookmarkModal.context
          },
          comment: trimmed
        });
        if (saved.result === "not_found") {
          setBookmarkStatus("Bookmark no longer exists; reopen the modal to recreate it.");
          return;
        }
        setBookmarkEntries((current) => {
          const next = new Map(current);
          next.set(bookmarkModal.citationId, {
            bookmarked: true,
            comment: saved.comment
          });
          return next;
        });
        setBookmarkStatus(
          saved.result === "unchanged"
            ? "Comment unchanged"
            : `Updated comment in ${saved.paths.json}`
        );
      } else {
        const saved = await addBookmarkCitation({
          citation: {
            matched_text: bookmarkModal.matchedText,
            context: bookmarkModal.context
          },
          provenance: {
            source_path: result.document.source_path,
            source_format: result.document.source_format,
            span: { start: selectedCitation?.start ?? 0, end: selectedCitation?.end ?? 0 }
          },
          comment: trimmed
        });
        setBookmarkEntries((current) => {
          const next = new Map(current);
          next.set(bookmarkModal.citationId, {
            bookmarked: true,
            comment: saved.comment
          });
          return next;
        });
        setBookmarkStatus(
          saved.result === "provenance_added"
            ? `Added provenance to ${saved.paths.json}`
            : `Saved to ${saved.paths.text} and ${saved.paths.json}`
        );
      }
      setBookmarkModal(null);
    } catch (bookmarkError) {
      setBookmarkStatus(
        bookmarkError instanceof Error ? bookmarkError.message : "Bookmark failed"
      );
    }
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
  const retrievalLabel =
    loadingStage === "retrieved" && retrievalProgress
      ? `Validating ${retrievalProgress.completed}/${retrievalProgress.total}`
      : "Retrieve";

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
            disabled={isLoading || !result || !stageControl.canRetrieve}
            onClick={runRetrieval}
            type="button"
          >
            {loadingStage === "retrieved" ? (
              <Loader2 className="spin" size={18} />
            ) : (
              <ShieldCheck size={18} />
            )}
            <span>{retrievalLabel}</span>
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
                    stageFullCaseTotal,
                    allCitations.length,
                    reextractionCount
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
                  <CitationTags
                    citation={citation}
                    assessment={result?.assessment ?? null}
                    workflowStage={workflowStage}
                    bookmarked={isCitationBookmarked(
                      citation,
                      result?.document.text ?? "",
                      bookmarkEntries
                    )}
                  />
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
                  workflowStage={workflowStage}
                  variant="details"
                  bookmarked={selectedCitationIsBookmarked}
                />
              </div>
              <div className="details-actions">
                <button
                  className={`secondary-action compact-action bookmark-action${
                    selectedCitationIsBookmarked ? " bookmarked" : ""
                  }`}
                  type="button"
                  onClick={() => openBookmarkModalForSelected()}
                  disabled={
                    !selectedCitation ||
                    bookmarkStatus === "Saving" ||
                    isBookmarkFixture
                  }
                  title={
                    isBookmarkFixture
                      ? "Bookmarking is disabled while viewing the bookmark test fixture."
                      : selectedCitationIsBookmarked
                        ? "Edit bookmark comment"
                        : "Bookmark this citation context"
                  }
                >
                  {bookmarkStatus === "Saving" ? (
                    <Loader2 className="spin" size={16} />
                  ) : selectedCitationIsBookmarked ? (
                    <BookmarkCheck size={16} />
                  ) : (
                    <Bookmark size={16} />
                  )}
                  <span>
                    {isBookmarkFixture
                      ? "Bookmark fixture"
                      : selectedCitationIsBookmarked
                        ? "Edit bookmark"
                        : "Bookmark context"}
                  </span>
                </button>
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
            {bookmarkStatus ? (
              <p className="bookmark-status" role="status" aria-live="polite">
                {bookmarkStatus}
              </p>
            ) : null}
            <TraceWorkspace
              citation={selectedCitation}
              citationIndex={selectedCitationIndex}
              jurisdictions={result?.jurisdictions ?? null}
              extractedAttempts={selectedExtractedAttempts}
              extractedAttemptIndex={selectedExtractedAttemptIndex}
              candidateIndex={selectedClusterIndex}
              node={result?.node_graph?.nodes.find((node) => node.citation_id === selectedCitation.id) ?? null}
              loadingStage={loadingStage}
              selectedSection={selectedTraceSection}
              onSectionChange={setSelectedTraceSection}
              onCandidateChange={selectCourtListenerCandidate}
              onExtractedAttemptChange={selectExtractedAttempt}
              comment={selectedCitationComment}
              bookmarked={selectedCitationIsBookmarked}
            />
          </>
        ) : (
          <div className="details-empty">Select a citation to inspect fields and retrieval.</div>
        )}
      </section>
      {bookmarkModal ? (
        <BookmarkModal
          matchedText={bookmarkModal.matchedText}
          context={bookmarkModal.context}
          existingComment={bookmarkModal.existingComment}
          saving={bookmarkStatus === "Saving"}
          onSave={(comment) => void submitBookmarkModal(comment)}
          onCancel={closeBookmarkModal}
        />
      ) : null}
    </main>
  );
}

function TraceWorkspace({
  citation,
  citationIndex,
  jurisdictions,
  extractedAttempts,
  extractedAttemptIndex,
  candidateIndex,
  node,
  loadingStage,
  selectedSection,
  onSectionChange,
  onCandidateChange,
  onExtractedAttemptChange,
  comment,
  bookmarked
}: {
  citation: ReviewCitation;
  citationIndex: number;
  jurisdictions: JurisdictionPayload[] | null;
  extractedAttempts: ExtractedCitationAttempt[];
  extractedAttemptIndex: number;
  candidateIndex: number;
  node: CitationNodePayload | null;
  loadingStage: LoadingStage | null;
  selectedSection: TraceSectionId;
  onSectionChange: (section: TraceSectionId) => void;
  onCandidateChange: (citationId: string, candidateIndex: number) => void;
  onExtractedAttemptChange: (citationId: string, attemptIndex: number) => void;
  comment: string | null;
  bookmarked: boolean;
}) {
  const sections: Array<{ id: TraceSectionId; label: string; status: TraceOperationStatus }> = [
    {
      id: "retrieval",
      label: "Retrieval",
      status: retrievalOperationStatus(citation.retrieval, loadingStage)
    },
    {
      id: "assessment",
      label: "Assessment",
      status: assessmentOperationStatus(citation.assessment, loadingStage)
    },
    {
      id: "jurisdiction-inference",
      label: "Jurisdiction inference",
      status: citationIndex >= 0 && jurisdictions && citationIndex < jurisdictions.length ? "complete" : "unavailable"
    },
    {
      id: "comment",
      label: "Comment",
      status: comment?.trim() ? "complete" : bookmarked ? "not-run" : "unavailable"
    }
  ];

  function moveTab(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") {
      return;
    }
    event.preventDefault();
    const direction = event.key === "ArrowRight" ? 1 : -1;
    const nextIndex = (index + direction + sections.length) % sections.length;
    onSectionChange(sections[nextIndex].id);
    const tabList = event.currentTarget.parentElement;
    requestAnimationFrame(() => {
      (tabList?.querySelectorAll<HTMLButtonElement>('[role="tab"]')[nextIndex])?.focus();
    });
  }

  return (
    <div className="trace-workspace">
      <div className="bibliographic-column">
        <BibliographicComparison
          citation={citation}
          extractedAttempts={extractedAttempts}
          extractedAttemptIndex={extractedAttemptIndex}
          clusterIndex={candidateIndex}
          onClusterChange={onCandidateChange}
          onExtractedAttemptChange={onExtractedAttemptChange}
        />
      </div>

      <section className="trace-region" aria-label="Citation traces">
        <div className="trace-nav" role="tablist" aria-label="Trace sections" aria-orientation="horizontal">
          {sections.map((section, index) => (
            <button
              key={section.id}
              id={`trace-tab-${section.id}`}
              className={`trace-tab ${section.status}${
                selectedSection === section.id ? " selected" : ""
              }`}
              type="button"
              role="tab"
              aria-selected={selectedSection === section.id}
              aria-controls={`trace-panel-${section.id}`}
              tabIndex={selectedSection === section.id ? 0 : -1}
              onClick={() => onSectionChange(section.id)}
              onKeyDown={(event) => moveTab(event, index)}
            >
              <span>{section.label}</span>
              <small>
                <i aria-hidden="true" />
                {formatTraceOperationStatus(section.status)}
              </small>
            </button>
          ))}
        </div>
        <div
          id={`trace-panel-${selectedSection}`}
          className="trace-panel"
          role="tabpanel"
          aria-labelledby={`trace-tab-${selectedSection}`}
          tabIndex={0}
        >
          {selectedSection === "retrieval" ? (
            <RetrievalDetails
              retrieval={citation.retrieval}
              citationId={citation.id}
              candidateIndex={candidateIndex}
              onCandidateChange={onCandidateChange}
            />
          ) : null}
          {selectedSection === "assessment" ? (
            <AssessmentDetails
              assessment={citation.assessment}
              candidateIndex={candidateIndex}
              candidateId={candidateIdForIndex(citation.retrieval, candidateIndex)}
            />
          ) : null}
          {selectedSection === "jurisdiction-inference" && citationIndex >= 0 && jurisdictions ? (
            <JurisdictionDetails inference={jurisdictions[citationIndex] ?? null} />
          ) : null}
          {selectedSection === "comment" ? (
            <CommentDetails comment={comment} bookmarked={bookmarked} />
          ) : null}
        </div>
        <CitationNodeGraph
          node={node}
          citation={citation}
          jurisdiction={citationIndex >= 0 && jurisdictions ? jurisdictions[citationIndex] ?? null : null}
          loadingStage={loadingStage}
        />
      </section>
    </div>
  );
}

function CitationNodeGraph({
  node,
  citation,
  jurisdiction,
  loadingStage
}: {
  node: CitationNodePayload | null;
  citation: ReviewCitation;
  jurisdiction: JurisdictionPayload | null;
  loadingStage: LoadingStage | null;
}) {
  const steps = effectiveNodeGraphSteps(node, citation, jurisdiction);
  const graphItems = useMemo<CitationNodeGraphItem[]>(() => {
    const extractionData = node
      ? { input: node.input }
      : {
          citation_id: citation.id,
          matched_text: citation.matched_text,
          citation: citation.fields
        };
    return [
      {
        key: "extraction.input",
        step_id: "extraction.input",
        operation: "extraction.input",
        status: "succeeded",
        depends_on: [],
        lane: null,
        summary: node?.input.matched_text ?? citation.matched_text,
        data: extractionData,
        error: null
      },
      ...steps.map((step, index) => ({
        ...step,
        key: step.step_id ?? `${step.operation}-${index}`
      }))
    ];
  }, [citation.fields, citation.id, citation.matched_text, node, steps]);
  const [selectedStepKey, setSelectedStepKey] = useState<string>("extraction.input");
  useEffect(() => {
    if (!graphItems.some((item) => item.key === selectedStepKey)) {
      setSelectedStepKey(graphItems[0]?.key ?? "extraction.input");
    }
  }, [graphItems, selectedStepKey]);
  const selectedStep = graphItems.find((item) => item.key === selectedStepKey) ?? graphItems[0] ?? null;
  const hasGraph = Boolean(node) || steps.length > 0;
  const statusLabel = node ? formatNodeStatus(node.status) : steps.length ? "Derived" : null;
  return (
    <section className="node-graph-panel" aria-label="Citation node graph">
      <div className="node-graph-heading">
        <div>
          <h3>Node graph</h3>
          <p>
            {hasGraph
              ? `${statusLabel} · ${steps.length} step${steps.length === 1 ? "" : "s"}`
              : "No node trace is attached to this citation yet."}
          </p>
        </div>
        {node ? <span className={`node-status-pill ${node.status}`}>{formatNodeStatus(node.status)}</span> : null}
      </div>
      {hasGraph ? (
        <div className="node-graph-workspace">
          <ol className="node-graph">
            {graphItems.map((item) => (
              <li
                className={`node-graph-item ${item.status}${item.key === selectedStep?.key ? " selected" : ""}`}
                key={item.key}
              >
                <button
                  className="node-graph-button"
                  type="button"
                  onClick={() => setSelectedStepKey(item.key)}
                  aria-current={item.key === selectedStep?.key ? "step" : undefined}
                >
                  <span className="node-graph-dot" aria-hidden="true" />
                  <span>
                    <strong>{formatNodeOperation(item.operation)}</strong>
                    <small>{item.summary}</small>
                    {item.lane ? <em>lane: {item.lane}</em> : null}
                    {item.error ? <em>{item.error}</em> : null}
                  </span>
                </button>
              </li>
            ))}
            {!steps.length && loadingStage ? (
              <li className="node-graph-item blocked">
                <button className="node-graph-button" type="button" disabled>
                  <span className="node-graph-dot" aria-hidden="true" />
                  <span>
                    <strong>{loadingStage}.pending</strong>
                    <small>Waiting for this node to receive its next trace step.</small>
                  </span>
                </button>
              </li>
            ) : null}
          </ol>
          {selectedStep ? <CitationNodeStepInspector step={selectedStep} /> : null}
        </div>
      ) : (
        <p className="node-graph-empty">
          Load a citation-node snapshot or run a stage that emits node traces.
        </p>
      )}
    </section>
  );
}

function CitationNodeStepInspector({ step }: { step: CitationNodeGraphItem }) {
  return (
    <aside className="node-step-inspector" aria-label="Selected node step details">
      <div className="node-step-inspector-heading">
        <div>
          <h4>{formatNodeOperation(step.operation)}</h4>
          <p>{step.summary}</p>
        </div>
        <span className={`node-step-chip ${step.status}`}>{formatStatusLabel(step.status)}</span>
      </div>
      <dl className="node-step-meta">
        <div>
          <dt>Step id</dt>
          <dd>{step.step_id ?? "derived"}</dd>
        </div>
        <div>
          <dt>Depends on</dt>
          <dd>{step.depends_on.length ? step.depends_on.join(", ") : "none"}</dd>
        </div>
        <div>
          <dt>Lane</dt>
          <dd>{step.lane ?? "main"}</dd>
        </div>
      </dl>
      {step.error ? <p className="node-step-error">{step.error}</p> : null}
      <pre className="node-step-data">{JSON.stringify(step.data, null, 2)}</pre>
    </aside>
  );
}

function effectiveNodeGraphSteps(
  node: CitationNodePayload | null,
  citation: ReviewCitation,
  jurisdiction: JurisdictionPayload | null
): CitationNodeStepPayload[] {
  const steps = [...(node?.steps ?? [])];
  const hasOperation = (operation: string) => steps.some((step) => step.operation === operation);
  if (jurisdiction && !hasOperation("jurisdiction.inference")) {
    steps.push({
      step_id: "derived:jurisdiction",
      operation: "jurisdiction.inference",
      status: "succeeded",
      depends_on: [],
      lane: null,
      summary: `Reporter ${jurisdiction.reporter_inference.status}; court ${jurisdiction.court_inference.status}.`,
      data: { jurisdiction },
      error: null
    });
  }
  if (citation.retrieval && !hasOperation("retrieval.exact_lookup")) {
    steps.push({
      step_id: "derived:retrieval:exact_lookup",
      operation: "retrieval.exact_lookup",
      status:
        citation.retrieval.status === "lookup_failed" || citation.retrieval.status === "throttled"
          ? "failed"
          : citation.retrieval.status === "skipped" || citation.retrieval.status === "invalid"
            ? "skipped"
            : "succeeded",
      depends_on: [],
      lane: null,
      summary: `Exact locator lookup returned ${citation.retrieval.status}.`,
      data: { retrieval: citation.retrieval },
      error: citation.retrieval.request_trace?.error_message ?? null
    });
  }
  if (citation.assessment && !hasOperation("assessment.field_check")) {
    steps.push({
      step_id: "derived:assessment",
      operation: "assessment.field_check",
      status:
        citation.assessment.status === "failed"
          ? "failed"
          : citation.assessment.status === "waiting"
            ? "blocked"
            : citation.assessment.status === "skipped"
              ? "skipped"
              : "succeeded",
      depends_on: [],
      lane: null,
      summary: `Assessment status is ${citation.assessment.status}.`,
      data: { assessment: citation.assessment },
      error: citation.assessment.status === "failed" ? citation.assessment.error : null
    });
  }
  return steps;
}

function formatNodeStatus(status: CitationNodePayload["status"]): string {
  return status
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatNodeOperation(operation: string): string {
  return operation
    .split(".")
    .map((part) => part.replaceAll("_", " "))
    .join(" → ");
}

function retrievalOperationStatus(
  retrieval: RetrievalPayload | null,
  loadingStage: LoadingStage | null
): TraceOperationStatus {
  if (retrieval) {
    return retrieval.status === "lookup_failed" || retrieval.status === "throttled"
      ? "failed"
      : "complete";
  }
  return loadingStage === "retrieved" ? "running" : "not-run";
}

function assessmentOperationStatus(
  assessment: CitationAssessmentPayload | null,
  loadingStage: LoadingStage | null
): TraceOperationStatus {
  if (assessment) {
    return assessment.status === "waiting" ? "running" : "complete";
  }
  return loadingStage === "assessed" ? "running" : "not-run";
}

function formatTraceOperationStatus(status: TraceOperationStatus) {
  return status === "not-run" ? "Not run" : formatStatusLabel(status);
}

function CommentDetails({ comment, bookmarked }: { comment: string | null; bookmarked: boolean }) {
  return (
    <div className="detail-group comment-detail-group">
      <div className="assessment-trace-heading">
        <h3>Review comment</h3>
        <p>Notes saved with this citation context.</p>
      </div>
      <div className={`comment-note${comment?.trim() ? " has-comment" : ""}`}>
        {comment?.trim() ? (
          <p>{comment}</p>
        ) : (
          <p>
            {bookmarked
              ? "No comment yet. Use Edit bookmark to add one."
              : "Bookmark this citation context to save a comment."}
          </p>
        )}
      </div>
    </div>
  );
}

function JurisdictionDetails({
  inference
}: {
  inference: JurisdictionPayload | null;
}) {
  if (!inference) {
    return (
      <div className="detail-group trace-empty-state">
        <h3>Jurisdiction inference</h3>
        <p>This citation type has no jurisdiction inference.</p>
      </div>
    );
  }

  const { reporter_inference, court_inference } = inference;

  return (
    <div className="detail-group assessment-trace-group">
      <div className="assessment-trace-heading">
        <h3>Jurisdiction inference</h3>
        <p>Triad taxonomy components for inferring citation jurisdiction.</p>
      </div>

      <div className="assessment-trace-heading" style={{ marginTop: '16px' }}>
        <h4 style={{ margin: '0', fontSize: '14px', color: 'var(--accent-fg)' }}>Reporter Inference</h4>
      </div>
      <dl className="assessment-fields">
        <div><dt>Edition</dt><dd>{formatValue(reporter_inference.reporter?.edition_short_name ?? null)}</dd></div>
        <div><dt>Name</dt><dd>{formatValue(reporter_inference.reporter?.name ?? null)}</dd></div>
        <div><dt>Cite type</dt><dd>{formatValue(reporter_inference.reporter?.cite_type ?? null)}</dd></div>
        <div><dt>Status</dt><dd>{formatStatusLabel(reporter_inference.status)}</dd></div>
      </dl>
      {reporter_inference.mlz_jurisdictions.length > 0 && (
        <dl className="assessment-fields" style={{ marginTop: '0' }}>
          <div>
            <dt>MLZ Jurisdictions</dt>
            <dd>
              <ul style={{ margin: '0', paddingLeft: '16px', color: 'var(--text-secondary)' }}>
                {reporter_inference.mlz_jurisdictions.slice(0, 3).map((j) => (
                  <li key={j}>{j}</li>
                ))}
                {reporter_inference.mlz_jurisdictions.length > 3 && (
                  <li>... ({reporter_inference.mlz_jurisdictions.length} in total)</li>
                )}
              </ul>
            </dd>
          </div>
        </dl>
      )}

      <div className="assessment-trace-heading" style={{ marginTop: '16px' }}>
        <h4 style={{ margin: '0', fontSize: '14px', color: 'var(--accent-fg)' }}>Court Inference</h4>
      </div>
      <dl className="assessment-fields">
        <div><dt>Extracted Court</dt><dd>{formatValue(court_inference.extracted_court)}</dd></div>
        <div><dt>Status</dt><dd>{formatStatusLabel(court_inference.status)}</dd></div>
      </dl>
      {court_inference.courts_db_classification && (
        <dl className="assessment-fields" style={{ marginTop: '0' }}>
          <div><dt>Court ID</dt><dd>{formatValue(court_inference.courts_db_classification.court_id)}</dd></div>
          <div><dt>System</dt><dd>{formatValue(court_inference.courts_db_classification.system)}</dd></div>
          <div><dt>Jurisdiction</dt><dd>{formatValue(court_inference.courts_db_classification.jurisdiction)}</dd></div>
          <div><dt>Type</dt><dd>{formatValue(court_inference.courts_db_classification.type)}</dd></div>
        </dl>
      )}
    </div>
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
  workflowStage,
  variant = "rail",
  bookmarked = false
}: {
  citation: ReviewCitation;
  assessment?: ReviewAssessment | null;
  workflowStage: WorkflowStage;
  variant?: "rail" | "details";
  bookmarked?: boolean;
}) {
  // The event-level status rolls up to the re-extraction result when one exists,
  // so a recovered citation reads as its final (re-extracted) verdict.
  const effective = effectiveAssessment(citation, assessment);
  const assessmentStatus = assessmentStatusFromPayload(effective);
  const showRetrieval = Boolean(
    citation.retrieval && (workflowStage === "retrieved" || variant === "details")
  );
  const showAssessment = Boolean(
    assessmentStatus && (workflowStage === "assessed" || variant === "details")
  );
  const showKind = workflowStage !== "assessed";
  const showYearMismatch = Boolean(
    effective?.year.status.toLowerCase().replaceAll("-", "_") === "mismatch" &&
      (workflowStage === "assessed" || variant === "details")
  );
  const courtFinalStatus = effective
    ? effective.court?.status.toLowerCase().replaceAll("-", "_")
    : undefined;
  const showCourtMismatch = Boolean(
    courtFinalStatus === "mismatch" && (workflowStage === "assessed" || variant === "details")
  );
  return (
    <span className={`citation-tags ${variant}`} aria-label="Citation labels">
      {showKind ? <span className="citation-tag kind">{citation.kind}</span> : null}
      {bookmarked ? <span className="citation-tag bookmark">Bookmarked</span> : null}
      {showRetrieval && citation.retrieval ? (
        <span className={`citation-tag retrieval ${citationStatus(citation)}`}>
          {formatStatusLabel(citation.retrieval.status)}
        </span>
      ) : null}
      {showAssessment && effective && assessmentStatus ? (
        <span className={`citation-tag assessment ${assessmentStatus}`}>
          {formatAssessmentLabel(assessmentStatus)}
        </span>
      ) : null}
      {showCourtMismatch ? <span className="citation-tag court mismatch">Court mismatch</span> : null}
      {showYearMismatch ? <span className="citation-tag year mismatch">Year mismatch</span> : null}
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
  const clusters = comparableCourtListenerClusters(citation.retrieval);
  const safeClusterIndex = clusters.length
    ? Math.min(Math.max(clusterIndex, 0), clusters.length - 1)
    : 0;
  const cluster = clusters[safeClusterIndex] ?? null;
  const safeExtractedAttemptIndex = extractedAttempts.length
    ? Math.min(Math.max(extractedAttemptIndex, 0), extractedAttempts.length - 1)
    : 0;
  const extractedAttempt = extractedAttempts[safeExtractedAttemptIndex] ?? extractedAttempts[0];
  const courtResolution = courtResolutionForCandidate(
    citation.retrieval,
    safeClusterIndex
  );
  const activeResult = activeAssessmentResult(citation.assessment, safeClusterIndex);
  const rows = bibliographicRows(extractedAttempt, cluster, courtResolution, activeResult);
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
            <span>Extracted</span>
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
          <div className={`comparison-row ${row.matchType}`} key={row.id} role="row">
            <span role="cell">{row.label}</span>
            <span role="cell">
              {row.id === "case_name" && hasMultipleExtractedAttempts ? (
                <span className="row-attempt-switcher" aria-label="Re-extraction selector">
                  <button
                    aria-label="Show previous extracted case name"
                    type="button"
                    onClick={() => changeExtractedAttempt(-1)}
                  >
                    <ChevronLeft size={13} aria-hidden="true" />
                  </button>
                  <span className="row-attempt-value">{formatValue(row.extracted)}</span>
                  <button
                    aria-label="Show next extracted case name"
                    type="button"
                    onClick={() => changeExtractedAttempt(1)}
                  >
                    <ChevronRight size={13} aria-hidden="true" />
                  </button>
                </span>
              ) : (
                formatValue(row.extracted)
              )}
            </span>
            <span role="cell">{formatValue(row.courtListener)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function RetrievalDetails({
  retrieval,
  citationId,
  candidateIndex,
  onCandidateChange
}: {
  retrieval: RetrievalPayload | null;
  citationId: string;
  candidateIndex: number;
  onCandidateChange: (citationId: string, candidateIndex: number) => void;
}) {
  if (!retrieval) {
    return (
      <div className="detail-group">
        <h3>Retrieval trace</h3>
        <dl>
          <div>
            <dt>Status</dt>
            <dd>Not requested</dd>
          </div>
        </dl>
      </div>
    );
  }

  const candidates = ambiguousRetrievalCandidates(retrieval);
  const safeCandidateIndex = candidates.length
    ? Math.min(Math.max(candidateIndex, 0), candidates.length - 1)
    : 0;
  const selectedCandidate = candidates[safeCandidateIndex] ?? null;
  const resolution = selectedCandidate?.court_resolution ?? retrieval.candidate?.court_resolution;
  const search = retrieval.candidate_search;

  function changeCandidate(direction: -1 | 1) {
    if (candidates.length < 2) {
      return;
    }
    const nextIndex =
      (safeCandidateIndex + direction + candidates.length) % candidates.length;
    onCandidateChange(citationId, nextIndex);
  }

  return (
    <div className="detail-group assessment-trace-group">
      <div className="assessment-trace-heading">
        <div>
          <h3>Retrieval trace</h3>
          <p>
            CourtListener existence lookup, then court resolution or case-name search
            {selectedCandidate
              ? ` — candidate ${safeCandidateIndex + 1}/${candidates.length}`
              : ""}
          </p>
        </div>
        {candidates.length > 1 ? (
          <span className="case-switcher" aria-label="Retrieval candidate selector">
            <button
              aria-label="Show previous retrieval candidate"
              type="button"
              onClick={() => changeCandidate(-1)}
            >
              <ChevronLeft size={14} aria-hidden="true" />
            </button>
            <strong>
              {safeCandidateIndex + 1}/{candidates.length}
            </strong>
            <button
              aria-label="Show next retrieval candidate"
              type="button"
              onClick={() => changeCandidate(1)}
            >
              <ChevronRight size={14} aria-hidden="true" />
            </button>
          </span>
        ) : null}
      </div>
      <ol className="assessment-trace">
        <AssessmentTraceStep index="1" title="Citation lookup" status={retrieval.status}>
          <RetrievalLookupDetails retrieval={retrieval} />
        </AssessmentTraceStep>
        {resolution ? (
          <AssessmentTraceStep index="2" title="Court resolution" status={resolution.resolved_via}>
            <CourtResolutionDetails resolution={resolution} />
          </AssessmentTraceStep>
        ) : search ? (
          <AssessmentTraceStep index="2" title="Case-name search" status={search.status}>
            <CaseNameSearchDetails search={search} />
          </AssessmentTraceStep>
        ) : (
          <AssessmentTraceStep index="2" title="Court resolution" status="not_attempted">
            <p className="assessment-step-note">
              Court resolution runs only when the citation is found in CourtListener.
            </p>
          </AssessmentTraceStep>
        )}
      </ol>
    </div>
  );
}

function CaseNameSearchDetails({ search }: { search: CaseNameSearchTracePayload }) {
  return (
    <dl className="assessment-fields">
      {search.query ? (
        <div className="search-query-field">
          <dt>Search query</dt>
          <dd>{search.query}</dd>
        </div>
      ) : null}
      {search.probes.map((probe) => (
        <div key={probe.corpus}>
          <dt>{probe.corpus === "o" ? "Opinion search" : "RECAP search"}</dt>
          <dd className="search-probe-result">
            <span>{probe.status === "searched" ? `${probe.case_count} cases` : formatStatusLabel(probe.status)}</span>
            <HttpMetadata trace={probe.request_trace} />
            {probe.candidates.length ? (
              <ol>
                {probe.candidates.map((candidate, index) => (
                  <li key={`${probe.corpus}-${candidate.cluster_id ?? candidate.docket_id ?? index}`}>
                    <strong>{candidate.case_name ?? "Unnamed case"}</strong>
                    <span>
                      {[candidate.court_id, candidate.date_filed, candidate.docket_number]
                        .filter(Boolean)
                        .join(" · ")}
                    </span>
                  </li>
                ))}
              </ol>
            ) : null}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function RetrievalLookupDetails({ retrieval }: { retrieval: RetrievalPayload }) {
  const requestTrace = retrieval.request_trace;
  return (
    <dl className="assessment-fields">
      <div>
        <dt>Source</dt>
        <dd>{retrieval.source}</dd>
      </div>
      <div>
        <dt>Locator</dt>
        <dd>{formatValue(retrieval.locator)}</dd>
      </div>
      <div>
        <dt>Case name</dt>
        <dd>{retrieval.case_names.length ? retrieval.case_names.join(", ") : "-"}</dd>
      </div>
      <div>
        <dt>Status</dt>
        <dd>{formatStatusLabel(retrieval.status)}</dd>
      </div>
      <div>
        <dt>HTTP</dt>
        <dd><HttpMetadata trace={requestTrace} /></dd>
      </div>
    </dl>
  );
}

function CourtResolutionDetails({ resolution }: { resolution: CourtResolutionTracePayload }) {
  return (
    <dl className="assessment-fields">
      <div>
        <dt>CourtListener court</dt>
        <dd>{formatValue(resolution.courtlistener_court_id)}</dd>
      </div>
      <div>
        <dt>Resolved via</dt>
        <dd>{formatStatusLabel(resolution.resolved_via)}</dd>
      </div>
      {resolution.docket_id ? (
        <div>
          <dt>Docket</dt>
          <dd>{resolution.docket_url ?? resolution.docket_id}</dd>
        </div>
      ) : null}
      {resolution.request_trace ? (
        <div>
          <dt>HTTP</dt>
          <dd><HttpMetadata trace={resolution.request_trace} /></dd>
        </div>
      ) : null}
    </dl>
  );
}

function HttpMetadata({ trace }: { trace: CourtListenerRequestTracePayload }) {
  const successful = trace.http_status !== null && trace.http_status >= 200 && trace.http_status < 400;
  return (
    <span className="http-metadata" aria-label="HTTP request metadata">
      <span className={`http-status ${successful ? "success" : "failed"}`}>
        HTTP {trace.http_status ?? "No response"}
      </span>
      <span className="http-cache">Cache {trace.cache ?? "n/a"}</span>
      {trace.error_message ? <span className="http-error">{trace.error_message}</span> : null}
    </span>
  );
}

function AssessmentDetails({
  assessment,
  candidateIndex = 0,
  candidateId = null
}: {
  assessment: CitationAssessmentPayload | null;
  candidateIndex?: number;
  candidateId?: string | null;
}) {
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
            <dd>Run assessment after retrieval to check case-name extraction.</dd>
          </div>
        </dl>
      </div>
    );
  }

  if (assessment.status === "ambiguous") {
    return (
      <AmbiguousAssessmentDetails
        assessment={assessment}
        candidateIndex={candidateIndex}
        candidateId={candidateId}
      />
    );
  }

  if (assessment.status !== "assessed" || !assessment.result) {
    return (
      <div className="detail-group">
        <h3>Citation assessment</h3>
        <dl>
          <div>
            <dt>Status</dt>
            <dd>{formatStatusLabel(assessment.status)}</dd>
          </div>
          {assessment.status === "skipped" ? (
            <div>
              <dt>Message</dt>
              <dd>{assessment.message}</dd>
            </div>
          ) : null}
          {assessment.status === "failed" ? (
            <div>
              <dt>Error</dt>
              <dd>{assessment.error}</dd>
            </div>
          ) : null}
        </dl>
      </div>
    );
  }

  return (
    <div className="detail-group assessment-trace-group">
      <div className="assessment-trace-heading">
        <div>
          <h3>Citation assessment</h3>
          <p>Field-by-field verdicts — expand a field to see its trace</p>
        </div>
      </div>
      <AssessmentResultFields result={assessment.result} />
    </div>
  );
}

function AmbiguousAssessmentDetails({
  assessment,
  candidateIndex,
  candidateId
}: {
  assessment: Extract<CitationAssessmentPayload, { status: "ambiguous" }>;
  candidateIndex: number;
  candidateId: string | null;
}) {
  if (assessment.gated || assessment.candidates.length === 0) {
    return (
      <div className="detail-group">
        <h3>Citation assessment</h3>
        <dl>
          <div>
            <dt>Status</dt>
            <dd>Ambiguous — not enumerated</dd>
          </div>
          <div>
            <dt>Message</dt>
            <dd>{assessment.message || "Too many CourtListener candidates to assess individually."}</dd>
          </div>
        </dl>
      </div>
    );
  }

  const total = assessment.candidates.length;
  const referencedIndex = candidateId
    ? assessment.candidates.findIndex((candidate) => candidate.candidate_id === candidateId)
    : -1;
  const index =
    referencedIndex >= 0
      ? referencedIndex
      : Math.min(Math.max(candidateIndex, 0), total - 1);
  const candidate = assessment.candidates[index];

  return (
    <div className="detail-group assessment-trace-group">
      <div className="assessment-trace-heading">
        <div>
          <h3>Citation assessment</h3>
          <p>
            Ambiguous — candidate {index + 1}/{total} assessed on its own
            {candidate.candidate_id ? ` (${candidate.candidate_id})` : ""}. Use the CourtListener
            candidate switcher above to compare.
          </p>
        </div>
      </div>
      <AssessmentResultFields result={candidate.result} />
    </div>
  );
}

function AssessmentResultFields({ result }: { result: AssessmentPayload }) {
  const { case_name, court, year } = result;
  return (
    <div className="assessment-field-sections">
      <CollapsibleField
        title="Case name"
        subtitle="Initial verdict, correction attempt, and final verdict"
        status={finalCaseNameStatus(case_name)}
      >
        <ol className="assessment-trace">
          <AssessmentTraceStep index="1" title="Initial assessment" status={case_name.initial.status}>
            <CaseNameAssessmentDetails assessment={case_name.initial} />
          </AssessmentTraceStep>
          <CaseNameFollowupDetails followup={case_name.followup} />
        </ol>
      </CollapsibleField>
      <CollapsibleField
        title="Court"
        subtitle="Extracted court against CourtListener"
        status={court.status}
      >
        <CourtAssessmentDetails assessment={court} />
      </CollapsibleField>
      <CollapsibleField title="Year" subtitle="Extracted year against CourtListener" status={year.status}>
        <YearAssessmentDetails assessment={year} />
      </CollapsibleField>
    </div>
  );
}

function CollapsibleField({
  title,
  subtitle,
  status,
  children
}: {
  title: string;
  subtitle?: string;
  status: string;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <section className={`assessment-field-section${open ? " open" : ""}`}>
      <button
        type="button"
        className="assessment-field-toggle"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <span className="assessment-field-label">
          <span className="assessment-field-title">{title}</span>
          {subtitle ? <span className="assessment-field-subtitle">{subtitle}</span> : null}
        </span>
        <span className="assessment-field-toggle-meta">
          <span className={`assessment-step-status ${assessmentStatusTone(status)}`}>
            {formatStatusLabel(status)}
          </span>
          {open ? <ChevronUp size={16} aria-hidden="true" /> : <ChevronDown size={16} aria-hidden="true" />}
        </span>
      </button>
      {open ? <div className="assessment-field-body">{children}</div> : null}
    </section>
  );
}

function finalCaseNameStatus(run: CaseNameAssessmentRunPayload): string {
  return run.followup.status === "reassessed" ? run.followup.result.status : run.initial.status;
}

function CaseNameAssessmentDetails({ assessment }: { assessment: CaseNameAssessmentPayload }) {
  return (
    <dl className="assessment-fields">
      <div>
        <dt>Message</dt>
        <dd>{assessment.message}</dd>
      </div>
      <div>
        <dt>Extracted case</dt>
        <dd>{formatCaseNameValue(assessment.extracted_case_name)}</dd>
      </div>
      <div>
        <dt>CourtListener case</dt>
        <dd>{formatValue(assessment.courtlistener_case_name)}</dd>
      </div>
    </dl>
  );
}

function YearAssessmentDetails({ assessment }: { assessment: YearAssessmentPayload }) {
  return (
    <dl className="assessment-fields">
      <div>
        <dt>Message</dt>
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
    </dl>
  );
}

function CourtAssessmentDetails({ assessment }: { assessment: CourtAssessmentPayload }) {
  return (
    <dl className="assessment-fields">
      <div>
        <dt>Message</dt>
        <dd>{assessment.message}</dd>
      </div>
      <div>
        <dt>Extracted court</dt>
        <dd>{formatValue(assessment.extracted_court)}</dd>
      </div>
      <div>
        <dt>CourtListener docket court</dt>
        <dd>{formatValue(assessment.courtlistener_court_id)}</dd>
      </div>
    </dl>
  );
}

function CaseNameFollowupDetails({ followup }: { followup: CaseNameFollowupPayload }) {
  const hasReextraction = "reextracted_case_name" in followup;
  const reextractionStatus =
    followup.status === "not_required"
      ? "not_required"
      : followup.status === "reextraction_failed"
        ? "failed"
        : "completed";

  return (
    <>
      <AssessmentTraceStep index="2" title="Re-extraction" status={reextractionStatus}>
        {hasReextraction ? (
          <dl className="assessment-fields">
            <div>
              <dt>Case name</dt>
              <dd>{followup.reextracted_case_name.case_name}</dd>
            </div>
            <div>
              <dt>Document span</dt>
              <dd>
                {followup.reextracted_case_name.case_name_span.start}–
                {followup.reextracted_case_name.case_name_span.end}
              </dd>
            </div>
          </dl>
        ) : followup.status === "reextraction_failed" ? (
          <p className="assessment-step-error">{followup.error}</p>
        ) : (
          <p className="assessment-step-note">The initial result did not require correction.</p>
        )}
      </AssessmentTraceStep>
      <AssessmentTraceStep
        index="3"
        title="Reassessment"
        status={
          followup.status === "reassessed"
            ? followup.result.status
            : followup.status === "reassessment_failed"
              ? "failed"
              : followup.status === "reextraction_failed"
                ? "blocked"
                : "not_required"
        }
      >
        {followup.status === "reassessed" ? (
          <CaseNameAssessmentDetails assessment={followup.result} />
        ) : followup.status === "reassessment_failed" ? (
          <p className="assessment-step-error">{followup.error}</p>
        ) : followup.status === "reextraction_failed" ? (
          <p className="assessment-step-note">Reassessment could not run because re-extraction failed.</p>
        ) : (
          <p className="assessment-step-note">The initial result is the final case-name assessment.</p>
        )}
      </AssessmentTraceStep>
    </>
  );
}

function AssessmentTraceStep({
  index,
  title,
  status,
  children
}: {
  index: string;
  title: string;
  status: string;
  children: ReactNode;
}) {
  return (
    <li className="assessment-trace-step">
      <span className="assessment-step-index" aria-hidden="true">
        {index}
      </span>
      <div className="assessment-step-body">
        <div className="assessment-section-heading">
          <h4>{title}</h4>
          <span className={`assessment-step-status ${assessmentStatusTone(status)}`}>
            {formatStatusLabel(status)}
          </span>
        </div>
        {children}
      </div>
    </li>
  );
}

function assessmentStatusTone(status: string) {
  const normalized = status.toLowerCase().replaceAll("-", "_");
  if (
    [
      "exact_match",
      "semantic_match",
      "completed",
      "found",
      "cluster_provided",
      "docket_lookup"
    ].includes(normalized)
  ) {
    return "success";
  }
  if (
    [
      "failed",
      "different_case",
      "not_semantic_match",
      "not_found",
      "invalid",
      "lookup_failed",
      "docket_lookup_failed",
      "search_failed"
    ].includes(normalized)
  ) {
    return "danger";
  }
  if (
    [
      "irregular_form",
      "unassessable",
      "blocked",
      "ambiguous",
      "throttled",
      "no_docket_id",
      "not_attempted",
      "skipped",
      "searched",
      "skipped_no_case_name",
      "skipped_partial_case_name",
      "search_unavailable"
    ].includes(normalized)
  ) {
    return "attention";
  }
  return "neutral";
}

function extractedCitationAttempts(
  citation: ReviewCitation,
  _assessment: ReviewAssessment | null
): ExtractedCitationAttempt[] {
  const original: ExtractedCitationAttempt = {
    label: "Extracted",
    isReextracted: false,
    fields: citation.fields,
    locator: citation.retrieval?.locator ?? citation.matched_text,
    citation,
    reassessment: null,
    span: null
  };
  const result = completedAssessmentResult(citation.assessment);
  const followup = result?.case_name.followup;
  if (!followup || !("reextracted_case_name" in followup)) {
    return [original];
  }
  const item = followup.reextracted_case_name;
  return [
    original,
    {
      label: "Re-extracted",
      isReextracted: true,
      fields: { ...citation.fields, case_name: item.case_name },
      locator: citation.retrieval?.locator ?? citation.matched_text,
      citation,
      reassessment: followup.status === "reassessed" ? followup.result : null,
      span: item.case_name_span
    }
  ];
}

function bibliographicRows(
  extractedAttempt: ExtractedCitationAttempt,
  cluster: CourtListenerCitationRecord | null,
  courtResolution: CourtResolutionTracePayload | null,
  primaryAssessment: AssessmentPayload | null
): BibliographicRow[] {
  const { citation, fields } = extractedAttempt;
  const citationLocator = extractedAttempt.locator ?? citation.retrieval?.locator ?? citation.matched_text;
  const extractedLocatorParts = splitLocator(citationLocator);
  const courtListenerLocator = cluster ? citation.retrieval?.locator : null;
  const courtListenerLocatorParts = splitLocator(courtListenerLocator);
  const extractedCaseName = caseNameFromFields(fields);
  const extractedCaseNameDisplay = extractedCaseName ?? MISSING_EXTRACTED_CASE_NAME_LABEL;
  const courtListenerCaseName = readString(cluster, ["case_name", "caseName"]);
  // The immediate cluster payload rarely carries the court; it is resolved
  // deterministically during retrieval (docket lookup) into court_resolution.
  const courtListenerCourt = cluster
    ? readString(cluster, ["court_id", "court", "courtId"]) ??
      courtResolution?.courtlistener_court_id ??
      null
    : null;
  const courtListenerDate = readString(cluster, ["date_filed", "dateFiled"]);

  return [
    {
      id: "plaintiff",
      label: "Plaintiff",
      extracted: fields.plaintiff ?? null,
      courtListener: null,
      matchType: "unchecked"
    },
    {
      id: "defendant",
      label: "Defendant",
      extracted: fields.defendant ?? null,
      courtListener: null,
      matchType: "unchecked"
    },
    {
      id: "case_name",
      // The re-extraction only revises the case name, so the row itself
      // (not the whole extracted column) reflects that it is a re-extracted value.
      label: extractedAttempt.isReextracted ? "Re-extracted case name" : "Case name",
      extracted: extractedCaseNameDisplay,
      courtListener: courtListenerCaseName,
      matchType: caseNameRowMatchType(
        extractedAttempt.reassessment ?? primaryAssessment?.case_name.initial,
        extractedCaseName,
        courtListenerCaseName
      )
    },
    {
      id: "locator",
      label: "Locator",
      extracted: citationLocator,
      courtListener: courtListenerLocator,
      matchType: directRowMatchType(citationLocator, courtListenerLocator)
    },
    {
      id: "volume",
      label: "Volume",
      extracted: fields.volume ?? extractedLocatorParts.volume,
      courtListener: courtListenerLocatorParts.volume,
      matchType: directRowMatchType(
        fields.volume ?? extractedLocatorParts.volume,
        courtListenerLocatorParts.volume
      )
    },
    {
      id: "reporter",
      label: "Reporter",
      extracted: typeof fields.reporter === "object" && fields.reporter
        ? (fields.reporter as Record<string, unknown>).edition_short_name as string
        : (fields.reporter ?? extractedLocatorParts.reporter),
      courtListener: courtListenerLocatorParts.reporter,
      matchType: directRowMatchType(
        typeof fields.reporter === "object" && fields.reporter
          ? (fields.reporter as Record<string, unknown>).edition_short_name as string
          : (fields.reporter ?? extractedLocatorParts.reporter),
        courtListenerLocatorParts.reporter
      )
    },
    {
      id: "page",
      label: "Page",
      extracted: fields.page ?? extractedLocatorParts.page,
      courtListener: courtListenerLocatorParts.page,
      matchType: directRowMatchType(
        fields.page ?? extractedLocatorParts.page,
        courtListenerLocatorParts.page
      )
    },
    {
      id: "pin_cite",
      label: "Pin cite",
      extracted: fields.pin_cite ?? null,
      courtListener: null,
      matchType: "unchecked"
    },
    {
      id: "year",
      label: "Year",
      extracted: fields.year,
      courtListener: courtListenerDate?.slice(0, 4),
      matchType: yearRowMatchType(
        primaryAssessment?.year,
        fields.year,
        courtListenerDate?.slice(0, 4)
      )
    },
    {
      id: "court",
      label: "Court",
      extracted: fields.court,
      courtListener: courtListenerCourt,
      matchType: courtRowMatchType(primaryAssessment?.court, fields.court, courtListenerCourt)
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
    if (status === "semantic_match") {
      return "semantic";
    }
    if (status === "irregular_form") {
      return "warning";
    }
    if (status === "different_case" || status === "not_semantic_match") {
      return "error";
    }
    if (status === "unassessable") {
      return "warning";
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

function courtRowMatchType(
  assessment: CourtAssessmentPayload | null | undefined,
  extracted: unknown,
  courtListener: unknown,
): ComparisonMatchType {
  return courtAssessmentRowMatchType(assessment, extracted, courtListener);
}

function courtAssessmentRowMatchType(
  assessment: CourtAssessmentPayload | null | undefined,
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

function readString(record: CourtListenerCitationRecord | null, keys: string[]) {
  if (!record) {
    return null;
  }
  const extraData = record.extra_data;
  const records = [
    record,
    typeof extraData === "object" && extraData !== null
      ? (extraData as Record<string, unknown>)
      : null
  ];
  for (const candidate of records) {
    for (const key of keys) {
      const value = candidate?.[key];
      if (typeof value === "string" && value.trim()) {
        return value;
      }
      if (typeof value === "number") {
        return String(value);
      }
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

function formatCaseNameValue(value: unknown) {
  if (!hasDisplayValue(value)) {
    return MISSING_EXTRACTED_CASE_NAME_LABEL;
  }
  return formatValue(value);
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

function comparableCourtListenerClusters(retrieval: RetrievalPayload | null) {
  if (!retrieval || !hasCourtListenerCandidate(retrieval)) {
    return [];
  }
  return courtListenerMatches(retrieval);
}

function hasCourtListenerCandidate(retrieval: RetrievalPayload) {
  return (retrieval.status === "found" || citationStatusFromRetrieval(retrieval) === "ambiguous") &&
    courtListenerMatches(retrieval).length > 0;
}

function courtListenerMatches(retrieval: RetrievalPayload): CourtListenerCitationRecord[] {
  const candidates = ambiguousRetrievalCandidates(retrieval);
  if (candidates.length) {
    return candidates.map((candidate) => candidate.record);
  }
  return retrieval.candidate ? [retrieval.candidate.record] : [];
}

function ambiguousRetrievalCandidates(
  retrieval: RetrievalPayload | null
): RetrievedCandidatePayload[] {
  return retrieval?.status === "ambiguous" && Array.isArray(retrieval.candidates)
    ? retrieval.candidates
    : [];
}

function courtResolutionForCandidate(
  retrieval: RetrievalPayload | null,
  candidateIndex: number
): CourtResolutionTracePayload | null {
  if (!retrieval) {
    return null;
  }
  const candidates = ambiguousRetrievalCandidates(retrieval);
  if (candidates.length) {
    const safeIndex = Math.min(Math.max(candidateIndex, 0), candidates.length - 1);
    return candidates[safeIndex]?.court_resolution ?? null;
  }
  return retrieval.candidate?.court_resolution ?? null;
}

function candidateIdForIndex(
  retrieval: RetrievalPayload | null,
  candidateIndex: number
): string | null {
  if (!retrieval) {
    return null;
  }
  const candidates = ambiguousRetrievalCandidates(retrieval);
  if (candidates.length) {
    const safeIndex = Math.min(Math.max(candidateIndex, 0), candidates.length - 1);
    return candidates[safeIndex]?.candidate_id ?? null;
  }
  return retrieval.candidate?.candidate_id ?? null;
}

function citationStatus(citation: ReviewCitation): CitationStatus {
  return citation.retrieval ? citationStatusFromRetrieval(citation.retrieval) : "not_checked";
}

// A citation had a re-extraction when a grounded re-extraction produced an
// additional extracted attempt for it (the source of the case-name re-extraction switcher).
function citationHasReextraction(
  citation: ReviewCitation,
  _assessment: ReviewAssessment | null
): boolean {
  const result = completedAssessmentResult(citation.assessment);
  return Boolean(result && "reextracted_case_name" in result.case_name.followup);
}

// The effective (event-level) assessment for a citation: the re-extraction
// reassessment when one exists, otherwise the original first-pass assessment.
function effectiveAssessment(
  citation: ReviewCitation,
  _assessment: ReviewAssessment | null
): AssessmentPayload | null {
  const primary = completedAssessmentResult(citation.assessment);
  if (!primary) {
    return null;
  }
  const followup = primary.case_name.followup;
  return followup.status === "reassessed"
    ? { ...primary, case_name: { ...primary.case_name, initial: followup.result } }
    : primary;
}

function completedAssessmentResult(
  assessment: CitationAssessmentPayload | null
): AssessmentPayload | null {
  return assessment?.status === "assessed" ? assessment.result : null;
}

function activeAssessmentResult(
  assessment: CitationAssessmentPayload | null,
  candidateIndex: number
): AssessmentPayload | null {
  if (!assessment) {
    return null;
  }
  if (assessment.status === "assessed") {
    return assessment.result;
  }
  if (assessment.status === "ambiguous" && !assessment.gated) {
    const candidate = assessment.candidates[candidateIndex]?.result;
    return candidate ?? null;
  }
  return null;
}

function assessmentStatusFromPayload(assessment: AssessmentPayload | null): AssessmentStatus | null {
  if (!assessment) {
    return null;
  }
  const normalized = assessment.case_name.initial.status.toLowerCase().replaceAll("-", "_");
  if (normalized === "exact_match") {
    return "exact_match";
  }
  if (normalized === "semantic_match") {
    return "semantic_match";
  }
  if (normalized === "not_semantic_match") {
    return "not_semantic_match";
  }
  if (normalized === "irregular_form") {
    return "irregular_form";
  }
  if (normalized === "different_case") {
    return "different_case";
  }
  if (normalized === "unassessable") {
    return "unassessable";
  }
  return null;
}

function isAssessmentStatusFilter(filter: CitationFilter): filter is AssessmentStatus {
  return (
    filter === "exact_match" ||
    filter === "semantic_match" ||
    filter === "not_semantic_match" ||
    filter === "irregular_form" ||
    filter === "different_case" ||
    filter === "unassessable"
  );
}

function isRetrievalStatusFilter(
  filter: CitationFilter
): filter is Exclude<RetrievalFilter, "all"> {
  return (
    filter === "found" ||
    filter === "ambiguous" ||
    filter === "not_found" ||
    filter === "throttled"
  );
}

function isFullCaseCitation(citation: ReviewCitation) {
  return citation.kind.toLowerCase() === "fullcasecitation";
}

function isAssessmentCandidateCitation(citation: ReviewCitation) {
  const status = citation.retrieval?.status.toLowerCase().replaceAll("-", "_");
  return status === "found" || status === "ambiguous";
}

function citationStatusFromRetrieval(retrieval: RetrievalPayload): CitationStatus {
  const normalized = retrieval.status.toLowerCase().replaceAll("-", "_");
  if (normalized === "found") {
    return "found";
  }
  if (normalized === "ambiguous") {
    return "ambiguous";
  }
  if (normalized === "not_found") {
    return "not_found";
  }
  if (normalized === "throttled") {
    return "throttled";
  }
  return "not_checked";
}

function citationStatusCounts(citations: ReviewCitation[]) {
  return citations.reduce(
    (counts, citation) => {
      const status = citationStatus(citation);
      counts[status] += 1;
      return counts;
    },
    { found: 0, ambiguous: 0, not_found: 0, throttled: 0, not_checked: 0 } satisfies Record<
      CitationStatus,
      number
    >
  );
}

function mergeCitationRetrieval(result: ReviewResult, retrieval: RetrievalPayload): ReviewResult {
  return withRetrievalStats({
    ...result,
    citations: result.citations.map((citation) =>
      citation.id === retrieval.citation_id
        ? { ...citation, retrieval, assessment: null }
        : citation
    )
  });
}

function withRetrievalStats(result: ReviewResult): ReviewResult {
  const retrievals = result.citations
    .map((citation) => citation.retrieval)
    .filter(isRetrievalPayload);
  const retrieved = retrievals.filter((retrieval) => retrieval.status !== "skipped").length;
  const found = retrievals.filter((retrieval) => retrieval.status === "found").length;

  return {
    ...result,
    retrieval: {
      retrievals,
      counts: {
        total: retrievals.length,
        found
      }
    },
    stats: {
      ...result.stats,
      retrieved,
      found
    }
  };
}

function isRetrievalPayload(value: RetrievalPayload | null): value is RetrievalPayload {
  return value !== null;
}

function assessmentStatusCounts(
  citations: ReviewCitation[],
  assessment: ReviewAssessment | null
) {
  return citations.reduce(
    (counts, citation) => {
      const status = assessmentStatusFromPayload(effectiveAssessment(citation, assessment));
      if (status) {
        counts[status] += 1;
      }
      return counts;
    },
    {
      exact_match: 0,
      semantic_match: 0,
      not_semantic_match: 0,
      irregular_form: 0,
      different_case: 0,
      unassessable: 0
    } satisfies Record<AssessmentStatus, number>
  );
}

function filterLabel(
  filter: FilterOption,
  counts: Record<CitationStatus, number>,
  assessmentCounts: Record<AssessmentStatus, number>,
  fullCaseTotal: number,
  allCitationTotal: number,
  reextractionTotal: number
) {
  return `${filter.label} (${filterCount(
    filter.value,
    counts,
    assessmentCounts,
    fullCaseTotal,
    allCitationTotal,
    reextractionTotal
  )})`;
}

function filterCount(
  filter: CitationFilter,
  counts: Record<CitationStatus, number>,
  assessmentCounts: Record<AssessmentStatus, number>,
  fullCaseTotal: number,
  allCitationTotal: number,
  reextractionTotal: number
) {
  if (filter === "all") {
    return fullCaseTotal;
  }
  if (filter === "all_citations") {
    return allCitationTotal;
  }
  if (filter === "reextraction") {
    return reextractionTotal;
  }
  if (isAssessmentStatusFilter(filter)) {
    return assessmentCounts[filter];
  }
  return counts[filter];
}

function stageFilterOptions(
  stage: WorkflowStage,
  counts: Record<CitationStatus, number>,
  assessmentCounts: Record<AssessmentStatus, number>,
  fullCaseTotal: number,
  allCitationTotal: number,
  reextractionTotal: number
): FilterOption[] {
  if (stage === "retrieved") {
    return retrievalFilters;
  }

  const filters = stage === "assessed" ? assessmentFilters : extractionFilters;
  return filters.filter(
    (filter) =>
      filter.value === "all" ||
      filter.value === "all_citations" ||
      filterCount(
        filter.value,
        counts,
        assessmentCounts,
        fullCaseTotal,
        allCitationTotal,
        reextractionTotal
      ) > 0
  );
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

async function retrieveReviewCitation(citation: ReviewCitation): Promise<RetrievalPayload> {
  const response = await fetch("/api/e2e/retrieve-review-citation", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ citation })
  });
  return parseJsonResponse<RetrievalPayload>(response);
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

async function fetchBookmarkStatuses(
  citations: ReviewCitation[],
  documentText: string
) {
  const response = await fetch("/api/bookmark", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      action: "status",
      citations: citations.map((citation) => ({
        citation_id: citation.id,
        ...bookmarkIdentityForCitation(citation, documentText)
      }))
    })
  });
  const payload = await parseJsonResponse<BookmarkStatusResponse>(response);
  return payload.statuses;
}

async function addBookmarkCitation(payload: {
  citation: { matched_text: string; context: string };
  provenance: {
    source_path: string | null;
    source_format: string;
    span: { start: number; end: number };
  };
  comment: string | null;
}): Promise<BookmarkMutationResult> {
  const response = await fetch("/api/bookmark", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ action: "add", ...payload })
  });
  return parseJsonResponse<BookmarkMutationResult>(response);
}

async function updateBookmarkCommentAction(payload: {
  citation: { matched_text: string; context: string };
  comment: string | null;
}): Promise<BookmarkCommentUpdateResult> {
  const response = await fetch("/api/bookmark", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ action: "update_comment", ...payload })
  });
  return parseJsonResponse<BookmarkCommentUpdateResult>(response);
}

function isBookmarkFixturePath(sourcePath: string | null) {
  if (!sourcePath) {
    return false;
  }
  const filename = sourcePath.replaceAll("\\", "/").split("/").pop()?.toLowerCase();
  return filename === "bookmarked.txt";
}

function citationContextWindow(documentText: string, citation: ReviewCitation) {
  const contextChars = 200;
  const start = Math.max(0, citation.start - contextChars);
  const end = Math.min(documentText.length, citation.end + contextChars);
  return documentText.slice(start, end);
}

function bookmarkIdentityForCitation(citation: ReviewCitation, documentText: string) {
  return {
    matched_text: citation.matched_text,
    context: citationContextWindow(documentText, citation)
  };
}

function isCitationBookmarked(
  citation: ReviewCitation,
  documentText: string,
  entries: Map<string, BookmarkStatusEntry>
) {
  return entries.get(citation.id)?.bookmarked ?? false;
}

function citationBookmarkEntry(
  citation: ReviewCitation,
  entries: Map<string, BookmarkStatusEntry>
) {
  return entries.get(citation.id);
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

function BookmarkModal({
  matchedText,
  context,
  existingComment,
  saving,
  onSave,
  onCancel
}: {
  matchedText: string;
  context: string;
  existingComment: string | null;
  saving: boolean;
  onSave: (comment: string | null) => void;
  onCancel: () => void;
}) {
  const [comment, setComment] = useState<string>(existingComment ?? "");
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    setComment(existingComment ?? "");
  }, [existingComment]);

  useEffect(() => {
    inputRef.current?.focus();
    inputRef.current?.select();
  }, []);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (saving) {
      return;
    }
    onSave(comment.trim() ? comment : null);
  }

  return (
    <div className="bookmark-modal-backdrop" role="presentation">
      <div
        className="bookmark-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="bookmark-modal-title"
      >
        <header className="bookmark-modal-header">
          <h2 id="bookmark-modal-title">
            {existingComment === null ? "Bookmark citation" : "Edit bookmark comment"}
          </h2>
          <button
            className="icon-action"
            type="button"
            onClick={onCancel}
            disabled={saving}
            aria-label="Close bookmark modal"
          >
            ×
          </button>
        </header>
        <form onSubmit={handleSubmit} className="bookmark-modal-form">
          <p className="bookmark-modal-citation">{matchedText}</p>
          <details className="bookmark-modal-context">
            <summary>Context window</summary>
            <pre>{context}</pre>
          </details>
          <label className="bookmark-modal-label" htmlFor="bookmark-modal-comment">
            Comment (optional)
          </label>
          <textarea
            ref={inputRef}
            id="bookmark-modal-comment"
            className="bookmark-modal-textarea"
            value={comment}
            onChange={(event) => setComment(event.target.value)}
            placeholder="Why is this citation worth bookmarking?"
            rows={4}
            disabled={saving}
          />
          <p className="bookmark-modal-hint">
            The bookmark is identified by the citation text and surrounding
            context. The same text in a different document is recognized as
            the same bookmark.
          </p>
          <div className="bookmark-modal-actions">
            <button
              className="secondary-action compact-action"
              type="button"
              onClick={onCancel}
              disabled={saving}
            >
              Cancel
            </button>
            <button
              className="primary-action compact-action"
              type="submit"
              disabled={saving}
            >
              {saving ? <Loader2 className="spin" size={14} /> : null}
              <span>{existingComment === null ? "Save bookmark" : "Update comment"}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
