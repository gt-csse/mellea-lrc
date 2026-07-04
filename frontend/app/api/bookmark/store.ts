import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";

const schemaVersion = 1;
const separator = "========";

type CitationField = string | number | boolean;

export type BookmarkCitationInput = {
  citation_id: string;
  matched_text: string;
  kind: string;
  fields: Record<string, CitationField>;
};

type BookmarkProvenanceInput = {
  source_path: string | null;
  source_format: string;
  citation_id: string;
  span: { start: number; end: number };
  context: string;
};

export type BookmarkStatusRequest = {
  action: "status";
  citations: BookmarkCitationInput[];
};

export type BookmarkAddRequest = {
  action: "add";
  citation: BookmarkCitationInput;
  provenance: BookmarkProvenanceInput;
};

type StoredCitation = Omit<BookmarkCitationInput, "citation_id">;

type StoredProvenance = BookmarkProvenanceInput & {
  provenance_id: string;
  bookmarked_at: string | null;
};

type BookmarkRecord = {
  bookmark_id: string;
  citation: StoredCitation | null;
  test_context: string;
  provenances: StoredProvenance[];
};

type BookmarkStore = {
  schema_version: 1;
  bookmarks: BookmarkRecord[];
};

let operationQueue: Promise<void> = Promise.resolve();

export function isBookmarkStatusRequest(value: unknown): value is BookmarkStatusRequest {
  if (!isRecord(value) || value.action !== "status" || !Array.isArray(value.citations)) {
    return false;
  }
  return value.citations.every(isCitationInput);
}

export function isBookmarkAddRequest(value: unknown): value is BookmarkAddRequest {
  if (!isRecord(value) || value.action !== "add") {
    return false;
  }
  return isCitationInput(value.citation) && isProvenanceInput(value.provenance);
}

export async function bookmarkStatuses(citations: BookmarkCitationInput[]) {
  return withStoreLock(async () => {
    const paths = bookmarkPaths();
    const store = await loadOrMigrateStore(paths);
    const available = new Set(store.bookmarks.map((bookmark) => bookmark.bookmark_id));
    return {
      statuses: Object.fromEntries(
        citations.map((citation) => [citation.citation_id, available.has(bookmarkId(citation))])
      ),
      paths: publicPaths(paths)
    };
  });
}

export async function addBookmark(request: BookmarkAddRequest) {
  return withStoreLock(async () => {
    const paths = bookmarkPaths();
    const store = await loadOrMigrateStore(paths);
    const id = bookmarkId(request.citation);
    const provenance = storedProvenance(request.provenance);
    const existing = store.bookmarks.find((bookmark) => bookmark.bookmark_id === id);
    let result: "created" | "provenance_added" | "already_bookmarked";

    if (!existing) {
      store.bookmarks.push({
        bookmark_id: id,
        citation: {
          matched_text: request.citation.matched_text.trim(),
          kind: request.citation.kind,
          fields: sortedFields(request.citation.fields)
        },
        test_context: request.provenance.context.trim(),
        provenances: [provenance]
      });
      result = "created";
    } else if (
      existing.provenances.some((item) => item.provenance_id === provenance.provenance_id)
    ) {
      result = "already_bookmarked";
    } else {
      existing.provenances.push(provenance);
      result = "provenance_added";
    }

    if (result !== "already_bookmarked") {
      await writeStoreAndFixture(paths, store);
    }
    return {
      result,
      bookmark_id: id,
      provenance_count: existing?.provenances.length ?? 1,
      paths: publicPaths(paths)
    };
  });
}

function bookmarkId(citation: BookmarkCitationInput) {
  const fields = sortedFields(citation.fields);
  const identity = {
    kind: citation.kind,
    fields,
    ...(Object.keys(fields).length ? {} : { matched_text: citation.matched_text.trim() })
  };
  return `citation:${digest(stableJson(identity))}`;
}

function storedProvenance(input: BookmarkProvenanceInput): StoredProvenance {
  const identity = {
    source_path: input.source_path,
    source_format: input.source_format,
    span: input.span,
    context: input.context.trim()
  };
  return {
    ...input,
    context: input.context.trim(),
    provenance_id: `provenance:${digest(stableJson(identity))}`,
    bookmarked_at: new Date().toISOString()
  };
}

function bookmarkPaths() {
  const repoRoot = process.cwd().endsWith(`${path.sep}frontend`)
    ? path.dirname(process.cwd())
    : process.cwd();
  const directory = process.env.MELLEA_LRC_BOOKMARK_DIR
    ? path.resolve(process.env.MELLEA_LRC_BOOKMARK_DIR)
    : path.join(repoRoot, "local", "bookmarked");
  return {
    directory,
    json: path.join(directory, "bookmarks.json"),
    text: path.join(directory, "bookmarked.txt"),
    repoRoot
  };
}

async function loadOrMigrateStore(paths: ReturnType<typeof bookmarkPaths>): Promise<BookmarkStore> {
  await mkdir(paths.directory, { recursive: true });
  try {
    const parsed = JSON.parse(await readFile(paths.json, "utf-8")) as unknown;
    if (!isBookmarkStore(parsed)) {
      throw new Error("Bookmark JSON does not match schema version 1.");
    }
    return parsed;
  } catch (error) {
    if (!isMissingFile(error)) {
      throw error;
    }
  }

  const store = await migrateLegacyText(paths.text);
  await atomicWrite(paths.json, `${JSON.stringify(store, null, 2)}\n`);
  try {
    await readFile(paths.text, "utf-8");
  } catch (error) {
    if (!isMissingFile(error)) {
      throw error;
    }
    await atomicWrite(paths.text, "");
  }
  return store;
}

async function migrateLegacyText(textPath: string): Promise<BookmarkStore> {
  let text = "";
  try {
    text = await readFile(textPath, "utf-8");
  } catch (error) {
    if (!isMissingFile(error)) {
      throw error;
    }
  }
  const contexts = text
    .split(/^========\s*$/m)
    .map((context) => context.trim())
    .filter(Boolean);
  return {
    schema_version: schemaVersion,
    bookmarks: contexts.map((context) => {
      const id = digest(context);
      return {
        bookmark_id: `legacy:${id}`,
        citation: null,
        test_context: context,
        provenances: [
          {
            provenance_id: `legacy-provenance:${id}`,
            source_path: "local/bookmarked/bookmarked.txt",
            source_format: "text",
            citation_id: "legacy",
            span: { start: 0, end: context.length },
            context,
            bookmarked_at: null
          }
        ]
      };
    })
  };
}

async function writeStoreAndFixture(
  paths: ReturnType<typeof bookmarkPaths>,
  store: BookmarkStore
) {
  const json = `${JSON.stringify(store, null, 2)}\n`;
  const text = store.bookmarks.length
    ? `${store.bookmarks.map((bookmark) => `${separator}\n${bookmark.test_context}`).join("\n\n")}\n`
    : "";
  await atomicWrite(paths.json, json);
  await atomicWrite(paths.text, text);
}

async function atomicWrite(destination: string, contents: string) {
  const temporary = `${destination}.${process.pid}.tmp`;
  await writeFile(temporary, contents, "utf-8");
  await rename(temporary, destination);
}

function publicPaths(paths: ReturnType<typeof bookmarkPaths>) {
  const relativeJson = path.relative(paths.repoRoot, paths.json);
  const relativeText = path.relative(paths.repoRoot, paths.text);
  return {
    json: relativeJson.startsWith("..") ? paths.json : relativeJson,
    text: relativeText.startsWith("..") ? paths.text : relativeText
  };
}

function sortedFields(fields: Record<string, CitationField>) {
  return Object.fromEntries(Object.entries(fields).sort(([left], [right]) => left.localeCompare(right)));
}

function stableJson(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(stableJson).join(",")}]`;
  }
  if (isRecord(value)) {
    return `{${Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => `${JSON.stringify(key)}:${stableJson(item)}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

function digest(value: string) {
  return createHash("sha256").update(value).digest("hex");
}

function isCitationInput(value: unknown): value is BookmarkCitationInput {
  return (
    isRecord(value) &&
    typeof value.citation_id === "string" &&
    Boolean(value.citation_id) &&
    typeof value.matched_text === "string" &&
    Boolean(value.matched_text.trim()) &&
    typeof value.kind === "string" &&
    Boolean(value.kind) &&
    isCitationFields(value.fields)
  );
}

function isProvenanceInput(value: unknown): value is BookmarkProvenanceInput {
  return (
    isRecord(value) &&
    (typeof value.source_path === "string" || value.source_path === null) &&
    typeof value.source_format === "string" &&
    typeof value.citation_id === "string" &&
    isSpan(value.span) &&
    typeof value.context === "string" &&
    Boolean(value.context.trim())
  );
}

function isBookmarkStore(value: unknown): value is BookmarkStore {
  return (
    isRecord(value) &&
    value.schema_version === schemaVersion &&
    Array.isArray(value.bookmarks) &&
    value.bookmarks.every(isBookmarkRecord)
  );
}

function isBookmarkRecord(value: unknown): value is BookmarkRecord {
  return (
    isRecord(value) &&
    typeof value.bookmark_id === "string" &&
    (value.citation === null || isStoredCitation(value.citation)) &&
    typeof value.test_context === "string" &&
    Array.isArray(value.provenances) &&
    value.provenances.every(isStoredProvenance)
  );
}

function isStoredCitation(value: unknown): value is StoredCitation {
  return (
    isRecord(value) &&
    typeof value.matched_text === "string" &&
    typeof value.kind === "string" &&
    isCitationFields(value.fields)
  );
}

function isStoredProvenance(value: unknown): value is StoredProvenance {
  if (!isRecord(value)) {
    return false;
  }
  const record = value;
  if (!isProvenanceInput(record)) {
    return false;
  }
  return (
    "provenance_id" in record &&
    typeof record.provenance_id === "string" &&
    "bookmarked_at" in record &&
    (typeof record.bookmarked_at === "string" || record.bookmarked_at === null)
  );
}

function isCitationFields(value: unknown): value is Record<string, CitationField> {
  return (
    isRecord(value) &&
    Object.values(value).every(
      (field) => typeof field === "string" || typeof field === "number" || typeof field === "boolean"
    )
  );
}

function isSpan(value: unknown): value is { start: number; end: number } {
  return (
    isRecord(value) &&
    Number.isInteger(value.start) &&
    Number.isInteger(value.end) &&
    Number(value.start) >= 0 &&
    Number(value.end) >= Number(value.start)
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isMissingFile(error: unknown): error is NodeJS.ErrnoException {
  return isRecord(error) && error.code === "ENOENT";
}

async function withStoreLock<T>(operation: () => Promise<T>): Promise<T> {
  const previous = operationQueue;
  let release: () => void = () => undefined;
  operationQueue = new Promise<void>((resolve) => {
    release = resolve;
  });
  await previous;
  try {
    return await operation();
  } finally {
    release();
  }
}
