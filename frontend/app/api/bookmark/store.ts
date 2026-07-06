import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";

const schemaVersion = 2;
const separator = "========";

const contextChars = 200;

type BookmarkProvenance = {
  source_path: string | null;
  source_format: string;
  span: { start: number; end: number };
};

export type BookmarkCitationInput = {
  citation_id?: string;
  matched_text: string;
  context: string;
};

export type BookmarkStatusRequest = {
  action: "status";
  citations: BookmarkCitationInput[];
};

export type BookmarkAddRequest = {
  action: "add";
  citation: BookmarkCitationInput;
  provenance: BookmarkProvenance;
  comment: string | null;
};

export type BookmarkUpdateCommentRequest = {
  action: "update_comment";
  citation: BookmarkCitationInput;
  comment: string | null;
};

type StoredCitation = BookmarkCitationInput;

type StoredProvenance = BookmarkProvenance & {
  provenance_id: string;
  seen_at: string;
};

type BookmarkRecord = {
  bookmark_id: string;
  citation: StoredCitation;
  comment: string | null;
  provenances: StoredProvenance[];
  created_at: string;
  updated_at: string;
};

type BookmarkStore = {
  schema_version: 2;
  bookmarks: BookmarkRecord[];
};

type BookmarkStatusEntry = {
  bookmarked: boolean;
  comment: string | null;
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
  return isCitationInput(value.citation) && isProvenance(value.provenance) && isComment(value.comment);
}

export function isBookmarkUpdateCommentRequest(
  value: unknown
): value is BookmarkUpdateCommentRequest {
  if (!isRecord(value) || value.action !== "update_comment") {
    return false;
  }
  return isCitationInput(value.citation) && isComment(value.comment);
}

export async function bookmarkStatuses(citations: BookmarkCitationInput[]) {
  return withStoreLock(async () => {
    const paths = bookmarkPaths();
    const store = await loadOrMigrateStore(paths);
    const lookup = new Map(
      store.bookmarks.map((bookmark) => [bookmark.bookmark_id, bookmark])
    );
    return {
      statuses: Object.fromEntries(
        citations.map((citation) => {
          const id = bookmarkId(citation);
          const existing = lookup.get(id);
          const entry: BookmarkStatusEntry = {
            bookmarked: existing !== undefined,
            comment: existing?.comment ?? null
          };
          return [citation.citation_id ?? id, entry];
        })
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
    const now = new Date().toISOString();
    const provenance = storedProvenance(request.provenance, now);
    const existing = store.bookmarks.find((bookmark) => bookmark.bookmark_id === id);
    let result: "created" | "provenance_added" | "already_bookmarked";
    let commentChanged = false;

    if (!existing) {
      store.bookmarks.push({
        bookmark_id: id,
        citation: {
          matched_text: request.citation.matched_text.trim(),
          context: request.citation.context.trim()
        },
        comment: request.comment?.trim() ? request.comment.trim() : null,
        provenances: [provenance],
        created_at: now,
        updated_at: now
      });
      result = "created";
      commentChanged = request.comment !== null;
    } else if (
      existing.provenances.some((item) => item.provenance_id === provenance.provenance_id)
    ) {
      result = "already_bookmarked";
    } else {
      existing.provenances.push(provenance);
      result = "provenance_added";
    }

    if (result !== "already_bookmarked" || commentChanged) {
      await writeStoreAndFixture(paths, store);
    }
    return {
      result,
      bookmark_id: id,
      comment: store.bookmarks.find((b) => b.bookmark_id === id)?.comment ?? null,
      provenance_count: existing?.provenances.length ?? 1,
      paths: publicPaths(paths)
    };
  });
}

export async function updateBookmarkComment(
  request: BookmarkUpdateCommentRequest
) {
  return withStoreLock(async () => {
    const paths = bookmarkPaths();
    const store = await loadOrMigrateStore(paths);
    const id = bookmarkId(request.citation);
    const bookmark = store.bookmarks.find((item) => item.bookmark_id === id);
    if (!bookmark) {
      return {
        result: "not_found" as const,
        bookmark_id: id,
        comment: null,
        paths: publicPaths(paths)
      };
    }
    const trimmed = request.comment?.trim() ? request.comment.trim() : null;
    if (bookmark.comment === trimmed) {
      return {
        result: "unchanged" as const,
        bookmark_id: id,
        comment: bookmark.comment,
        paths: publicPaths(paths)
      };
    }
    bookmark.comment = trimmed;
    bookmark.updated_at = new Date().toISOString();
    await writeStoreAndFixture(paths, store);
    return {
      result: "updated" as const,
      bookmark_id: id,
      comment: bookmark.comment,
      paths: publicPaths(paths)
    };
  });
}

function bookmarkId(citation: BookmarkCitationInput) {
  return `citation:${digest(normalizeBookmarkText(citation))}`;
}

function normalizeBookmarkText(citation: BookmarkCitationInput) {
  return stableJson({
    matched_text: citation.matched_text.trim().replace(/\s+/g, " "),
    context: citation.context.trim().replace(/\s+/g, " ")
  });
}

function storedProvenance(input: BookmarkProvenance, seenAt: string): StoredProvenance {
  const identity = {
    source_path: input.source_path,
    source_format: input.source_format,
    span: input.span
  };
  return {
    ...input,
    provenance_id: `provenance:${digest(stableJson(identity))}`,
    seen_at: seenAt
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
      throw new Error(`Bookmark JSON does not match schema version ${schemaVersion}.`);
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
      const now = new Date().toISOString();
      return {
        bookmark_id: `legacy:${id}`,
        citation: {
          matched_text: context,
          context
        },
        comment: null,
        provenances: [
          {
            provenance_id: `legacy-provenance:${id}`,
            source_path: "local/bookmarked/bookmarked.txt",
            source_format: "text",
            span: { start: 0, end: context.length },
            seen_at: now
          }
        ],
        created_at: now,
        updated_at: now
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
    ? `${store.bookmarks
        .map(
          (bookmark) =>
            `${separator}\n${formatBookmarkText(bookmark)}`
        )
        .join("\n\n")}\n`
    : "";
  await atomicWrite(paths.json, json);
  await atomicWrite(paths.text, text);
}

function formatBookmarkText(bookmark: BookmarkRecord) {
  const lines = [bookmark.citation.matched_text];
  if (bookmark.comment) {
    lines.push("", `> ${bookmark.comment}`);
  }
  lines.push("", `Context: ${bookmark.citation.context}`);
  return lines.join("\n");
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
    typeof value.matched_text === "string" &&
    Boolean(value.matched_text.trim()) &&
    typeof value.context === "string" &&
    Boolean(value.context.trim())
  );
}

function isProvenance(value: unknown): value is BookmarkProvenance {
  return (
    isRecord(value) &&
    (typeof value.source_path === "string" || value.source_path === null) &&
    typeof value.source_format === "string" &&
    isSpan(value.span)
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

function isComment(value: unknown): value is string | null {
  return value === null || typeof value === "string";
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
    isStoredCitation(value.citation) &&
    (value.comment === null || typeof value.comment === "string") &&
    typeof value.created_at === "string" &&
    typeof value.updated_at === "string" &&
    Array.isArray(value.provenances) &&
    value.provenances.every(isStoredProvenance)
  );
}

function isStoredCitation(value: unknown): value is StoredCitation {
  return (
    isRecord(value) &&
    typeof value.matched_text === "string" &&
    typeof value.context === "string"
  );
}

function isStoredProvenance(value: unknown): value is StoredProvenance {
  if (!isRecord(value)) {
    return false;
  }
  const record = value;
  if (!isProvenance(record)) {
    return false;
  }
  return (
    "provenance_id" in record &&
    typeof record.provenance_id === "string" &&
    "seen_at" in record &&
    typeof record.seen_at === "string"
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

export const __test = { contextChars, normalizeBookmarkText, bookmarkId };
