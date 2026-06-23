import { mkdir, appendFile } from "node:fs/promises";
import path from "node:path";

type BookmarkPayload = {
  context?: unknown;
};

export async function POST(request: Request) {
  const payload = (await request.json()) as BookmarkPayload;
  const context = typeof payload.context === "string" ? payload.context.trim() : "";
  if (!context) {
    return Response.json({ detail: "Bookmark context is required." }, { status: 400 });
  }

  const repoRoot = process.cwd().endsWith(`${path.sep}frontend`)
    ? path.dirname(process.cwd())
    : process.cwd();
  const outDir = path.join(repoRoot, "local", "bookmarked");
  const outPath = path.join(outDir, "bookmarked.txt");
  const block = ["========", context, ""].join("\n");

  await mkdir(outDir, { recursive: true });
  await appendFile(outPath, block, "utf-8");

  return Response.json({ path: "local/bookmarked/bookmarked.txt" });
}
