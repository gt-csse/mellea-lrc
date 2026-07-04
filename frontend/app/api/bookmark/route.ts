import {
  addBookmark,
  bookmarkStatuses,
  isBookmarkAddRequest,
  isBookmarkStatusRequest
} from "./store";

export const runtime = "nodejs";

export async function POST(request: Request) {
  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return Response.json({ detail: "Bookmark request must be valid JSON." }, { status: 400 });
  }

  try {
    if (isBookmarkStatusRequest(payload)) {
      return Response.json(await bookmarkStatuses(payload.citations));
    }
    if (isBookmarkAddRequest(payload)) {
      return Response.json(await addBookmark(payload));
    }
    return Response.json({ detail: "Unsupported bookmark request shape." }, { status: 400 });
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Bookmark operation failed.";
    return Response.json({ detail }, { status: 500 });
  }
}
