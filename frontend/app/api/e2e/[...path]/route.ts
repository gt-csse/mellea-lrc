type RouteContext = {
  params: Promise<{
    path?: string[];
  }>;
};

const forbiddenRequestHeaders = new Set(["host", "content-length"]);
const forbiddenResponseHeaders = new Set(["content-encoding", "content-length", "transfer-encoding"]);

export async function POST(request: Request, context: RouteContext) {
  return proxyToBackend(request, context);
}

export async function GET(request: Request, context: RouteContext) {
  return proxyToBackend(request, context);
}

async function proxyToBackend(request: Request, context: RouteContext) {
  const backendUrl = process.env.MELLEA_E2E_BACKEND_URL?.replace(/\/$/, "");
  if (!backendUrl) {
    return Response.json(
      { detail: "MELLEA_E2E_BACKEND_URL is not configured for the Next.js frontend." },
      { status: 503 }
    );
  }

  const { path = [] } = await context.params;
  const search = new URL(request.url).search;
  const upstream = await fetch(`${backendUrl}/api/${path.join("/")}${search}`, {
    method: request.method,
    headers: forwardedRequestHeaders(request.headers),
    body: request.method === "GET" ? undefined : await request.arrayBuffer(),
    cache: "no-store"
  });

  return new Response(upstream.body, {
    status: upstream.status,
    headers: forwardedResponseHeaders(upstream.headers)
  });
}

function forwardedRequestHeaders(headers: Headers) {
  const forwarded = new Headers(headers);
  forbiddenRequestHeaders.forEach((header) => forwarded.delete(header));
  return forwarded;
}

function forwardedResponseHeaders(headers: Headers) {
  const forwarded = new Headers(headers);
  forbiddenResponseHeaders.forEach((header) => forwarded.delete(header));
  return forwarded;
}
