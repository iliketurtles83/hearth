from fastapi import APIRouter
from fastapi.responses import Response


def create_health_router(services) -> APIRouter:
    router = APIRouter()

    app = services.app

    @router.get("/health")
    async def health():
        graph_ready = getattr(app.state, "assistant_graph", None) is not None
        embed_router = getattr(app.state, "embedding_router", None)
        return {
            "status": "ok" if graph_ready else "starting",
            "embedding_router": embed_router is not None,
        }

    @router.head("/health")
    async def health_head():
        # Respond to HEAD probes (no body).
        return Response(status_code=200)

    return router
