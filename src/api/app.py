from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import router


app = FastAPI(

    title="Multimodal Legal Risk Auditor",

    version="1.0.0",

)


app.include_router(

    router,

)