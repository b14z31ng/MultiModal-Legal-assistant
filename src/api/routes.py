from __future__ import annotations

import tempfile
import shutil

from fastapi import APIRouter
from fastapi import UploadFile
from fastapi import File
from fastapi import Depends

from src.api.dependencies import (
    get_predictor,
    get_processor,
)

router = APIRouter()


@router.get("/health")
def health():

    return {

        "status": "ok",

    }


@router.post("/predict")
async def predict(

    file: UploadFile = File(...),

    predictor=Depends(get_predictor),

    processor=Depends(get_processor),

):

    suffix = file.filename.split(".")[-1]

    with tempfile.NamedTemporaryFile(

        delete=False,

        suffix=f".{suffix}",

    ) as tmp:

        shutil.copyfileobj(

            file.file,

            tmp,

        )

        path = tmp.name

    document = processor.process(

        path,

    )

    result = predictor.predict(

        document,

    )

    return result