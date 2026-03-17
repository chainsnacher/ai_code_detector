from __future__ import annotations

import io
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.survey.bot_detector import SurveyBotDetector


app = FastAPI(title="Survey Integrity API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SurveyCheckRequest(BaseModel):
    response: str = Field(..., description="Free-text survey response")
    risk_threshold: float = Field(0.65, ge=0.0, le=1.0)


class SurveyCheckResponse(BaseModel):
    risk_score: float
    flagged: bool
    label: str
    reasons: list[str]
    metrics: Dict[str, Any]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/survey/check", response_model=SurveyCheckResponse)
def check_survey(req: SurveyCheckRequest) -> SurveyCheckResponse:
    detector = SurveyBotDetector()
    out = detector.score_text(req.response, risk_threshold=req.risk_threshold)
    return SurveyCheckResponse(**out)


@app.post("/survey/clean-file")
async def clean_file(
    file: UploadFile = File(...),
    text_col: str = "response",
    risk_threshold: float = 0.65,
) -> Dict[str, Any]:
    filename = (file.filename or "").lower()
    if not (filename.endswith(".csv") or filename.endswith(".xlsx") or filename.endswith(".xls")):
        raise HTTPException(status_code=400, detail="Upload must be .csv, .xlsx, or .xls")

    content = await file.read()
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    if text_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Missing column: {text_col}")

    detector = SurveyBotDetector()
    scored, report = detector.analyze_dataframe(df, text_col=text_col, risk_threshold=float(risk_threshold))

    # Return JSON (client can download scored file from Streamlit or regenerate from df)
    # Keep payload modest: include summary + first 25 flagged samples.
    flagged_df = scored[scored["flagged"] == True].copy()
    sample = flagged_df.head(25).to_dict(orient="records")
    return {"report": report, "flagged_samples": sample}

