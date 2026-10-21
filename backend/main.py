import io, json, os
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from PIL import Image
from pydantic import BaseModel
from torchvision import models, transforms
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env", encoding="utf-8-sig")
MODEL_DIR = BASE.parent / "ml" / "models"

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
CONF_THRESHOLD = 0.70
RARE = ("Cavity", "Impacted Tooth")
MAX_BYTES = 5 * 1024 * 1024
MAX_XRAY_BYTES = 15 * 1024 * 1024
LOW_CONF = 0.40
DISCLAIMER = ("Decision support only. This is not a diagnosis. All findings "
              "must be confirmed by a qualified dentist.")

# ---- single-tooth classifier (v2) ----
CFG = json.load(open(MODEL_DIR / "model_config_v2.json"))
classes = CFG["classes"]
SIZE = CFG["size"]
BOOST = CFG["cavity_boost"]
CAV = classes.index("Cavity")

net = models.efficientnet_b0()
net.classifier[1] = nn.Linear(net.classifier[1].in_features, len(classes))
net.load_state_dict(torch.load(MODEL_DIR / "tooth_classifier_v2.pt", map_location="cpu"))
net.eval()

tf = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---- full X-ray detector ----
DET = YOLO(str(MODEL_DIR / "xray_detector.pt"))
DET_NAMES = DET.names

app = FastAPI(title="Dental AI Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class Findings(BaseModel):
    top_condition: str
    confidence: float
    probabilities: dict[str, float]
    needs_review: bool = False
    warnings: list[str] = []


class Box(BaseModel):
    x: float
    y: float
    w: float
    h: float


class Detection(BaseModel):
    id: int = 0
    label: str
    confidence: float
    box: Box


class XrayFindings(BaseModel):
    image_width: int = 0
    image_height: int = 0
    detections: list[Detection] = []
    warnings: list[str] = []


class Msg(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[Msg] = []
    findings: Findings | None = None
    xray: XrayFindings | None = None


def classify(img):
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        p = (torch.softmax(net(x), 1) + torch.softmax(net(torch.flip(x, [3])), 1)) / 2
    p = p[0].tolist()
    boosted = list(p)
    boosted[CAV] *= BOOST
    top_idx = max(range(len(classes)), key=lambda i: boosted[i])
    probs = {c: round(v, 4) for c, v in zip(classes, p)}
    return probs, classes[top_idx]


def findings_text(f):
    lines = ["AI findings for the uploaded tooth image:",
             "Top prediction: %s (confidence %.0f%%)" % (f.top_condition, f.confidence * 100),
             "All class probabilities:"]
    for name, p in sorted(f.probabilities.items(), key=lambda kv: -kv[1]):
        lines.append("  - %s: %.0f%%" % (name, p * 100))
    if f.needs_review:
        lines.append("The system flagged this result as NEEDS REVIEW.")
    for w in f.warnings:
        lines.append("Warning: " + w)
    return "\n".join(lines)


def xray_text(f):
    lines = ["AI findings for the uploaded dental X-ray (object detector that draws "
             "boxes around suspicious areas):"]
    if not f.detections:
        lines.append("No suspicious areas were detected above the confidence threshold. "
                     "This does not rule out disease.")
    for i, d in enumerate(f.detections, 1):
        lines.append("  #%d %s (confidence %.0f%%)" % (d.id or i, d.label, d.confidence * 100))
    for w in f.warnings:
        lines.append("Warning: " + w)
    return "\n".join(lines)


def call_llm(system, messages, max_tokens=2500):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise HTTPException(500, "GROQ_API_KEY is not set. Add it to backend/.env")
    msgs = [{"role": "system", "content": system}] + messages
    try:
        r = Groq(api_key=key).chat.completions.create(
            model=LLM_MODEL, messages=msgs, max_tokens=max_tokens, temperature=0.3)
    except Exception as e:
        raise HTTPException(502, "LLM error: %s" % e)
    return r.choices[0].message.content or ""


REPORT_SYSTEM = (
    "You are a dental clinical-decision-support assistant writing for a licensed dentist. "
    "Use ONLY the AI findings provided. The AI is a single-tooth image classifier with limited "
    "accuracy (weaker on cavities and impacted teeth), so never state a definitive diagnosis. "
    "Write a concise report (about 300 words) with these headings: Summary, What the AI found, "
    "Possible conditions to consider, Severity, Treatment options, Follow-up. If the finding is "
    "Normal, say so but still recommend a clinical examination. If the result is flagged for "
    "review, say clearly that it needs manual checking. End by stating that a dentist must "
    "confirm everything."
)

XRAY_REPORT_SYSTEM = (
    "You are a dental clinical-decision-support assistant writing for a licensed dentist. "
    "Use ONLY the AI findings provided. The AI is an object detector trained on about 700 "
    "panoramic X-rays. It finds roughly 6 in 10 real problems and also raises false alarms, "
    "so never state a definitive diagnosis. Do not invent tooth numbers or locations; refer "
    "to findings by their # numbers. Write a concise report (about 350 words) with these "
    "headings: Summary, Findings (one short line per finding: type, confidence, what it "
    "usually indicates), Suggested next steps (clinical examination, further imaging such as "
    "periapical or bitewing films, and typical treatment options for each condition type), "
    "Limitations. If nothing was detected, say that this does not rule out disease. Mention "
    "any warnings given. End by stating that a dentist must confirm everything."
)

CHAT_SYSTEM = (
    "You are a dental clinical-decision-support assistant answering questions from a licensed "
    "dentist. Answer only dental and oral-health questions; politely decline anything else. "
    "Be accurate, concise and practical. If you are unsure, say so. Never present your answer "
    "as a definitive diagnosis, and remind the dentist to use clinical judgment when it matters."
)


@app.get("/health")
def health():
    return {"status": "ok", "classes": classes, "llm_model": LLM_MODEL,
            "cavity_boost": BOOST, "detector_classes": list(DET_NAMES.values()),
            "llm_key_set": bool(os.getenv("GROQ_API_KEY"))}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(400, "Please upload a PNG or JPG image.")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image too large (max 5 MB).")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise HTTPException(400, "Could not read the image.")
    probs, top = classify(img)
    conf = probs[top]
    warnings = []
    needs_review = conf < CONF_THRESHOLD
    if needs_review:
        warnings.append("Low confidence (%.0f%%). Please review manually." % (conf * 100))
    for c in RARE:
        if c != top and probs.get(c, 0) >= 0.20:
            needs_review = True
            warnings.append("%s probability is %.0f%%. The model is weaker on this "
                            "condition, so please check manually." % (c, probs[c] * 100))
    return {"top_condition": top, "confidence": conf, "probabilities": probs,
            "needs_review": needs_review, "warnings": warnings,
            "disclaimer": DISCLAIMER}


@app.post("/detect")
def detect(file: UploadFile = File(...), conf: float = 0.25):
    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(400, "Please upload a PNG or JPG image.")
    data = file.file.read()
    if len(data) > MAX_XRAY_BYTES:
        raise HTTPException(413, "Image too large (max 15 MB).")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        img = img.convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read the image.")
    conf = min(max(conf, 0.05), 0.90)
    res = DET.predict(img, conf=conf, imgsz=1024, device="cpu",
                      verbose=False, max_det=50)[0]
    dets = []
    for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.tolist(),
                                      res.boxes.conf.tolist(),
                                      res.boxes.cls.tolist()):
        dets.append({"label": DET_NAMES[int(k)], "confidence": round(c, 4),
                     "box": {"x": round(x1, 1), "y": round(y1, 1),
                             "w": round(x2 - x1, 1), "h": round(y2 - y1, 1)}})
    dets.sort(key=lambda d: -d["confidence"])
    for i, d in enumerate(dets, 1):
        d["id"] = i
    W, H = img.size
    warnings = []
    if W / H < 1.3:
        warnings.append("This does not look like a panoramic X-ray. The detector was "
                        "trained on panoramic images, so results may be unreliable.")
    if W < 800:
        warnings.append("The image resolution is low, so small problems may be missed.")
    low = [d for d in dets if d["confidence"] < LOW_CONF]
    if low:
        warnings.append("%d finding(s) have low confidence (under %d%%). "
                        "Please check them manually." % (len(low), int(LOW_CONF * 100)))
    if not dets:
        warnings.append("No suspicious areas found at this sensitivity. This does not "
                        "mean the X-ray is healthy.")
    counts = {}
    for d in dets:
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    return {"image_width": W, "image_height": H, "conf_threshold": conf,
            "detections": dets, "counts": counts,
            "needs_review": bool(low) or bool(warnings),
            "warnings": warnings, "disclaimer": DISCLAIMER}


@app.post("/report")
def report(f: Findings):
    text = call_llm(REPORT_SYSTEM,
                    [{"role": "user", "content": findings_text(f) + "\n\nWrite the report."}])
    return {"report": text, "disclaimer": DISCLAIMER}


@app.post("/report_xray")
def report_xray(f: XrayFindings):
    text = call_llm(XRAY_REPORT_SYSTEM,
                    [{"role": "user", "content": xray_text(f) + "\n\nWrite the report."}])
    return {"report": text, "disclaimer": DISCLAIMER}


@app.post("/chat")
def chat(req: ChatRequest):
    system = CHAT_SYSTEM
    if req.findings:
        system += "\n\nCurrent patient context:\n" + findings_text(req.findings)
    if req.xray:
        system += "\n\nCurrent X-ray context:\n" + xray_text(req.xray)
    hist = [m.model_dump() for m in req.history][-10:]
    while hist and hist[0]["role"] != "user":
        hist.pop(0)
    hist.append({"role": "user", "content": req.question})
    return {"answer": call_llm(system, hist, 1500), "disclaimer": DISCLAIMER}
