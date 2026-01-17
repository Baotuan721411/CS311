import re
import os
from typing import List, Optional
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from dataclasses import dataclass

def simple_tokenize(text):
    if not text:
        return []
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def load_prompt_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def render_prompt(template, variables):
    for k, v in variables.items():
        template = template.replace(f"{{{{{k}}}}}", str(v))
    return template

def find_latest_checkpoint(folder):
    ckpts = [d for d in os.listdir(folder) if d.startswith("checkpoint-")]
    if not ckpts:
        raise FileNotFoundError(f"Không tìm thấy checkpoint trong {folder}")
    latest = max(ckpts, key=lambda x: int(x.split("-")[1]))
    return os.path.join(folder, latest)

def clean_answer(text):
    found = re.findall(r"[ABCD]", str(text).upper())
    if not found:
        return "C"
    return ",".join(sorted(set(found)))

def calculate_score(pred, gold):
    if not pred or not gold:
        return 0.0
    p, g = set(pred.split(",")), set(gold.split(",")) 
    if p == g:
        return 1.0
    if not p.isdisjoint(g):
        return 0.5
    return 0.0

def list_checkpoints_sorted(ckpt_dir: str) -> List[str]:
    checkpoints = []
    for name in os.listdir(ckpt_dir):
        path = os.path.join(ckpt_dir, name)
        if not os.path.isdir(path):
            continue
        match = re.search(r"\d+", name)
        if match:
            order = int(match.group())
            checkpoints.append((order, path))
    checkpoints.sort(key=lambda x: x[0])
    return [p for _, p in checkpoints]


def export_compare_epochs_pdf(summaries, pdf_path):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_path)
    elements = []

    elements.append(Paragraph("Epoch Comparison Report", styles["Title"]))

    model_name = summaries[0]["model_name"] if summaries else "N/A"
    elements.append(Paragraph(f"Model: {model_name}", styles["Normal"]))
    elements.append(Paragraph("<br/>", styles["Normal"]))

    # ---------- TABLE ----------
    table_data = [[
        "Epoch",
        "Full",
        "Partial",
        "Wrong",
        "Total score",
        "Avg score"
    ]]

    for s in summaries:
        table_data.append([
            s["epoch"],
            s["full_correct"],
            s["partial"],
            s["wrong"],
            f"{s['total_score']:.2f}",
            f"{s['avg_score']:.4f}" if s["avg_score"] is not None else "N/A"
        ])

    table = Table(
        table_data,
        colWidths=[60, 60, 60, 60, 90, 90]
    )

    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))

    elements.append(table)
    doc.build(elements)

@dataclass
class Document:
    doc_id: str
    title: str
    snippet: str
    content: str
    
@dataclass
class Topic:
    topic_id: int
    docs: List[Document]



@dataclass
class Question:
    question_id: str
    topic_id: int
    target_event: str
    option_A: Optional[str] = None
    option_B: Optional[str] = None
    option_C: Optional[str] = None
    option_D: Optional[str] = None

@dataclass
class TrainQuestion(Question):
    golden_answer: Optional[str] = None

@dataclass
class TestQuestion(Question):
    pass