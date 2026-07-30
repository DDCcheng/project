"""Build the compact English COMP9444 summary report PDF."""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs" / "COMP9444_SUMMARY_REPORT.md"
OUTPUT = ROOT / "docs" / "COMP9444_SUMMARY_REPORT.pdf"
FIGURE = ROOT / "results" / "figures" / "validation_macro_f1_comparison.png"


def clean_inline(text: str) -> str:
    """Convert the small Markdown subset used by the report to safe text."""

    text = text.replace("**", "")
    text = text.replace("`", "")
    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
    return text


def add_page_number(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawRightString(7.5 * inch, 0.42 * inch, f"COMP9444 Summary Report | {document.page}")
    canvas.restoreState()


def build_story() -> list:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=19,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17324D"),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor("#17324D"),
            spaceBefore=7,
            spaceAfter=4,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.4,
            leading=10.5,
            spaceAfter=4,
            textColor=colors.HexColor("#1F2937"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBullet",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=10,
            leftIndent=12,
            firstLineIndent=-7,
            bulletIndent=0,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.3,
            leading=8.8,
            textColor=colors.HexColor("#475467"),
        )
    )

    story = []
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    inserted_results_figure = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 2))
            continue
        if stripped.startswith("# "):
            story.append(Paragraph(clean_inline(stripped[2:]), styles["ReportTitle"]))
            continue
        if stripped.startswith("## "):
            heading = clean_inline(stripped[3:])
            story.append(Paragraph(heading, styles["ReportHeading"]))
            if heading.startswith("5. Results") and FIGURE.exists() and not inserted_results_figure:
                image = Image(str(FIGURE))
                image._restrictSize(6.4 * inch, 1.55 * inch)
                story.append(image)
                story.append(Spacer(1, 3))
                inserted_results_figure = True
            continue
        if stripped.startswith("### "):
            story.append(Paragraph(clean_inline(stripped[4:]), styles["ReportHeading"]))
            continue
        if stripped.startswith("- "):
            story.append(Paragraph(clean_inline(stripped[2:]), styles["ReportBullet"], bulletText="-"))
            continue
        if re.match(r"^\d+\. ", stripped):
            story.append(Paragraph(clean_inline(re.sub(r"^\d+\. ", "", stripped)), styles["ReportBullet"]))
            continue
        if stripped.startswith("`") and stripped.endswith("`"):
            story.append(Paragraph(clean_inline(stripped), styles["ReportSmall"]))
            continue
        story.append(Paragraph(clean_inline(stripped), styles["ReportBody"]))

    story.append(Spacer(1, 5))
    table = Table(
        [
            ["Configuration", "Validation Macro-F1", "Test Macro-F1"],
            ["Frozen ResNet-18 baseline", "0.9761", "-"],
            ["DenseNet-201 + SVM, no augmentation", "0.9878", "0.9828"],
            ["ResNeXt-101 + PCA + SVM", "0.9528", "-"],
        ],
        colWidths=[3.8 * inch, 1.25 * inch, 1.25 * inch],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("LEADING", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.insert(1, Spacer(1, 3))
    story.insert(2, table)
    story.insert(3, Spacer(1, 7))
    return story


def main() -> None:
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.48 * inch,
        bottomMargin=0.62 * inch,
        title="COMP9444 Summary Report",
        author="COMP9444 Group Project",
    )
    document.build(build_story(), onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(OUTPUT)


if __name__ == "__main__":
    main()
