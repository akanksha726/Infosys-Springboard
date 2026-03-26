from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
import re
import os
import json

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_PATH = os.path.join(BASE_DIR, "data", "output", "market_dashboard_data.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "output", "market_report.pdf")


def clean_text(text):
    if isinstance(text, tuple):
        text = text[0]

    if isinstance(text, list):
        text = " ".join([str(t) for t in text])

    if text is None:
        return ""

    text = str(text)

    # -------------------------
    # 🔥 FIX BROKEN WORDS
    # -------------------------
    text = re.sub(r"(\w)\s*•\s*(\w)", r"\1-\2", text)

    # -------------------------
    # 🔥 FIX MARKDOWN BOLD (**)
    # -------------------------
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)

    # -------------------------
    # 🔥 NORMALIZE BULLETS
    # -------------------------
    text = text.replace("\n•", "<br/>• ")

    # -------------------------
    # 🔥 HANDLE NEWLINES
    # -------------------------
    text = text.replace("\n", "<br/>")

    # Remove duplicate breaks
    while "<br/><br/>" in text:
        text = text.replace("<br/><br/>", "<br/>")

    return text.strip()

def generate_pdf_report():

    if not os.path.exists(INPUT_PATH):
        print("❌ Dashboard data not found")
        return

    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    doc = SimpleDocTemplate(OUTPUT_PATH, pagesize=A4)
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='Justify',
        alignment=TA_JUSTIFY,
        leading=14
    ))

    styles.add(ParagraphStyle(
        name='HeadingCustom',
        parent=styles['Heading2'],
        spaceAfter=6
    ))

    elements = []

    # -------------------------
    # TITLE
    # -------------------------
    title = Paragraph(
        "<b><font size=18>AI Market Intelligence Report</font></b>",
        styles["Title"]
    )
    elements.append(title)
    elements.append(Spacer(1, 20))

    # -------------------------
    # MARKET OVERVIEW
    # -------------------------
    overview = data["market_overview"]

    elements.append(Paragraph("<b>Market Overview</b>", styles["HeadingCustom"]))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph(f"Trend Direction: <b>{overview['trend_direction']}</b>", styles["Normal"]))
    elements.append(Paragraph(f"Trend Slope: {overview['trend_slope']}", styles["Normal"]))
    elements.append(Paragraph(f"Current Sentiment: {overview['current_sentiment']}", styles["Normal"]))

    elements.append(Spacer(1, 16))

    # -------------------------
    # BRAND INSIGHTS
    # -------------------------
    brand = data["brand_insights"]

    elements.append(Paragraph("<b>Brand Insights</b>", styles["HeadingCustom"]))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph(f"Top Positive Brand: <b>{brand['top_positive_brand']}</b>", styles["Normal"]))
    elements.append(Paragraph(f"Top Negative Brand: <b>{brand['top_negative_brand']}</b>", styles["Normal"]))
    elements.append(Paragraph(f"Most Volatile Brand: {brand['most_volatile_brand']}", styles["Normal"]))

    elements.append(Spacer(1, 16))

    # -------------------------
    # TOPIC INSIGHTS
    # -------------------------
    topic = data["topic_insights"]

    elements.append(Paragraph("<b>Topic Insights</b>", styles["HeadingCustom"]))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph(f"Top Topics: {', '.join(topic['top_topics'])}", styles["Normal"]))
    elements.append(Paragraph(f"Rising Topic: {topic['fastest_rising_topic']}", styles["Normal"]))
    elements.append(Paragraph(f"Declining Topic: {topic['fastest_declining_topic']}", styles["Normal"]))

    elements.append(Spacer(1, 16))

    # -------------------------
    # AI INSIGHTS
    # -------------------------
    elements.append(Paragraph("<b>AI Market Insight</b>", styles["HeadingCustom"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(clean_text(data["ai_insight"]), styles["Justify"]))

    elements.append(Spacer(1, 16))

    elements.append(Paragraph("<b>Brand AI Insight</b>", styles["HeadingCustom"]))
    elements.append(Spacer(1, 6))
    lines = clean_text(data["brand_ai_insight"]).split("<br/>")

    for line in lines:
        if line.strip():
            elements.append(Paragraph(line.strip(), styles["Justify"]))
            elements.append(Spacer(1, 6))
    elements.append(Spacer(1, 16))

    elements.append(Paragraph("<b>Topic AI Insight</b>", styles["HeadingCustom"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(clean_text(data["topic_ai_insight"]), styles["Justify"]))

    elements.append(Spacer(1, 16))

    # -------------------------
    # RISK
    # -------------------------
    elements.append(Paragraph("<b>Risk Signals</b>", styles["HeadingCustom"]))
    elements.append(Spacer(1, 6))
    risk_text = clean_text(data["risk_signals"]).split("<br/>")

    for line in risk_text:
        if line.strip():
            elements.append(Paragraph(line.strip(), styles["Justify"]))
            elements.append(Spacer(1, 4))

    elements.append(Spacer(1, 16))

    # -------------------------
    # ALERTS
    # -------------------------
    elements.append(Paragraph("<b>Alerts</b>", styles["HeadingCustom"]))
    elements.append(Spacer(1, 6))

    for alert in data["alerts"]:
        elements.append(Paragraph(
            f"<b>{alert['type']}</b> — {alert['brand']} ({alert['severity']})",
            styles["Normal"]
        ))

    elements.append(Spacer(1, 12))

    doc.build(elements)

    print(f"📄 PDF Report Generated: {OUTPUT_PATH}")