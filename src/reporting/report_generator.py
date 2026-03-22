from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER
import os
import json

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_PATH = os.path.join(BASE_DIR, "data", "output", "market_dashboard_data.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "output", "market_report.pdf")


def clean_text(text):

    # 🔥 FIX: handle tuple (answer, sources)
    if isinstance(text, tuple):
        text = text[0]

    # 🔥 FIX: handle list
    if isinstance(text, list):
        text = " ".join([str(t) for t in text])

    if text is None:
        return ""

    text = str(text)

    return text.replace("*", "<br/>•").replace("\n", "<br/>")

def generate_pdf_report():

    if not os.path.exists(INPUT_PATH):
        print("❌ Dashboard data not found")
        return

    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    doc = SimpleDocTemplate(OUTPUT_PATH, pagesize=A4)
    styles = getSampleStyleSheet()

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

    elements.append(Paragraph("<b>Market Overview</b>", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph(f"Trend Direction: <b>{overview['trend_direction']}</b>", styles["Normal"]))
    elements.append(Paragraph(f"Trend Slope: {overview['trend_slope']}", styles["Normal"]))
    elements.append(Paragraph(f"Current Sentiment: {overview['current_sentiment']}", styles["Normal"]))

    elements.append(Spacer(1, 16))

    # -------------------------
    # BRAND INSIGHTS
    # -------------------------
    brand = data["brand_insights"]

    elements.append(Paragraph("<b>Brand Insights</b>", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph(f"Top Positive Brand: <b>{brand['top_positive_brand']}</b>", styles["Normal"]))
    elements.append(Paragraph(f"Top Negative Brand: <b>{brand['top_negative_brand']}</b>", styles["Normal"]))
    elements.append(Paragraph(f"Most Volatile Brand: {brand['most_volatile_brand']}", styles["Normal"]))

    elements.append(Spacer(1, 16))

    # -------------------------
    # TOPIC INSIGHTS
    # -------------------------
    topic = data["topic_insights"]

    elements.append(Paragraph("<b>Topic Insights</b>", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph(f"Top Topics: {', '.join(topic['top_topics'])}", styles["Normal"]))
    elements.append(Paragraph(f"Rising Topic: {topic['fastest_rising_topic']}", styles["Normal"]))
    elements.append(Paragraph(f"Declining Topic: {topic['fastest_declining_topic']}", styles["Normal"]))

    elements.append(Spacer(1, 16))

    # -------------------------
    # AI INSIGHTS
    # -------------------------
    elements.append(Paragraph("<b>AI Market Insight</b>", styles["Heading2"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(clean_text(data["ai_insight"]), styles["Normal"]))

    elements.append(Spacer(1, 16))

    elements.append(Paragraph("<b>Brand AI Insight</b>", styles["Heading2"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(clean_text(data["brand_ai_insight"]), styles["Normal"]))

    elements.append(Spacer(1, 16))

    elements.append(Paragraph("<b>Topic AI Insight</b>", styles["Heading2"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(clean_text(data["topic_ai_insight"]), styles["Normal"]))

    elements.append(Spacer(1, 16))

    # -------------------------
    # RISK
    # -------------------------
    elements.append(Paragraph("<b>Risk Signals</b>", styles["Heading2"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(clean_text(data["risk_signals"]), styles["Normal"]))

    elements.append(Spacer(1, 16))

    # -------------------------
    # ALERTS
    # -------------------------
    elements.append(Paragraph("<b>Alerts</b>", styles["Heading2"]))
    elements.append(Spacer(1, 6))

    for alert in data["alerts"]:
        elements.append(Paragraph(
            f"• {alert['type']} - <b>{alert['brand']}</b> ({alert['severity']})",
            styles["Normal"]
        ))

    elements.append(Spacer(1, 12))

    doc.build(elements)

    print(f"📄 PDF Report Generated: {OUTPUT_PATH}")