import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def clean_text_for_reportlab(text):
    # Convert markdown bold/italic/code to HTML tags supported by ReportLab Paragraph
    # Replace **bold** with <b>bold</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Replace *italic* with <i>italic</i>
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    # Replace `code` with <font name="Courier">\1</font>
    text = re.sub(r'`(.*?)`', r'<font name="Courier" color="#8B0000">\1</font>', text)
    # Strip any remaining unparsed markdown symbols like $[...]$ or \[...\]
    text = re.sub(r'\$+', '', text)
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    # Restore reportlab tags
    text = text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
    text = text.replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')
    text = text.replace('&lt;font name="Courier" color="#8B0000"&gt;', '<font name="Courier" color="#8B0000">').replace('&lt;/font&gt;', '</font>')
    return text

def parse_markdown_to_pdf(md_filepath, pdf_filepath):
    print(f"Converting {md_filepath} -> {pdf_filepath}")
    with open(md_filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    doc = SimpleDocTemplate(
        pdf_filepath,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1A202C'),
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'DocH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2D3748'),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )

    h3_style = ParagraphStyle(
        'DocH3',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#4A5568'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=body_style,
        leftIndent=15,
        spaceAfter=4
    )

    quote_style = ParagraphStyle(
        'DocQuote',
        parent=body_style,
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#2C5282'),
        backColor=colors.HexColor('#EBF8FF'),
        borderColor=colors.HexColor('#3182CE'),
        borderWidth=1,
        borderPadding=6,
        spaceBefore=6,
        spaceAfter=8
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#1A202C'),
        alignment=0
    )

    story = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Horizontal Rule
        if line.startswith('---'):
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E2E8F0'), spaceAfter=10, spaceBefore=10))
            i += 1
            continue

        # Headers
        if line.startswith('# '):
            story.append(Paragraph(clean_text_for_reportlab(line[2:]), title_style))
            i += 1
            continue
        elif line.startswith('## '):
            story.append(Paragraph(clean_text_for_reportlab(line[3:]), h1_style))
            i += 1
            continue
        elif line.startswith('### '):
            story.append(Paragraph(clean_text_for_reportlab(line[4:]), h2_style))
            i += 1
            continue
        elif line.startswith('#### '):
            story.append(Paragraph(clean_text_for_reportlab(line[5:]), h3_style))
            i += 1
            continue

        # Images
        img_match = re.match(r'!\[(.*?)\]\((.*?)\)', line)
        if img_match:
            caption, img_path = img_match.group(1), img_match.group(2)
            img_path = os.path.normpath(img_path)
            if os.path.exists(img_path):
                try:
                    img = Image(img_path, width=6.5*inch, height=3.5*inch, kind='proportional')
                    story.append(Spacer(1, 4))
                    story.append(img)
                    story.append(Paragraph(f"<b>Figure:</b> <i>{clean_text_for_reportlab(caption)}</i>", ParagraphStyle('Cap', parent=body_style, fontSize=8, alignment=1, spaceAfter=8)))
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
            i += 1
            continue

        # Blockquotes / Alerts
        if line.startswith('>'):
            quote_lines = []
            while i < len(lines) and lines[i].startswith('>'):
                quote_lines.append(lines[i].lstrip('> ').rstrip())
                i += 1
            quote_text = " ".join(quote_lines)
            quote_text = re.sub(r'\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]', r'<b>[\1]</b>', quote_text)
            story.append(Paragraph(clean_text_for_reportlab(quote_text), quote_style))
            continue

        # Markdown Tables
        if '|' in line and i + 1 < len(lines) and '|' in lines[i+1] and ('---' in lines[i+1] or ':-' in lines[i+1]):
            table_lines = []
            while i < len(lines) and '|' in lines[i]:
                table_lines.append(lines[i].strip())
                i += 1

            # Process table
            headers = [c.strip() for c in table_lines[0].split('|')[1:-1]]
            rows = []
            for tl in table_lines[2:]:
                cells = [c.strip() for c in tl.split('|')[1:-1]]
                if len(cells) == len(headers):
                    rows.append(cells)

            table_data = []
            # Header row
            table_data.append([Paragraph(clean_text_for_reportlab(h), table_header_style) for h in headers])
            # Body rows
            for r in rows:
                table_data.append([Paragraph(clean_text_for_reportlab(c), table_cell_style) for c in r])

            col_widths = [504 / len(headers)] * len(headers)
            t = Table(table_data, colWidths=col_widths)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2B6CB0')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F7FAFC'), colors.white]),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
            ]))
            story.append(Spacer(1, 4))
            story.append(t)
            story.append(Spacer(1, 8))
            continue

        # Bullet Points
        if line.startswith('- ') or line.startswith('* '):
            story.append(Paragraph(f"• {clean_text_for_reportlab(line[2:])}", bullet_style))
            i += 1
            continue

        # Regular Paragraph
        story.append(Paragraph(clean_text_for_reportlab(line), body_style))
        i += 1

    doc.build(story)
    print(f"Successfully generated PDF: {pdf_filepath}")

if __name__ == "__main__":
    base_dir = r"C:\Users\Sanjay Pandey\Downloads\WorldModelLens-pvl\WorldModelLens-pvl"
    parse_markdown_to_pdf(
        os.path.join(base_dir, "docs", "whitepaper_pvl.md"),
        os.path.join(base_dir, "docs", "whitepaper_pvl.pdf")
    )
    parse_markdown_to_pdf(
        os.path.join(base_dir, "docs", "whitepaper_aaf.md"),
        os.path.join(base_dir, "docs", "whitepaper_aaf.pdf")
    )
    parse_markdown_to_pdf(
        os.path.join(base_dir, "docs", "whitepaper_pvl_aaf.md"),
        os.path.join(base_dir, "docs", "whitepaper_pvl_aaf.pdf")
    )
