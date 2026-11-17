"""Build the tables-only interaction deliverable from saved measurements.

Run from any directory: python scripts/build_interaction_summary.py
Optional document dependency: reportlab==4.4.9. No training is performed.
"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
bench = json.loads((ROOT / 'performance_benchmarks.json').read_text())
orderings = json.loads((ROOT / 'results/orderings.json').read_text())['orderings']
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCustom', fontName='Helvetica', fontSize=10,
                          leading=14, spaceAfter=9, alignment=TA_LEFT))
styles['Title'].textColor = colors.HexColor('#17324D')
styles['Heading2'].textColor = colors.HexColor('#17324D')
styles['Heading2'].spaceBefore = 12
story = []


def table(rows, widths):
    cells = [[Paragraph(str(c), styles['BodyCustom']) for c in row] for row in rows]
    t = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E3EDF5')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F3F6F8')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LINEBELOW', (0, 0), (-1, 0), 0.7, colors.HexColor('#A7BDCE')),
    ]))
    story.extend([t, Spacer(1, 10)])


story.append(Paragraph('Compression technique interactions', styles['Title']))
story.append(Spacer(1, 12))
rows = [['Experiment / saved format', 'Test accuracy', 'Loss (pp)', 'Size reduction']]
choices = [
    ('Baseline / Keras', 'baseline'),
    ('Pruning only / Keras', 'interaction_individual_controls_pruning_model'),
    ('Quantization only / INT8 TFLite', 'interaction_individual_controls_quantization_tflite_path'),
    ('Distillation only / Keras', 'interaction_individual_controls_distillation_model'),
    ('Pruning then Quantization / INT8 TFLite', 'interaction_pruning_then_quantization_tflite_path'),
    ('Quantization then Pruning / INT8 TFLite', 'interaction_quantization_then_pruning_tflite_path'),
    ('Pruning with Distillation / Keras', 'interaction_pruning_with_distillation_model'),
    ('Quantization with Distillation / INT8 TFLite', 'interaction_quantization_with_distillation_tflite_path'),
    ('Pruning then Quantization then Distillation / INT8 TFLite', 'interaction_all_three_combined_tflite_path'),
]
for label, key in choices:
    r = bench[key]
    rows.append([label, f"{100*r['test_accuracy']:.2f}%", f"{r['accuracy_loss_percent']:.2f}",
                 f"{r['compression_ratio']:.2f}x"])
table(rows, [224, 84, 72, 111])
story.append(Spacer(1, 24))
rows = [['Sequence (reference models)', 'Validation accuracy', 'Size reduction', 'Training time']]
for key, r in orderings.items():
    label = key.capitalize()
    rows.append([label, f"{100*r['final_accuracy']:.2f}%", f"{r['compression_ratio']:.2f}x",
                 f"{r['training_time_seconds']/60:.1f} min"])
table(rows, [224, 94, 84, 89])
if __name__ == '__main__':
    output = ROOT / 'interaction_analysis_summary.pdf'
    SimpleDocTemplate(str(output), pagesize=A4, rightMargin=52, leftMargin=52,
                      topMargin=42, bottomMargin=45,
                      title='Compression technique interactions',
                      author='').build(story)
    print(output)
