from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    KeepTogether, HRFlowable
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from pathlib import Path

ROOT = Path('/Users/prathmesh/Documents/Anomaly detection')
OUT = ROOT / 'output/pdf/MITS_SpatioTemporal_Anomaly_Detection_Milestone_Report.pdf'

NAVY = HexColor('#102A43')
BLUE = HexColor('#1F6FEB')
TEAL = HexColor('#007C91')
SKY = HexColor('#EAF2FF')
MINT = HexColor('#E8F6F3')
ORANGE = HexColor('#F59E0B')
GREY = HexColor('#52616B')
LIGHT = HexColor('#F5F7FA')

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='ReportTitle', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=24,
                          leading=29, textColor=NAVY, alignment=TA_CENTER, spaceAfter=12))
styles.add(ParagraphStyle(name='Subtitle', parent=styles['BodyText'], fontName='Helvetica', fontSize=12,
                          leading=17, textColor=GREY, alignment=TA_CENTER, spaceAfter=12))
styles.add(ParagraphStyle(name='H1x', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=16,
                          leading=20, textColor=NAVY, spaceBefore=4, spaceAfter=9))
styles.add(ParagraphStyle(name='H2x', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12,
                          leading=16, textColor=TEAL, spaceBefore=9, spaceAfter=5))
styles.add(ParagraphStyle(name='Bodyx', parent=styles['BodyText'], fontName='Helvetica', fontSize=9.4,
                          leading=14, textColor=HexColor('#1F2933'), spaceAfter=7))
styles.add(ParagraphStyle(name='Small', parent=styles['BodyText'], fontName='Helvetica', fontSize=8,
                          leading=10.5, textColor=GREY, spaceAfter=4))
styles.add(ParagraphStyle(name='Callout', parent=styles['BodyText'], fontName='Helvetica-Bold', fontSize=10,
                          leading=14, textColor=NAVY, leftIndent=8, rightIndent=8, spaceBefore=4, spaceAfter=7))

def P(text, style='Bodyx'):
    return Paragraph(text, styles[style])

def header_footer(canv, doc):
    canv.saveState()
    canv.setStrokeColor(BLUE)
    canv.setLineWidth(.7)
    canv.line(doc.leftMargin, A4[1] - 1.25*cm, A4[0] - doc.rightMargin, A4[1] - 1.25*cm)
    canv.setFont('Helvetica', 8)
    canv.setFillColor(GREY)
    canv.drawString(doc.leftMargin, .7*cm, 'MITS Capstone - Spatio-Temporal Anomaly Detection')
    canv.drawRightString(A4[0] - doc.rightMargin, .7*cm, f'Page {doc.page}')
    canv.restoreState()

def section(title):
    return [P(title, 'H1x'), HRFlowable(width='100%', thickness=.7, color=HexColor('#B8D3F8')), Spacer(1, 4)]

def make_table(rows, widths, header=True, font=8.2):
    data = [[Paragraph(str(v), styles['Small']) for v in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign='LEFT')
    commands = [
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), .35, HexColor('#D9E2EC')),
        ('LEFTPADDING', (0,0), (-1,-1), 6), ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 5), ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('FONTSIZE', (0,0), (-1,-1), font),
    ]
    if header:
        commands += [('BACKGROUND', (0,0), (-1,0), NAVY), ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                     ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')]
    for r in range(1 if header else 0, len(rows)):
        if r % 2 == 0:
            commands.append(('BACKGROUND', (0,r), (-1,r), LIGHT))
    t.setStyle(TableStyle(commands))
    return t

def box(d, x, y, w, h, label, fill=SKY, fs=8.3):
    d.add(Rect(x, y, w, h, rx=7, ry=7, fillColor=fill, strokeColor=BLUE, strokeWidth=1))
    lines = label.split('\n')
    total = len(lines) * (fs + 2)
    for i, line in enumerate(lines):
        tw = stringWidth(line, 'Helvetica-Bold', fs)
        d.add(String(x + (w-tw)/2, y + h/2 + total/2 - (i+1)*(fs+2), line,
                     fontName='Helvetica-Bold', fontSize=fs, fillColor=NAVY))

def arrow(d, x1, y1, x2, y2, color=GREY):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.2))
    # simple arrow head
    d.add(Line(x2, y2, x2-5, y2+3, strokeColor=color, strokeWidth=1.2))
    d.add(Line(x2, y2, x2-5, y2-3, strokeColor=color, strokeWidth=1.2))

def training_diagram():
    d = Drawing(480, 220)
    box(d, 12, 157, 105, 43, 'UCF-Crime videos\nGoogle Drive', MINT)
    box(d, 132, 157, 105, 43, 'Manifest and\npreprocessing')
    box(d, 252, 157, 105, 43, '16-frame clips\nstride 8')
    box(d, 372, 157, 96, 43, 'Custom 3D CNN\nfrom scratch', SKY)
    box(d, 74, 61, 125, 43, 'MIL ranking loss\n+ smoothness/sparsity', HexColor('#FFF4D6'))
    box(d, 237, 61, 110, 43, 'Validation and\nthreshold tuning', MINT)
    box(d, 375, 61, 93, 43, 'Drive outputs\n.pt, CSV, plots', SKY)
    for x1,x2 in [(117,132),(237,252),(357,372)]: arrow(d,x1,178,x2,178)
    arrow(d,420,157,137,104); arrow(d,199,82,237,82); arrow(d,347,82,375,82)
    return d

def inference_diagram():
    d = Drawing(480, 110)
    labels = [('Input video', 8, MINT), ('Decode and\nsample', 104, SKY), ('16-frame\nwindow', 200, SKY),
              ('Saved 3D CNN', 296, HexColor('#FFF4D6')), ('Score + EMA\nthreshold', 392, SKY)]
    for label,x,fill in labels:
        box(d,x,40,78,42,label,fill,7.7)
    for x in [86,182,278,374]: arrow(d,x,61,x+18,61)
    d.add(String(165, 11, 'Outputs: clip-score CSV, score plot, annotated MP4', fontName='Helvetica-Bold', fontSize=8.3, fillColor=TEAL))
    return d

story = []
story += [Spacer(1, 2.0*cm), P('Milestone Report', 'ReportTitle'),
          P('Spatio-Temporal Video Anomaly Detection using a 3D CNN', 'Subtitle'),
          Spacer(1, 9), HRFlowable(width='78%', thickness=1.4, color=BLUE), Spacer(1, 14)]
cover = [
    ['Institution', 'Madhav Institute of Technology and Science (MITS)'],
    ['Implementation environment', 'Google Colab with Google Drive persistence'],
    ['Model approach', 'PyTorch 3D CNN trained from random initialization'],
    ['Dataset strategy', 'UCF-Crime untrimmed surveillance videos with weak video-level labels'],
    ['Current milestone', 'Architecture, training plan, evaluation plan, and reproducible Colab workflow'],
]
story.append(make_table(cover, [4.4*cm, 11.6*cm], header=False, font=9))
story += [Spacer(1, 20), P('<b>Project objective.</b> Build an intelligent video-analysis pipeline that learns motion and appearance jointly, scores abnormal activity in untrimmed surveillance video, and produces reviewable evidence without relying on pretrained feature extractors.', 'Callout'),
          Spacer(1, 20), P('Prepared from the approved project README and system architecture documents.', 'Small'), PageBreak()]

story += section('1. Executive Summary')
story.append(P('Conventional surveillance depends heavily on continuous human observation or frame-by-frame object detection. Those approaches do not reliably capture a behaviour that unfolds across time, such as a fight, accident, theft, or robbery. This capstone addresses that gap with a spatio-temporal anomaly detector that evaluates short video volumes instead of isolated frames.'))
story.append(P('The project is intentionally scoped for reproducible implementation in Google Colab. A custom PyTorch 3D CNN is trained from randomly initialized weights on UCF-Crime-derived video bags. It outputs an anomaly score for every overlapping 16-frame clip. Multiple-instance learning (MIL) permits training from video-level normal/anomalous labels, while smoothness and sparsity regularization encourage temporally stable and localized predictions.'))
story.append(P('The milestone deliverable is not a production camera platform. It is a complete training, validation, testing, and offline video-inference workflow that preserves checkpoints, configurations, metrics, score timelines, and annotated MP4 outputs in Google Drive.'))
story += section('2. Problem Statement and Objectives')
goals = [
    ['Objective', 'Milestone response'],
    ['Recognize temporal behaviour', 'Use 3D convolutions over 16-frame clips so motion and appearance are learned together.'],
    ['Work with long untrimmed video', 'Treat each video as a bag of clips and train with video-level labels using MIL ranking.'],
    ['Train from scratch', 'Implement a custom PyTorch 3D CNN with random initialization; do not use pretrained backbones or extracted C3D features.'],
    ['Provide reviewable outcomes', 'Save timestamped score CSVs, score plots, evaluation reports, and annotated MP4 videos.'],
    ['Maintain reproducibility', 'Use a deterministic manifest, seeded runs, pinned requirements, saved configuration, and Drive-backed checkpoints.'],
]
story.append(make_table(goals, [4.5*cm, 11.5*cm]))
story.append(PageBreak())

story += section('3. Proposed System Architecture')
story.append(P('The architecture separates model development from offline inference. Google Drive is the persistent storage layer, and Google Colab provides the GPU runtime. This avoids dependence on a long-running server while retaining all artefacts needed for a demonstration or result reproduction.'))
story.append(training_diagram())
story.append(P('<b>Figure 1.</b> Offline training architecture. The 3D CNN and anomaly head are optimized end-to-end from scratch using clip bags created from untrimmed videos.', 'Small'))
story.append(P('Input frames are decoded at a fixed target frame rate, resized to 128 x 171, cropped or resized to 112 x 112, and normalized consistently. Clips have tensor shape [C=3, T=16, H=112, W=112] and use an 8-frame overlap. Timestamp metadata travels with every clip to support temporal evaluation and annotated output generation.'))
story += section('4. Model and Learning Strategy')
model_table = [
    ['Layer / mechanism', 'Design decision', 'Reason'],
    ['3D CNN backbone', 'Custom PyTorch architecture with 3 x 3 x 3 convolution kernels and random initialization', 'Retains temporal information through the network and fulfills the from-scratch requirement.'],
    ['Input clip', 'RGB tensor: 3 x 16 x 112 x 112; stride 8', 'Balances temporal context, overlap, and Colab compute limits.'],
    ['Anomaly head', 'Learned features followed by sigmoid score', 'Returns one anomaly probability-like score for each clip.'],
    ['Training bag', 'All clips from one normal or anomalous video', 'Uses available video-level labels without requiring expensive segment labels.'],
    ['Calibration', 'Validation-set threshold only', 'Prevents test-set leakage and makes final evaluation credible.'],
]
story.append(make_table(model_table, [3.3*cm, 6.2*cm, 6.5*cm]))
story.append(PageBreak())

story += section('5. MIL Loss and Temporal Decision Logic')
story.append(P('In an anomalous bag A and a normal bag N, the network should rank the highest-scoring clip in A above the highest-scoring clip in N by a margin m. The loss used for the project is:'))
story.append(KeepTogether([P('<b>L_rank = max(0, m - max(s(A)) + max(s(N)))</b>', 'Callout'),
                            P('The total objective augments the ranking term with sparsity and temporal smoothness:', 'Bodyx'),
                            P('<b>L = L_rank + lambda_sparse sum(s(A)) + lambda_smooth sum(|s_t - s_(t-1)|)</b>', 'Callout')]))
story.append(P('Sparsity represents the fact that an anomalous event often occupies only a short part of an untrimmed video. Smoothness reduces unstable score changes between adjacent clips. During video prediction, scores are additionally smoothed with an exponential moving average (EMA). An anomaly interval begins after N consecutive clips exceed a threshold and closes after M clips fall below it.'))
story.append(inference_diagram())
story.append(P('<b>Figure 2.</b> Offline inference workflow. The same preprocessing module used during training is reused at prediction time.', 'Small'))
story += section('6. Google Colab Workflow')
workflow = [
    ['Notebook', 'Main responsibility', 'Persistent output'],
    ['01_setup.ipynb', 'Mount Drive, verify GPU, install pinned packages, set seeds and paths.', 'Environment record and run configuration'],
    ['02_prepare_data.ipynb', 'Create deterministic manifests, validate videos, inspect clips and labels.', 'CSV manifests and quality report'],
    ['03_train.ipynb', 'Define the 3D CNN, MIL loss, optimizer, training loop, and checkpoint schedule.', 'best_model.pt, last_model.pt, logs'],
    ['04_evaluate.ipynb', 'Load the frozen best checkpoint and compute held-out metrics.', 'metrics.json, ROC/PR curves, error analysis'],
    ['05_predict_video.ipynb', 'Score an input video and create visual evidence.', 'prediction CSV, score plot, annotated MP4'],
]
story.append(make_table(workflow, [3.4*cm, 7.2*cm, 5.4*cm]))
story.append(Spacer(1, 8))
story.append(P('<b>Google Drive layout</b><br/>MyDrive/anomaly_detection/data/ucf_crime - raw videos and manifests<br/>MyDrive/anomaly_detection/checkpoints - best and latest checkpoints<br/>MyDrive/anomaly_detection/runs - logs, curves, and metrics<br/>MyDrive/anomaly_detection/outputs - annotated MP4s and score CSVs', 'Callout'))
story.append(PageBreak())

story += section('7. Team Responsibilities and Implementation Status')
team = [
    ['Module', 'Owner focus', 'Milestone deliverable'],
    ['1. Data pipeline', 'Prepare UCF-Crime manifests, decode/normalize frames, build overlapping clips with timestamps.', 'Verified data contract and reusable preprocessing functions.'],
    ['2. 3D CNN training', 'Implement from-scratch model, MIL loss, validation loop, checkpoints, and metrics.', 'Training/evaluation notebook design and experiment configuration.'],
    ['3. Integration and presentation', 'Maintain shared Colab modules; generate plots and annotated video; document result reproduction.', 'Demonstration workflow and visual result assets.'],
]
story.append(make_table(team, [3.1*cm, 7.1*cm, 5.8*cm]))
story.append(Spacer(1, 10))
story.append(P('At this milestone, the project has a documented end-to-end architecture, model-learning formulation, Colab notebook plan, storage convention, evaluation protocol, and risk controls. Implementation proceeds by first validating the data pipeline on a small approved subset, then training the custom network and finally evaluating a frozen checkpoint on held-out videos.'))
story += section('8. Evaluation and Verification Plan')
verify = [
    ['Gate', 'Evidence to submit'],
    ['Dataset integrity', 'No corrupted videos; deterministic split manifests; no train/validation/test leakage; clip timestamps verified.'],
    ['Model quality', 'Validation ROC-AUC, PR-AUC, false alarms per video/hour, time-to-detect, and category-wise analysis.'],
    ['Threshold integrity', 'Threshold selected on validation data and frozen before final test evaluation.'],
    ['Reproducibility', 'Saved seed, configuration, manifest version, checkpoint hash, package versions, and repeated evaluation.'],
    ['Inference output', 'Timestamp-aligned prediction CSV, readable score plot, and playable annotated MP4.'],
    ['Failure analysis', 'Representative low-light, crowding, occlusion, camera-shake, and unseen-normal cases.'],
]
story.append(make_table(verify, [4.1*cm, 11.9*cm]))
story.append(PageBreak())

story += section('9. Risks and Mitigations')
risks = [
    ['Risk', 'Impact', 'Mitigation'],
    ['Class imbalance and novel anomalies', 'Missed anomalies or false positives', 'Use normal and anomalous bags, report per-category performance, and retain uncertain examples for analysis.'],
    ['Darkness, occlusion, crowding, camera shake', 'Scores may reflect video quality rather than activity', 'Include failure-case review and diverse validation examples; do not overstate automated decisions.'],
    ['Colab session resets', 'Loss of progress', 'Save checkpoints, configuration, and logs to Drive at every epoch; resume from latest checkpoint.'],
    ['Limited GPU time', 'Slow experimentation', 'Debug on a small approved subset, use efficient input settings, mixed precision where supported, and resumable runs.'],
    ['Inconsistent transforms', 'Unreliable test results', 'Centralize preprocessing and record transforms in run_config.json.'],
]
story.append(make_table(risks, [4.1*cm, 4.1*cm, 7.8*cm]))
story += section('10. Scope, Ethics, and Next Milestone')
story.append(P('The model is a decision-support system. It flags unusual activity for review; it does not prove criminal intent, identify people, or replace trained security personnel. In particular, anomaly labels must be interpreted carefully because rare behaviour can be harmless and surveillance footage may be biased by lighting, camera location, and data collection practices.'))
next_steps = [
    ['Next milestone activity', 'Expected artefact'],
    ['Implement and smoke-test preprocessing on a small subset', 'Manifest, sampled clip visualizations, and shape/timestamp tests'],
    ['Train baseline custom 3D CNN', 'Checkpoint, learning curves, and validation report'],
    ['Tune threshold and evaluate frozen model', 'Test metrics, ROC/PR curves, false-alarm analysis'],
    ['Demonstrate offline inference', 'Annotated UCF-Crime/user video and score timeline'],
]
story.append(make_table(next_steps, [7.3*cm, 8.7*cm]))
story += [Spacer(1, 13), P('Conclusion', 'H2x'), P('The approved design offers a technically grounded route from untrimmed surveillance video to explainable, timestamped anomaly evidence. The immediate focus is a reproducible, from-scratch 3D CNN implementation in Google Colab, with carefully controlled training, validation, and evaluation rather than premature real-time deployment claims.')]
story += section('11. Research Basis')
refs = [
    ['Reference', 'Contribution to this project'],
    ['Tran et al., Learning Spatiotemporal Features with 3D Convolutional Networks', 'Supports 3D convolutions, 3 x 3 x 3 kernels, 16-frame clips, and overlapping temporal feature extraction.'],
    ['Sultani, Chen, and Shah, Real-world Anomaly Detection in Surveillance Videos', 'Supports UCF-Crime, weak video-level supervision, MIL ranking, temporal smoothness, and sparsity constraints.'],
    ['Pang et al., Deep Learning for Anomaly Detection: A Review', 'Highlights rare and heterogeneous anomalies, class imbalance, false positives, novelty, and the need for human review.'],
]
story.append(make_table(refs, [7.0*cm, 9.0*cm]))

doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=1.35*cm, leftMargin=1.35*cm,
                        topMargin=1.65*cm, bottomMargin=1.35*cm, title='MITS Milestone Report - Spatio-Temporal Anomaly Detection', author='MITS Capstone Team')
doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
print(OUT)
