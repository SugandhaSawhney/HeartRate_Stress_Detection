import cv2
import gradio as gr
from collections import deque

from face_detector import FaceDetector
from rppg_extractor import RPPGExtractor
from signal_processor import SignalProcessor
from stress_analyser import StressAnalyzer

# Global objects
face_detector = FaceDetector()
rppg_extractor = RPPGExtractor(buffer_size=300, fps=30)
signal_processor = SignalProcessor(fps=30)
stress_analyzer = StressAnalyzer(history_size=20)

# State
bpm_history_for_plot = deque(maxlen=60)
current_results = {
    'bpm': None,
    'stress': 'Analyzing...',
    'stress_score': 0,
    'quality': 0,
    'frames_collected': 0
}


def process_frame(frame):
    """process each webcam frames"""
    if frame is None:
        return None, "No frame", "Waiting...", 0, "Please enable webcam"

    roi, annotated = face_detector.get_forehead_roi(frame)
    display_frame = annotated if annotated is not None else frame.copy()

    if roi is None:
        cv2.putText(
            display_frame,
            "No face detected - look at camera",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        return (
            display_frame,
            "No face detected",
            "Waiting...",
            0,
            "Please face the camera",
        )

    rppg_extractor.add_frame(roi)
    current_results['frames_collected'] += 1
    frames = rppg_extractor.get_buffer_length()

    progress = min(100, int(frames / 60 * 100))
    status_text = f"Collecting data: {frames}/60 frames ({progress}%)"

    if not rppg_extractor.is_ready():
        cv2.putText(
            display_frame, status_text,
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 165, 0), 2,
        )
        cv2.putText(
            display_frame, "Stay still...",
            (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 165, 0), 2,
        )
        return (
            display_frame,
            "Collecting data...",
            "Please wait",
            progress,
            status_text,
        )

    sig = rppg_extractor.get_signal(method='CHROM')

    if sig is None:
        return (
            display_frame,
            "Signal weak",
            "Processing...",
            50,
            "Stay still",
        )

    bpm = signal_processor.get_bpm(sig)
    quality = signal_processor.get_signal_quality(sig)

    bpm_display = "Calculating..."
    stress_display = "Analyzing..."
    advice = "Keep still..."

    if bpm:
        current_results['bpm'] = bpm
        stress_analyzer.add_bpm(bpm)
        bpm_history_for_plot.append(bpm)
        bpm_display = f"{bpm} BPM"

        if bpm < 60:
            bpm_color = (255, 165, 0)
            bpm_category = "Low HR"
        elif bpm <= 100:
            bpm_color = (0, 255, 0)
            bpm_category = "Normal"
        else:
            bpm_color = (0, 0, 255)
            bpm_category = "High HR"

        cv2.putText(
            display_frame,
            f"Heart Rate: {bpm} BPM ({bpm_category})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            bpm_color,
            2,
        )

    if stress_analyzer.is_ready():
        stress_level, stress_score, features = (
            stress_analyzer.get_stress_level()
        )
        if features:
            rmssd_val = features['rmssd']
            mean_bpm_val = features['mean_bpm']
            print(f"RMSSD: {rmssd_val:.1f}ms | Mean BPM: {mean_bpm_val:.1f}")

        current_results['stress'] = stress_level
        current_results['stress_score'] = stress_score
        stress_display = stress_level
        advice = stress_analyzer.get_wellness_advice(stress_level)

        cv2.putText(
            display_frame,
            f"Stress: {stress_level}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    quality_color = (0, 255, 0) if quality > 50 else (0, 165, 255)
    cv2.putText(
        display_frame,
        f"Signal Quality: {quality}%",
        (10, display_frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        quality_color,
        1,
    )

    return display_frame, bpm_display, stress_display, quality, advice


def reset_buffers():
    """Sab reset karo"""
    rppg_extractor.clear()
    stress_analyzer.bpm_history.clear()
    bpm_history_for_plot.clear()
    current_results['bpm'] = None
    current_results['stress'] = 'Analyzing...'
    current_results['frames_collected'] = 0
    return (
        None,
        "Reset done! Start again",
        "Waiting...",
        0,
        "Buffers cleared. Stay still.",
    )


CUSTOM_CSS = """
@import url(
  'https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap'
);

/* ── Root tokens ── */
:root {
    --bg-base:      #0a0c10;
    --bg-surface:   #0f1219;
    --bg-card:      #141720;
    --bg-raised:    #1a1f2e;
    --border:       #1e2535;
    --border-glow:  #e03050;
    --accent-red:   #e03050;
    --accent-red2:  #ff4d6d;
    --accent-teal:  #00c9a7;
    --accent-amber: #f5a623;
    --text-primary: #eef0f5;
    --text-muted:   #6b7491;
    --text-dim:     #3a4060;
    --mono: 'DM Mono', monospace;
    --sans: 'Syne', sans-serif;
    --radius: 12px;
    --radius-sm: 8px;
}

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg-base) !important;
    font-family: var(--sans) !important;
    color: var(--text-primary) !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}

/* ── Header banner ── */
.app-header {
    background: linear-gradient(135deg, #0f1219 0%, #150d14 50%, #0a1018 100%);
    border-bottom: 1px solid var(--border);
    padding: 32px 40px 28px;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(224,48,80,0.12) 0%,
    transparent 70%);
    border-radius: 50%;
}
.app-header::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 120px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,201,167,0.06) 0%,
    transparent 70%);
    border-radius: 50%;
}
.header-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(224,48,80,0.12);
    border: 1px solid rgba(224,48,80,0.3);
    border-radius: 999px;
    padding: 4px 14px;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.12em;
    color: var(--accent-red2);
    margin-bottom: 14px;
    text-transform: uppercase;
}
.header-pill::before {
    content: '';
    width: 6px; height: 6px;
    background: var(--accent-red2);
    border-radius: 50%;
    animation: pulse-dot 1.4s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.7); }
}
.header-title {
    font-family: var(--sans);
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    margin: 0 0 6px;
    line-height: 1.1;
}
.header-title span { color: var(--accent-red); }
.header-sub {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text-muted);
    letter-spacing: 0.05em;
}

/* ── Section wrappers ── */
.section-wrap {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    margin: 0;
}

/* ── Label overrides ── */
label span, .label-wrap span {
    font-family: var(--mono) !important;
    font-size: 10px !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}

/* ── Webcam / image blocks ── */
.webcam-block > div, .webcam-block .wrap {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Metric cards (textboxes as readouts) ── */
.metric-bpm textarea,
.metric-stress textarea {
   font-family: var(--sans) !important;
    font-size: 0.85rem !important;
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--accent-teal) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    line-height: 1.6 !important;
    padding: 14px 16px !important;
}
.metric-stress textarea {
    font-size:  0.85rem !important;
    color: var(--accent-teal) !important;
}
.metric-bpm textarea:hover,
.metric-stress textarea:hover {
    border-color: rgba(224,48,80,0.4) !important;
}

/* ── Advice box ── */
.advice-box textarea {
    font-family: var(--sans) !important;
    font-size: 0.85rem !important;
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--accent-teal) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    line-height: 1.6 !important;
    padding: 14px 16px !important;
}

/* ── Slider ── */
.quality-slider .wrap {
    background: transparent !important;
    border: none !important;
}
.quality-slider input[type=range] {
    accent-color: var(--accent-teal) !important;
    height: 4px !important;
}

/* ── Buttons ── */
button.reset-btn, .reset-btn button {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-muted) !important;
    padding: 10px 22px !important;
    transition: border-color 0.2s, color 0.2s !important;
    cursor: pointer !important;
}
button.reset-btn:hover, .reset-btn button:hover {
    border-color: var(--accent-red) !important;
    color: var(--accent-red2) !important;
}

/* ── Info panel markdown ── */
.info-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 22px;
}
.info-panel p, .info-panel li {
    font-family: var(--mono) !important;
    font-size: 12px !important;
    color: var(--text-muted) !important;
    line-height: 1.8 !important;
}
.info-panel strong {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}
.info-panel h3 {
    font-family: var(--sans) !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
    margin: 0 0 12px !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid var(--border) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 16px 0 !important; }

/* ── General block cleanup ── */
.block, .form { background: transparent !important; border: none !important; }
.gap { gap: 16px !important; }

/* ── Instructions row ── */
.instructions-row {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 24px;
    display: flex;
    gap: 32px;
    align-items: center;
}
.step-item {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 0.04em;
}
.step-num {
    width: 22px; height: 22px;
    border-radius: 50%;
    border: 1px solid var(--accent-red);
    display: flex; align-items: center; justify-content: center;
    font-size: 10px;
    color: var(--accent-red2);
    flex-shrink: 0;
}

/* ── Live readings header ── */
.readings-header {
    font-family: var(--mono) !important;
    font-size: 10px !important;
    font-weight: 500 !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
    margin: 0 0 14px !important;
    display: flex;
    align-items: center;
    gap: 8px;
}
.readings-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
"""

# ── App layout ─────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Vitals Monitor — rPPG",
    theme=gr.themes.Base(
        primary_hue="red",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Syne"),
    ),
    css=CUSTOM_CSS,
) as demo:

    # Header
    gr.HTML("""
    <div class="app-header">
        <div class="header-pill">rPPG · Contactless Sensing</div>
        <div class="header-title">Vitals <span>Monitor</span></div>
        <div class="header-sub">
            Remote Photoplethysmography · Heart Rate · Stress Index
        </div>
    </div>
    """)

    # Instructions strip
    gr.HTML("""
    <div style="padding: 16px 24px 0;">
        <div class="instructions-row">
            <div class="step-item">
                <div class="step-num">1</div>
                <span>
            Click <strong style="color:#eef0f5">Start Webcam</strong> below
            </span>
            </div>
            <div class="step-item">
                <div class="step-num">2</div>
                <span>
            Face the camera in <strong style="color:#eef0f5">
            good lighting</strong>
            </span>
            </div>
            <div class="step-item">
                <div class="step-num">3</div>
                <span>Hold <strong style="color:#eef0f5">s
            till</strong> — readings begin in ~2 s</span>
            </div>
            <div class="step-item">
                <div class="step-num">4</div>
                <span>
            Signal quality <strong style="color:#eef0f5">&gt; 50 %</strong>
            for reliable data</span>
            </div>
        </div>
    </div>
    """)

    # Main content
    with gr.Row(equal_height=False, elem_classes=["gap"]):
        # Left — camera
        with gr.Column(scale=3, elem_classes=["webcam-block"]):
            webcam_input = gr.Image(
                sources=["webcam"],
                streaming=True,
                label="Camera Feed",
                height=420,
                show_label=True,
            )
            with gr.Row():
                reset_btn = gr.Button(
                    "↺  Reset Session",
                    variant="secondary",
                    elem_classes=["reset-btn"],
                )
                gr.HTML("""
                <p style="font-family:'DM Mono',monospace;font-size:11px;
                           color:#6b7491;align-self:center;margin:0;
                           letter-spacing:0.04em;">
                    ⚠ Bright, even lighting improves accuracy
                </p>""")

        # Right — metrics
        with gr.Column(scale=2):

            gr.HTML('<div class="readings-header">Live Readings</div>')

            bpm_output = gr.Textbox(
                label="Heart Rate",
                value="──",
                interactive=False,
                elem_classes=["metric-bpm"],
            )

            stress_output = gr.Textbox(
                label="Stress Level",
                value="──",
                interactive=False,
                elem_classes=["metric-stress"],
            )

            quality_output = gr.Slider(
                label="Signal Quality",
                minimum=0,
                maximum=100,
                value=0,
                interactive=False,
                elem_classes=["quality-slider"],
            )

            advice_output = gr.Textbox(
                label="Wellness Advice",
                value="Enable webcam to begin analysis.",
                interactive=False,
                lines=2,
                elem_classes=["advice-box"],
            )

            # Reference panel
            gr.HTML("""
            <div class="info-panel" style="margin-top:8px;">
                <h3>Reference Ranges</h3>
                <p>
                    <strong>BPM 60 – 100</strong>
                    &nbsp;Normal resting heart rate<br>
                    <strong>Stress LOW</strong>
                     &nbsp;&nbsp;&nbsp;&nbsp;RMSSD &gt; 40 ms<br>
                    <strong>Stress MEDIUM</strong> &nbsp;RMSSD 20 – 40 ms<br>
                    <strong>Stress HIGH</strong>
                     &nbsp;&nbsp;&nbsp;RMSSD &lt; 20 ms<br>
                    <strong>Quality &gt; 50 %</strong> &nbsp;Reliable reading
                </p>
            </div>
            """)

    # Hidden processed output
    processed_output = gr.Image(label="Processed Frame", visible=False)

    # Footer
    gr.HTML("""
    <div style="padding:20px 24px;
                font-family:'DM Mono',monospace;
                font-size:10px;
                color:#3a4060;
                letter-spacing:0.06em;
                border-top:1px solid #1e2535;
                margin-top:8px;
                display:flex;
                justify-content:space-between;
                align-items:center;">
        <span>rPPG · CHROM METHOD · 30 FPS</span>
        <span>For wellness purposes only — not a medical device</span>
    </div>
    """)

    # ── Wiring (unchanged) ────────────────────────────────────────────────
    webcam_input.stream(
        fn=process_frame,
        inputs=[webcam_input],
        outputs=[
            processed_output, bpm_output,
            stress_output, quality_output, advice_output,
        ],
    )

    reset_btn.click(
        fn=reset_buffers,
        outputs=[
            processed_output, bpm_output,
            stress_output, quality_output, advice_output,
        ],
    )


if __name__ == "__main__":
    print("=" * 50)
    print("Starting Heart Rate Monitor...")
    print("Make sure to allow webcam access when prompted.")
    print("Sit in a well-lit area and stay still for a few seconds.")
    print("=" * 50)
    demo.launch(
        share=True,
        server_port=7860,
        show_error=True,
        quiet=False,
    )
