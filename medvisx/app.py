"""
MedVis-X — Professional Clinical Decision Support Interface
============================================================
A clean 3-step workflow:  OCR → Generation → XAI
Built with Gradio · 100% Local · No Cloud APIs

Run: python app.py
"""
import os
import sys
import time
import traceback
import threading
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from pipeline.simple_pipeline import MedVisXPipeline

# ── Thread-safe pipeline singleton ────────────────────────────────────────
_pipeline = None
_lock = threading.Lock()


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                _pipeline = MedVisXPipeline()
    return _pipeline


# ══════════════════════════════════════════════════════════════════════════
#  CALLBACK FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def run_full_analysis(image, text, progress=gr.Progress()):
    """Run all 3 models and return all outputs at once."""
    try:
        pipe = get_pipeline()

        use_text = text and text.strip()
        if not use_text and image is None:
            empty = None
            return (
                "⚠️ Please upload an image or enter clinical text.",  # ocr_md
                empty, empty, empty,          # gen_img, heatmap, shap
                "",                            # xai_md
                gr.update(visible=False),      # results_row
            )

        progress(0.05, desc="Running OCR...")

        # Step 1: OCR
        if use_text:
            ocr_result = pipe.run_ocr(text_input=text)
        else:
            pil = Image.fromarray(image.astype(np.uint8)) if isinstance(image, np.ndarray) else image
            ocr_result = pipe.run_ocr(image_input=pil)

        scoring = ocr_result["scoring"]
        entities = ocr_result["entities"]

        # Build OCR summary markdown
        ocr_md = _build_ocr_markdown(ocr_result, scoring, entities)

        progress(0.30, desc="Generating medical image...")

        # Step 2: Generation
        gen_result = pipe.run_generation(scoring)
        gen_image = gen_result["image"]

        progress(0.65, desc="Running XAI analysis...")

        # Step 3: XAI
        xai_result = pipe.run_xai(gen_image, ocr_result)

        # Build XAI explanation markdown
        xai_md = _build_xai_markdown(xai_result, scoring)

        progress(1.0, desc="Done!")

        return (
            ocr_md,
            gen_image,
            xai_result["heatmap"],
            xai_result["shap_plot"],
            xai_md,
            gr.update(visible=True),
        )

    except Exception as e:
        err = traceback.format_exc()
        return (
            f"**Error:** {e}",
            None, None, None,
            f"```\n{err}\n```",
            gr.update(visible=True),
        )


def _build_ocr_markdown(ocr_result, scoring, entities):
    """Build rich OCR results markdown."""
    method = ocr_result.get("method_used", "unknown")
    method_label = {
        "trained_crnn": "🔬 Trained CRNN Model",
        "easyocr": "📖 EasyOCR Engine",
        "manual_input": "⌨️ Manual Text Input",
    }.get(method, method)

    crnn_text = ocr_result.get("crnn_text", "")
    easy_text = ocr_result.get("easyocr_text", "")
    final_text = ocr_result.get("final_text", "")

    # Entity summary
    symptoms = [e["text"] for e in entities.get("symptoms", [])]
    medications = [e["text"] for e in entities.get("medications", [])]
    conditions = [e["text"] for e in entities.get("conditions", [])]

    # Top hypotheses
    hyp_lines = ""
    for disease, score in scoring.get("top_hypotheses", []):
        bar = "█" * int(score * 25) + "░" * (25 - int(score * 25))
        hyp_lines += f"| **{disease.title()}** | {bar} | `{score:.0%}` |\n"

    md = f"""### 📋 Extracted Text
> {final_text}

**Method Used:** {method_label} &nbsp;·&nbsp; **Processing Time:** `{ocr_result.get('time', 0):.2f}s`

"""

    # Show both models if available
    if crnn_text or easy_text:
        md += """<details><summary>🔍 <b>Model Comparison</b> (click to expand)</summary>\n\n"""
        if crnn_text:
            md += f"**Trained CRNN Output:** `{crnn_text}`\n\n"
        if easy_text:
            md += f"**EasyOCR Output:** `{easy_text}`\n\n"
        md += "</details>\n\n"

    md += f"""### 🏥 Detected Clinical Entities
| Category | Found |
|----------|-------|
| **Symptoms** | {', '.join(symptoms) if symptoms else '*None detected*'} |
| **Medications** | {', '.join(medications) if medications else '*None detected*'} |
| **Conditions** | {', '.join(conditions) if conditions else '*None detected*'} |

### 📊 Diagnostic Hypotheses
| Condition | Confidence | Score |
|-----------|------------|-------|
{hyp_lines}
**Top Diagnosis:** `{scoring.get('top_disease', 'N/A').title()}` &nbsp;·&nbsp; **Modality:** `{scoring.get('modality', 'N/A').replace('_', ' ').title()}`
"""
    return md


def _build_xai_markdown(xai_result, scoring):
    """Build XAI explanation markdown."""
    explanation = xai_result.get("explanation", "")
    xai_time = xai_result.get("time", 0)

    md = f"""### 🔬 Clinical Explanation (LLaVA-1.5-7B via Ollama)

**Analysis Time:** `{xai_time:.1f}s`

---

{explanation}
"""
    return md


# ══════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS — Professional Medical Theme
# ══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
/* ── Global ─────────────────────────────────────────── */
.gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    max-width: 1440px !important;
    margin: 0 auto !important;
}

/* ── Header ─────────────────────────────────────────── */
.header-container {
    background: linear-gradient(135deg, #0a1628 0%, #1a365d 50%, #2b6cb0 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 40px rgba(10, 22, 40, 0.3);
}
.header-container::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(66, 153, 225, 0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.header-container h1 {
    color: #fff;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}
.header-container .subtitle {
    color: rgba(255,255,255,0.8);
    font-size: 1.05rem;
    margin: 0.5rem 0 0;
    font-weight: 400;
}
.header-container .badge-row {
    display: flex;
    gap: 0.75rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.header-container .badge {
    background: rgba(255,255,255,0.12);
    color: rgba(255,255,255,0.9);
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
}

/* ── Model cards in header ──────────────────────────── */
.models-strip {
    display: flex;
    gap: 1rem;
    margin-top: 1.25rem;
}
.model-chip {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    flex: 1;
    text-align: center;
}
.model-chip .chip-num {
    display: inline-block;
    background: rgba(66, 153, 225, 0.4);
    color: #fff;
    font-weight: 700;
    width: 24px;
    height: 24px;
    line-height: 24px;
    border-radius: 50%;
    font-size: 0.75rem;
    margin-right: 0.4rem;
}
.model-chip .chip-label {
    color: #fff;
    font-weight: 600;
    font-size: 0.85rem;
}
.model-chip .chip-desc {
    color: rgba(255,255,255,0.55);
    font-size: 0.72rem;
    display: block;
    margin-top: 0.2rem;
}

/* ── Section cards ──────────────────────────────────── */
.section-card {
    background: var(--background-fill-primary);
    border: 1px solid var(--border-color-primary);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s;
}
.section-card:hover {
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin: 0 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title .step-badge {
    background: linear-gradient(135deg, #2b6cb0, #4299e1);
    color: white;
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Input area refinements ─────────────────────────── */
.input-panel {
    background: var(--background-fill-primary);
    border: 1px solid var(--border-color-primary);
    border-radius: 14px;
    padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
}

/* ── Run button ─────────────────────────────────────── */
.run-btn {
    background: linear-gradient(135deg, #2b6cb0 0%, #3182ce 50%, #4299e1 100%) !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 0.9rem 2rem !important;
    border-radius: 12px !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 15px rgba(43, 108, 176, 0.35) !important;
    transition: all 0.25s !important;
    text-transform: uppercase !important;
}
.run-btn:hover {
    box-shadow: 0 6px 25px rgba(43, 108, 176, 0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Image outputs ──────────────────────────────────── */
.output-image {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Footer ─────────────────────────────────────────── */
.footer-note {
    text-align: center;
    padding: 1.5rem 1rem;
    color: var(--body-text-color-subdued);
    font-size: 0.78rem;
    border-top: 1px solid var(--border-color-primary);
    margin-top: 1.5rem;
}

/* ── Example buttons ────────────────────────────────── */
.examples-row button {
    border-radius: 10px !important;
    font-size: 0.82rem !important;
}
"""


# ══════════════════════════════════════════════════════════════════════════
#  BUILD GRADIO INTERFACE
# ══════════════════════════════════════════════════════════════════════════

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)


def build_app():
    with gr.Blocks(title="MedVis-X · Clinical AI") as demo:

        # ── Header ────────────────────────────────────────────────────
        gr.HTML("""
        <div class="header-container">
            <h1>⚕️ MedVis-X</h1>
            <p class="subtitle">
                Traceable &amp; Explainable Multimodal Clinical Decision Support System
            </p>
            <div class="badge-row">
                <span class="badge">100% Local Inference</span>
                <span class="badge">No Cloud APIs</span>
                <span class="badge">CUDA Accelerated</span>
                <span class="badge">Research Tool</span>
            </div>
            <div class="models-strip">
                <div class="model-chip">
                    <span class="chip-num">1</span>
                    <span class="chip-label">OCR</span>
                    <span class="chip-desc">Trained CRNN + EasyOCR</span>
                </div>
                <div class="model-chip">
                    <span class="chip-num">2</span>
                    <span class="chip-label">Generation</span>
                    <span class="chip-desc">Stable Diffusion + LoRA</span>
                </div>
                <div class="model-chip">
                    <span class="chip-num">3</span>
                    <span class="chip-label">XAI</span>
                    <span class="chip-desc">Grad-CAM + LLaVA 7B</span>
                </div>
            </div>
        </div>
        """)

        # ── Main Layout ───────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # ── LEFT COLUMN: Input ────────────────────────────────────
            with gr.Column(scale=1, min_width=380):
                gr.HTML("""
                <div class="section-title">
                    <span class="step-badge">Input</span>
                    Clinical Data Entry
                </div>
                """)

                prescription_img = gr.Image(
                    label="Upload Prescription / Medical Document",
                    type="numpy",
                    height=260,
                    sources=["upload", "clipboard"],
                )

                clinical_text = gr.Textbox(
                    label="Or Enter Clinical Notes Directly",
                    placeholder="e.g. Patient presents with high fever, productive cough, crackles in lower left lobe. SpO2 < 92%...",
                    lines=4,
                    max_lines=8,
                )

                gr.HTML('<p style="font-size:0.8rem; opacity:0.6; margin:0.5rem 0;">Upload an image <b>or</b> type clinical text. If both are provided, text takes priority.</p>')

                run_btn = gr.Button(
                    "▶  Run Full Analysis",
                    variant="primary",
                    size="lg",
                    elem_classes=["run-btn"],
                )

                gr.HTML("<hr style='opacity:0.2; margin: 1rem 0;'>")

                # Example cases
                gr.Examples(
                    examples=[
                        [None, "Patient presents with high fever, productive cough, and crackles in lower left lobe. SpO2 < 92%."],
                        [None, "Severe chronic headaches, nausea, blurred vision. Neurological exam indicates localized intracranial pressure."],
                        [None, "Irregularly shaped mole on upper back, asymmetric borders, multiple color variations, diameter > 6mm."],
                        [None, "Type 2 diabetic patient with HbA1c 8.5%. Floaters and dark spots in vision. Possible diabetic retinopathy."],
                        [None, "Patient complains of wheezing, chest tightness, and nocturnal cough. Allergic history present."],
                        [None, "Persistent right flank pain, hematuria, CT findings show 8mm calculus in proximal ureter."],
                    ],
                    inputs=[prescription_img, clinical_text],
                    label="📝 Example Patient Cases",
                    examples_per_page=6,
                )

            # ── RIGHT COLUMN: Results ─────────────────────────────────
            with gr.Column(scale=2, min_width=600):

                # Results container (initially hidden)
                results_row = gr.Column(visible=False)

                with results_row:
                    # ── Step 1 Results: OCR ────────────────────────────
                    gr.HTML("""
                    <div class="section-title">
                        <span class="step-badge">Model 1</span>
                        OCR Results &amp; Clinical Analysis
                    </div>
                    """)
                    ocr_output = gr.Markdown(
                        value="*Waiting for analysis...*",
                    )

                    gr.HTML("<hr style='opacity:0.15; margin: 0.5rem 0;'>")

                    # ── Step 2 Results: Generation ─────────────────────
                    gr.HTML("""
                    <div class="section-title">
                        <span class="step-badge">Model 2</span>
                        Medical Image Generation
                    </div>
                    """)
                    generated_img = gr.Image(
                        label="Generated Medical Image (SD v1.5 + LoRA)",
                        height=380,
                        elem_classes=["output-image"],
                    )

                    gr.HTML("<hr style='opacity:0.15; margin: 0.5rem 0;'>")

                    # ── Step 3 Results: XAI ────────────────────────────
                    gr.HTML("""
                    <div class="section-title">
                        <span class="step-badge">Model 3</span>
                        Explainability &amp; XAI Analysis
                    </div>
                    """)
                    with gr.Tabs():
                        with gr.Tab("🔥 Grad-CAM Heatmap"):
                            heatmap_img = gr.Image(
                                label="Grad-CAM Activation Overlay",
                                height=380,
                                elem_classes=["output-image"],
                            )
                        with gr.Tab("📊 SHAP Attribution"):
                            shap_img = gr.Image(
                                label="SHAP Feature Attribution Plot",
                                height=380,
                                elem_classes=["output-image"],
                            )
                        with gr.Tab("📝 LLaVA Explanation"):
                            xai_output = gr.Markdown(
                                value="*Waiting for analysis...*",
                            )

                # Welcome message when no results yet
                welcome_msg = gr.HTML("""
                <div style="text-align:center; padding: 4rem 2rem; opacity: 0.5;">
                    <p style="font-size: 3rem; margin: 0;">⚕️</p>
                    <h3 style="margin: 1rem 0 0.5rem;">Ready for Analysis</h3>
                    <p style="font-size: 0.9rem;">
                        Upload a medical image or enter clinical text, then click
                        <strong>Run Full Analysis</strong> to begin.
                    </p>
                    <p style="font-size: 0.8rem; margin-top: 1rem; opacity: 0.7;">
                        The pipeline will run 3 models: OCR → Image Generation → XAI Explanation
                    </p>
                </div>
                """, visible=True)

        # ── Footer ────────────────────────────────────────────────────
        gr.HTML("""
        <div class="footer-note">
            <strong>⚠️ Research Tool Only</strong> — MedVis-X is a decision-support prototype.
            All outputs must be verified by a qualified clinician.
            Generated images are synthetic and <strong>NOT</strong> real patient data.<br>
            <span style="opacity:0.5;">Built with PyTorch · Stable Diffusion · Ollama LLaVA · Gradio</span>
        </div>
        """)

        # ── Bind events ───────────────────────────────────────────────
        run_btn.click(
            fn=run_full_analysis,
            inputs=[prescription_img, clinical_text],
            outputs=[
                ocr_output,
                generated_img,
                heatmap_img,
                shap_img,
                xai_output,
                results_row,
            ],
        ).then(
            fn=lambda: gr.update(visible=False),
            inputs=None,
            outputs=welcome_msg,
        )

    return demo


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = build_app()
    app.queue(max_size=2)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=THEME,
    )
