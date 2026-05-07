"""
MedVis-X — Traceable Clinical Decision Support System
Main Gradio Web Application

Run: python app.py
Open browser: http://localhost:7860
"""
import os
import sys
import traceback
import threading
import numpy as np
from PIL import Image

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from pipeline.full_pipeline import MedVisXPipeline

# ── Lazy-load pipeline (only once, thread-safe) ──────────────────────────
_pipeline = None
_pipeline_lock = threading.Lock()


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:              # double-checked locking
                _pipeline = MedVisXPipeline()
    return _pipeline


def run_pipeline(prescription_image, clinical_text, use_text_only):
    """Main inference function called by Gradio."""
    try:
        pipe = get_pipeline()

        if use_text_only or prescription_image is None:
            if not clinical_text or not clinical_text.strip():
                return (
                    None, None, None,
                    "Please provide either an image or clinical text."
                )
            results = pipe.run(image_input=None, text_input=clinical_text)
        else:
            # Convert gradio image (numpy) to PIL for OCR
            if isinstance(prescription_image, np.ndarray):
                pil_img = Image.fromarray(
                    prescription_image.astype(np.uint8)
                )
            else:
                pil_img = prescription_image
            results = pipe.run(image_input=pil_img)

        # Format OCR + scoring summary
        scoring = results["scoring"]
        ocr_text = results["ocr"]["corrected_text"]

        hypotheses_text = "**Top Hypotheses:**\n"
        for disease, score in scoring["top_hypotheses"]:
            bar = "\u2588" * int(score * 20)
            hypotheses_text += f"\n- **{disease.title()}**: {score:.2f} {bar}"

        # Timings
        timings = results.get("timings", {})
        timing_str = " | ".join(
            [f"{k}: {v:.1f}s" for k, v in timings.items()]
        )

        summary = f"""## Clinical Analysis Report

**OCR Extracted Text:**
> {ocr_text}

---

{hypotheses_text}

---

**Top Diagnosis:** `{scoring['top_disease'].title()}`

**Confidence:** `{scoring['top_score']:.1%}`

**Imaging Modality:** `{scoring['modality'].replace('_', ' ').title()}`

**Generation Prompt:** *{results['prompt']}*

**Pipeline Timings:** {timing_str}
"""

        return (
            results["generated_image"],  # Generated scan
            results["heatmap"],          # Grad-CAM overlay
            results["shap_plot"],        # SHAP attribution plot
            summary + "\n---\n\n" + results["explanation"],
        )

    except Exception as e:
        err = traceback.format_exc()
        return None, None, None, f"Pipeline Error:\n```\n{err}\n```"


# ── Gradio UI ─────────────────────────────────────────────────────────────
CUSTOM_CSS = """
.gradio-container {
    font-family: 'Segoe UI', system-ui, sans-serif !important;
    max-width: 1400px !important;
}
.title-banner {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    text-align: center;
}
.title-banner h1 { font-size: 2.2rem; margin: 0; }
.title-banner p  { opacity: 0.8; margin: 0.5rem 0 0; }
.stage-label {
    font-weight: 600;
    color: #2c5364;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
"""


def build_demo():
    """Build and return the Gradio Blocks demo."""
    with gr.Blocks(title="MedVis-X Clinical AI") as demo:

        gr.HTML("""
        <div class="title-banner">
            <h1>MedVis-X</h1>
            <p>Traceable & Explainable Multimodal Clinical Decision Support System</p>
            <p style="font-size:0.8rem; opacity:0.6;">
                OCR &rarr; NER &rarr; Hypothesis Scoring &rarr;
                Diffusion Generation &rarr; Grad-CAM &rarr; Ollama XAI
            </p>
            <p style="font-size:0.75rem; opacity:0.4; margin-top:0.5rem;">
                100% Local &mdash; No Cloud APIs &mdash; No Internet at Inference
            </p>
        </div>
        """)

        with gr.Row():
            # ── LEFT: Inputs ──────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Input")

                prescription_img = gr.Image(
                    label="Upload Prescription / Clinical Document",
                    type="numpy",
                    height=280,
                )

                with gr.Accordion(
                    "Or enter clinical text directly", open=False
                ):
                    clinical_text = gr.Textbox(
                        label="Clinical Notes / Symptoms",
                        placeholder=(
                            "e.g. Patient presents with high fever, "
                            "productive cough, and crackles in lower "
                            "left lobe. SpO2 < 92%."
                        ),
                        lines=5,
                    )
                    use_text_only = gr.Checkbox(
                        label="Use text input only (skip OCR)",
                        value=False,
                    )

                run_btn = gr.Button(
                    "Run Full Pipeline",
                    variant="primary",
                    size="lg",
                )

                gr.Markdown("---")
                gr.Examples(
                    examples=[
                        [
                            None,
                            "Patient presents with high fever, productive "
                            "cough, and crackles in lower left lobe. "
                            "SpO2 < 92%.",
                            True,
                        ],
                        [
                            None,
                            "Severe chronic headaches, nausea, blurred "
                            "vision in right eye. Neurological exam "
                            "indicates localized pressure.",
                            True,
                        ],
                        [
                            None,
                            "Irregularly shaped mole on upper back, "
                            "asymmetric, multiple color variations, "
                            "diameter > 6mm.",
                            True,
                        ],
                        [
                            None,
                            "Type 2 diabetic patient. HbA1c 8.5%. "
                            "Floaters and dark spots in vision.",
                            True,
                        ],
                        [
                            None,
                            "Patient complains of wheezing, chest "
                            "tightness, and nocturnal cough. Has "
                            "allergic history.",
                            True,
                        ],
                    ],
                    inputs=[prescription_img, clinical_text, use_text_only],
                    label="Example Patient Cases",
                )

            # ── RIGHT: Outputs ────────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### Analysis Outputs")

                with gr.Tabs():
                    with gr.Tab("Generated Scan + Heatmap"):
                        with gr.Row():
                            generated_out = gr.Image(
                                label="Stage 4: Generated Medical Image",
                                height=350,
                            )
                            heatmap_out = gr.Image(
                                label="Stage 5: Grad-CAM Activation Map",
                                height=350,
                            )

                    with gr.Tab("SHAP Attribution"):
                        shap_out = gr.Image(
                            label="Stage 5: SHAP Feature Attribution",
                            height=420,
                        )

                    with gr.Tab("Report + Explanation"):
                        report_out = gr.Markdown(
                            value=(
                                "*Run the pipeline to see the clinical "
                                "report here.*"
                            ),
                        )

        # ── Bind ──────────────────────────────────────────────────────
        run_btn.click(
            fn=run_pipeline,
            inputs=[prescription_img, clinical_text, use_text_only],
            outputs=[generated_out, heatmap_out, shap_out, report_out],
        )

        gr.HTML("""
        <div style="text-align:center; padding:1rem; opacity:0.5;
                    font-size:0.8rem;">
            MedVis-X is a research decision-support tool only.
            All outputs must be reviewed and verified by a qualified
            clinician. Generated images are synthetic and NOT real
            patient data.
        </div>
        """)

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue(max_size=1)          # only 1 job at a time (8 GB VRAM limit)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
    )
