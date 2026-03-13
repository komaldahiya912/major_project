"""
app.py — Brain Tumour Grading System v2.0
Dual Pipeline: MRI (Grade 1/2) + Clinical features (LGG/GBM)
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, io, math
from datetime import datetime
import pandas as pd
from skimage.transform import resize

from model_loader import BrainTumorPredictor, VQC2Predictor
from database    import PredictionDatabase

# Optional PDF
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Image as RLImage, Table, TableStyle)
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    _PDF = True
except ImportError:
    _PDF = False


# ════════════════════════════════════════════════════════════════════
#  CACHED LOADERS — loaded once per session, not per rerun
# ════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_p1():
    return BrainTumorPredictor()

@st.cache_resource
def load_p2():
    return VQC2Predictor()

@st.cache_resource
def load_db():
    return PredictionDatabase()


# ════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════
def make_overlay(orig_arr, mask, alpha=0.45):
    """Greyscale MRI with red tumour overlay. Returns PNG BytesIO."""
    if orig_arr.shape != mask.shape:
        mask = resize(mask, orig_arr.shape, preserve_range=True, anti_aliasing=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(orig_arr, cmap="gray", vmin=0, vmax=255)
    ov = np.zeros((*mask.shape, 4))
    ov[mask > 0.5, 0] = 1.0
    ov[mask > 0.5, 3] = alpha
    ax.imshow(ov, interpolation="nearest")
    patch = mpatches.Patch(color=(1,0,0,0.6), label="Tumour")
    ax.legend(handles=[patch], loc="lower right", fontsize=8)
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=130, facecolor="white")
    buf.seek(0); plt.close()
    return buf


def prob_bar(lgg_p, gbm_p):
    """Horizontal probability bar chart."""
    fig, ax = plt.subplots(figsize=(4.5, 1.1))
    ax.barh(["LGG", "GBM"], [lgg_p, gbm_p],
            color=["#22C55E", "#EF4444"], height=0.55)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="grey", ls="--", lw=0.8)
    for i, v in enumerate([lgg_p, gbm_p]):
        ax.text(v + 0.01, i, f"{v*100:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("Probability", fontsize=8)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0); plt.close()
    return buf


# ── PDF builders ─────────────────────────────────────────────────────
def pdf_mri(patient, results, orig_arr, mask):
    if not _PDF: return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    ss  = getSampleStyleSheet()
    S   = []
    S.append(Paragraph("<b>Brain Tumour MRI Analysis Report</b>", ss["Title"]))
    S.append(Spacer(1,12))
    S.append(Paragraph(f"<b>Patient:</b> {patient}", ss["Normal"]))
    S.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", ss["Normal"]))
    S.append(Paragraph("<b>Pipeline:</b> 1 — ResNet50-UNet + 4-qubit Quantum VQC", ss["Normal"]))
    S.append(Spacer(1,10))
    S.append(Paragraph("<b>Detection</b>", ss["Heading2"]))
    S.append(Paragraph(f"Tumour detected: {'Yes' if results['tumor_present'] else 'No'}", ss["Normal"]))
    S.append(Paragraph(f"Tumour area: {results['tumor_area']:.0f} pixels", ss["Normal"]))
    S.append(Spacer(1,8))
    S.append(Paragraph("<b>Grading</b>", ss["Heading2"]))
    S.append(Paragraph(f"Predicted grade: Grade {results['predicted_grade']}", ss["Normal"]))
    S.append(Paragraph(f"Confidence: {results['grade_confidence']:.4f}", ss["Normal"]))
    S.append(Spacer(1,8))
    S.append(Paragraph("<b>Segmentation Statistics</b>", ss["Heading2"]))
    for k, v in results["segmentation_stats"].items():
        S.append(Paragraph(f"  {k}: {v:.4f}", ss["Normal"]))
    S.append(Spacer(1,12))
    ov = make_overlay(orig_arr, mask)
    S.append(RLImage(ov, width=300, height=300))
    S.append(Spacer(1,12))
    S.append(Paragraph(
        "<i>⚠ Research system only — not for clinical diagnosis.</i>", ss["Normal"]
    ))
    doc.build(S); buf.seek(0)
    return buf


def pdf_clinical(patient, features, result):
    if not _PDF: return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    ss  = getSampleStyleSheet()
    S   = []
    S.append(Paragraph("<b>Brain Tumour Clinical Classification Report</b>", ss["Title"]))
    S.append(Spacer(1,12))
    S.append(Paragraph(f"<b>Patient:</b> {patient}", ss["Normal"]))
    S.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", ss["Normal"]))
    S.append(Paragraph("<b>Pipeline:</b> 2 — VQC-2 Quantum Classifier (5-qubit)", ss["Normal"]))
    S.append(Spacer(1,10))
    S.append(Paragraph("<b>Input Features</b>", ss["Heading2"]))
    feat_rows = [
        ["Feature", "Value"],
        ["IDH1 Mutation",    "Mutated (1)" if features["idh1"] else "Not Mutated (0)"],
        ["Age at Diagnosis", f"{features['age']:.1f} years"],
        ["PTEN Mutation",    "Mutated (1)" if features["pten"] else "Not Mutated (0)"],
        ["EGFR Mutation",    "Mutated (1)" if features["egfr"] else "Not Mutated (0)"],
        ["ATRX Mutation",    "Mutated (1)" if features["atrx"] else "Not Mutated (0)"],
    ]
    t = Table(feat_rows, colWidths=[200, 200])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#1E3A5F")),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#F8FAFC"), colors.white]),
        ("GRID",          (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTSIZE",      (0,0), (-1,-1), 10),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    S.append(t); S.append(Spacer(1,10))
    S.append(Paragraph("<b>Result</b>", ss["Heading2"]))
    S.append(Paragraph(
        f"<b>Predicted class: {result['predicted_class']}</b> — "
        f"{'Glioblastoma Multiforme (Grade IV)' if result['predicted_class']=='GBM' else 'Low Grade Glioma'}",
        ss["Normal"]
    ))
    S.append(Paragraph(f"Confidence: {result['confidence']*100:.1f}%", ss["Normal"]))
    S.append(Paragraph(f"LGG probability: {result['lgg_probability']*100:.1f}%", ss["Normal"]))
    S.append(Paragraph(f"GBM probability: {result['gbm_probability']*100:.1f}%", ss["Normal"]))
    S.append(Paragraph(f"Raw quantum output (PauliZ): {result['raw_output']:.4f}", ss["Normal"]))
    S.append(Spacer(1,10))
    S.append(Paragraph("<b>Model</b>", ss["Heading2"]))
    S.append(Paragraph("VQC-2  |  5 qubits  |  2 layers  |  RyRz+CZ / Ry+CNOT", ss["Normal"]))
    S.append(Paragraph("Accuracy: 84.81%  |  Recall: 93.67%  (862 TCGA patients, 5-fold CV)", ss["Normal"]))
    S.append(Paragraph(
        "Akpinar & Oduncuoglu (2025), Nature Scientific Reports, DOI: 10.1038/s41598-025-97067-3",
        ss["Normal"]
    ))
    S.append(Spacer(1,12))
    S.append(Paragraph(
        "<i>⚠ Research system only — not for clinical diagnosis.</i>", ss["Normal"]
    ))
    doc.build(S); buf.seek(0)
    return buf


# ════════════════════════════════════════════════════════════════════
#  PAGE FUNCTIONS
# ════════════════════════════════════════════════════════════════════
def page_mode1(db):
    st.title("🧠 Mode 1 — MRI Scan Analysis")
    st.caption("ResNet50-UNet segmentation · 4-qubit VQC · LGG Grade 1 vs Grade 2")
    st.info(
        "**Scope:** This mode grades LGG sub-types (Grade 1 vs Grade 2) from MRI images. "
        "For LGG vs GBM classification using genomic mutation data, use **Mode 2**."
    )
    st.markdown("---")

    patient_name = st.text_input("Patient Name *", placeholder="e.g. Ramesh Kumar")
    uploaded = st.file_uploader(
        "Upload Brain MRI Scan *",
        type=["png", "jpg", "jpeg", "tiff", "tif"],
    )

    if not patient_name:
        st.warning("Enter patient name to proceed.")
        return
    if not uploaded:
        st.info("Upload an MRI scan to continue.")
        return

    # Save upload
    os.makedirs("static/uploads", exist_ok=True)
    img_path = os.path.join("static", "uploads", uploaded.name)
    with open(img_path, "wb") as f:
        f.write(uploaded.getbuffer())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original MRI")
        orig_img = Image.open(img_path).convert("L")
        st.image(orig_img, use_column_width=True)

    if not st.button("🔬 Analyse MRI", type="primary", use_container_width=True):
        return

    with st.spinner("Running segmentation + quantum grading … (~30-60 s on CPU)"):
        try:
            predictor = load_p1()
            results   = predictor.predict(img_path)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.exception(e)
            return

    orig_arr = np.array(orig_img)
    with col2:
        st.subheader("Tumour Overlay")
        st.image(make_overlay(orig_arr, results["tumor_mask"]),
                 use_column_width=True)

    st.markdown("---")
    st.subheader("Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tumour Detected",  "Yes ✓" if results["tumor_present"] else "No")
    m2.metric("Predicted Grade",  f"Grade {results['predicted_grade']}")
    m3.metric("Confidence",       f"{results['grade_confidence']:.3f}")
    m4.metric("Tumour Area",      f"{results['tumor_area']:.0f} px")

    with st.expander("Segmentation Details"):
        seg = results["segmentation_stats"]
        s1,s2,s3,s4 = st.columns(4)
        s1.metric("Mean Prob",    f"{seg['mean_prob']:.4f}")
        s2.metric("Std Prob",     f"{seg['std_prob']:.4f}")
        s3.metric("Max Prob",     f"{seg['max_prob']:.4f}")
        s4.metric("Tumour Ratio", f"{seg['tumor_ratio']:.4f}")

    st.markdown("---")
    # Save to DB
    try:
        pid = db.save_prediction(patient_name, img_path, results)
        st.success(f"Saved to database (ID: {pid})")
    except Exception as e:
        st.error(f"Database save failed: {e}")

    # PDF download
    if _PDF:
        buf = pdf_mri(patient_name, results, orig_arr, results["tumor_mask"])
        if buf:
            st.download_button(
                "⬇ Download PDF Report", data=buf.getvalue(),
                file_name=f"{patient_name.replace(' ','_')}_MRI_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
            )

    st.warning(
        "⚠ **Expected accuracy ~54%** — WHO 2021 states LGG sub-grades differ "
        "molecularly, not visually. MRI texture features cannot reliably separate "
        "Grade 1 from Grade 2. This is a documented clinical finding. "
        "Use Mode 2 for stronger LGG-vs-GBM classification."
    )


def page_mode2(db):
    st.title("🧬 Mode 2 — Clinical Feature Classification")
    st.caption(
        "VQC-2 Quantum Classifier · 5 qubits · "
        "84.81% accuracy · 93.67% recall · 862 TCGA patients"
    )
    st.markdown(
        "> Replicates **Akpinar & Oduncuoglu (2025)**, *Nature Scientific Reports*, "
        "DOI: 10.1038/s41598-025-97067-3. "
        "Our VQC-2 exceeds the paper's result by **+10.81 pp** (84.81% vs 74%)."
    )
    st.info(
        "Enter genomic mutation status from biopsy or TCGA report. "
        "Classifies **LGG** (Low Grade Glioma) vs **GBM** (Glioblastoma Multiforme)."
    )
    st.markdown("---")

    # ── Input form ──────────────────────────────────────────────────
    st.subheader("Patient & Mutation Data")
    c1, c2 = st.columns(2)
    with c1:
        patient_name = st.text_input("Patient Name *", placeholder="e.g. Priya Sharma",
                                     key="p2_name")
        idh1 = st.selectbox(
            "IDH1 Mutation *", [0, 1],
            format_func=lambda v: "Not Mutated (0)" if v == 0 else "Mutated (1)",
            help="Strongest predictor — mutated in ~80% of LGG, ~5% of GBM",
        )
        pten = st.selectbox(
            "PTEN Mutation *", [0, 1],
            format_func=lambda v: "Not Mutated (0)" if v == 0 else "Mutated (1)",
            help="PTEN loss common in GBM (~40%)",
        )
    with c2:
        age = st.number_input(
            "Age at Diagnosis * (years)",
            min_value=0.0, max_value=120.0, value=45.0, step=0.5,
            help="LGG median ~38 yrs · GBM median ~60 yrs",
        )
        egfr = st.selectbox(
            "EGFR Mutation *", [0, 1],
            format_func=lambda v: "Not Mutated (0)" if v == 0 else "Mutated (1)",
            help="EGFR amplification is a GBM hallmark (~40-50%)",
        )
        atrx = st.selectbox(
            "ATRX Mutation *", [0, 1],
            format_func=lambda v: "Not Mutated (0)" if v == 0 else "Mutated (1)",
            help="ATRX mutation is characteristic of LGG astrocytoma",
        )

    st.markdown("---")
    if not patient_name:
        st.warning("Enter patient name to enable classification.")
        return

    if not st.button("⚛ Classify Tumour", type="primary", use_container_width=True):
        return

    with st.spinner("Running VQC-2 quantum classification … (~1-3 s)"):
        try:
            predictor = load_p2()
            result    = predictor.predict(idh1, age, pten, egfr, atrx)
        except Exception as e:
            st.error(f"Classification failed: {e}")
            st.exception(e)
            return

    # ── Result display ───────────────────────────────────────────────
    st.markdown("---")
    cls  = result["predicted_class"]
    conf = result["confidence"]

    if cls == "GBM":
        st.markdown(
            f'<div style="background:#FEF2F2;border:2px solid #EF4444;'
            f'border-radius:10px;padding:20px;text-align:center;">'
            f'<div style="font-size:1.8rem;font-weight:700;color:#DC2626">🔴 GBM Detected</div>'
            f'<div style="color:#555;margin-top:4px">Glioblastoma Multiforme — Grade IV</div>'
            f'<div style="font-size:1.2rem;margin-top:10px">Confidence: <b>{conf*100:.1f}%</b></div>'
            f'</div>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="background:#DCFCE7;border:2px solid #22C55E;'
            f'border-radius:10px;padding:20px;text-align:center;">'
            f'<div style="font-size:1.8rem;font-weight:700;color:#15803D">🟢 LGG Detected</div>'
            f'<div style="color:#555;margin-top:4px">Low Grade Glioma</div>'
            f'<div style="font-size:1.2rem;margin-top:10px">Confidence: <b>{conf*100:.1f}%</b></div>'
            f'</div>', unsafe_allow_html=True
        )

    if 0.45 < conf < 0.65:
        st.warning(
            "⚠ Confidence is close to 50% — this result is uncertain. "
            "Consider additional molecular testing."
        )

    st.markdown("")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Prediction",      cls)
    m2.metric("Confidence",      f"{conf*100:.1f}%")
    m3.metric("LGG Probability", f"{result['lgg_probability']*100:.1f}%")
    m4.metric("GBM Probability", f"{result['gbm_probability']*100:.1f}%")

    st.image(prob_bar(result["lgg_probability"], result["gbm_probability"]),
             width=420)

    # Feature interpretation
    with st.expander("🔬 Feature Interpretation"):
        interp = {
            "IDH1":  ("Mutated — strongly favours LGG (IDH mutation in 80% of LGG vs 5% of GBM)"
                       if idh1 else
                       "Not Mutated (wild-type) — strongly favours GBM (IDH wild-type = primary GBM hallmark)"),
            "Age":   (f"{age:.1f} yrs — younger age associated with LGG (median ~38 yrs)"
                       if age < 45 else
                       f"{age:.1f} yrs — older age associated with GBM (median ~60 yrs)"),
            "PTEN":  ("Mutated — favours GBM (PTEN loss in ~40% of GBM)"
                       if pten else "Not Mutated — no strong signal"),
            "EGFR":  ("Mutated — favours GBM (EGFR amplification in 40-50% of GBM)"
                       if egfr else "Not Mutated — favours LGG"),
            "ATRX":  ("Mutated — favours LGG astrocytoma (ATRX + IDH1 = classic LGG combo)"
                       if atrx else "Not Mutated — neutral to GBM signal"),
        }
        for feat, txt in interp.items():
            st.markdown(f"**{feat}:** {txt}")

    # Save to DB
    features_dict = {"idh1": idh1, "age": age, "pten": pten,
                     "egfr": egfr, "atrx": atrx}
    try:
        pid = db.save_clinical_prediction(patient_name, features_dict, result)
        st.success(f"Saved to database (ID: {pid})")
    except Exception as e:
        st.error(f"Database save failed: {e}")

    # PDF download
    if _PDF:
        buf = pdf_clinical(patient_name, features_dict, result)
        if buf:
            st.download_button(
                "⬇ Download PDF Report", data=buf.getvalue(),
                file_name=f"{patient_name.replace(' ','_')}_Clinical_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
            )

    st.warning(
        "⚠ Disclaimer: Research system only. Not for clinical diagnosis. "
        "Always consult a qualified medical professional."
    )


def page_history(db):
    st.title("📋 Prediction History")
    tab1, tab2 = st.tabs(["🧠 MRI Predictions (Pipeline 1)",
                           "🧬 Clinical Predictions (Pipeline 2)"])

    with tab1:
        preds = db.get_all_predictions()
        if preds:
            df = pd.DataFrame([{
                "ID":p[0],"Patient":p[1],"Date":p[2],
                "Tumour":"Yes" if p[4] else "No",
                "Grade":f"Grade {p[5]}","Confidence":f"{p[6]:.3f}",
                "Area(px)":f"{p[7]:.0f}",
            } for p in preds])
            st.dataframe(df, hide_index=True, use_container_width=True)
            s1,s2,s3 = st.columns(3)
            tc = sum(1 for p in preds if p[4])
            s1.metric("Total Scans",     len(preds))
            s2.metric("Tumours Detected",tc)
            s3.metric("No Tumour",       len(preds)-tc)
            if tc:
                gc = {}
                for p in preds:
                    if p[4]: gc[f"Grade {p[5]}"]=gc.get(f"Grade {p[5]}",0)+1
                st.bar_chart(pd.DataFrame(
                    list(gc.items()), columns=["Grade","Count"]
                ).set_index("Grade"))
        else:
            st.info("No MRI predictions yet — use Mode 1.")

    with tab2:
        cpreds = db.get_all_clinical_predictions()
        if cpreds:
            df2 = pd.DataFrame([{
                "ID":c[0],"Patient":c[1],"Date":c[2],
                "IDH1":c[3],"Age":f"{c[4]:.1f}","PTEN":c[5],
                "EGFR":c[6],"ATRX":c[7],"Prediction":c[8],
                "Confidence":f"{c[9]*100:.1f}%",
            } for c in cpreds])
            st.dataframe(df2, hide_index=True, use_container_width=True)
            lgg_c = sum(1 for c in cpreds if c[8]=="LGG")
            gbm_c = sum(1 for c in cpreds if c[8]=="GBM")
            a1,a2,a3 = st.columns(3)
            a1.metric("Total Clinical", len(cpreds))
            a2.metric("LGG Predicted",  lgg_c)
            a3.metric("GBM Predicted",  gbm_c)
            st.bar_chart(pd.DataFrame(
                {"Class":["LGG","GBM"],"Count":[lgg_c,gbm_c]}
            ).set_index("Class"))
        else:
            st.info("No clinical predictions yet — use Mode 2.")


def page_about():
    st.title("ℹ About This System")
    c1,c2 = st.columns([3,2])

    with c1:
        st.markdown("""
## Brain Tumour Grading System v2.0
Dual-pipeline AI combining deep learning and quantum machine learning.

---

### Pipeline 1 — MRI Image Analysis
| | |
|---|---|
| Segmentation | ResNet50-UNet + Attention Gates |
| Input | Greyscale MRI (512×512) |
| Segmentation | Dice: 85.71% · IoU: 82.30% · Pixel acc: 99.61% |
| Classifier | 4-qubit VQC, 3 layers, 24 parameters |
| Accuracy | ~54% (see note below) |
| Task | LGG Grade 1 vs Grade 2 |
| Dataset | 110 LGG patients · 3,929 MRI scans |

> ⚠ **Why 54%?** WHO 2021 classification: LGG sub-grades differ at the
> *molecular* level, not histologically. MRI texture alone cannot reliably
> separate Grade 1 from Grade 2. This is a documented clinical finding — not
> a model failure. Pipeline 2 addresses this with clinical genomic features.

---

### Pipeline 2 — Clinical Feature Classification
| | |
|---|---|
| Model | VQC-2 Variational Quantum Classifier |
| Qubits | 5 · Layers: 2 · Params: 20 (shape [2,5,2]) |
| Feature map | RyRz+CZ (encodes + entangles simultaneously) |
| Ansatz | Ry+CNOT circular (2 layers) |
| **Accuracy** | **84.81%** |
| **Recall** | **93.67% — highest of ALL 10 models** |
| Dataset | 862 TCGA patients (499 LGG, 363 GBM) · 5-fold CV |
| Features | IDH1, Age, PTEN, EGFR, ATRX |
| Task | LGG vs GBM |
| Paper | Akpinar & Oduncuoglu (2025), *Nature Sci. Reports* |
| DOI | 10.1038/s41598-025-97067-3 |
| Paper result | 74% — our VQC-2 exceeds by **+10.81 pp** |

---

### Why Quantum?
- **Superposition**: all 2⁵=32 basis states processed simultaneously
- **CZ entanglement**: models co-mutation patterns (e.g. IDH1 + ATRX)
- **Backprop gradients**: exact analytical gradients, no approximation
- **Expressive power**: 5-qubit Hilbert space vs 5-D real feature space
        """)

    with c2:
        st.markdown("""
### All 10 Models (Pipeline 2)

| Model | Accuracy | Recall |
|-------|----------|--------|
| Decision Tree | 86.08% | 90.63% |
| Random Forest | 85.96% | 90.63% |
| Logistic Reg. | 85.96% | 91.74% |
| XGBoost | 85.85% | 90.36% |
| SVC | 85.73% | 91.46% |
| KNN | 84.92% | 87.88% |
| **VQC-2 ★** | **84.81%** | **93.67% ★** |
| VQC-3 | 83.06% | 89.56% |
| VQC-1 | 82.96% | 88.19% |
| VQC-4 | 73.31% | 66.95% |

★ VQC-2 has the **highest recall** of all 10 models — clinically the most important metric (fewest missed cancers).

---

### System Requirements
- Python 3.10+
- PyTorch 2.5.0
- PennyLane 0.38.0
- Streamlit 1.38.0

### Important Notice
❌ NOT for clinical diagnosis  
❌ NOT FDA approved  
❌ NOT a medical device  

**Research and education only.**  
Always consult qualified medical professionals.

### License
GNU GPL-3.0
        """)


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="Brain Tumour Grading",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""<style>
        .block-container{padding-top:1.4rem}
        [data-testid="stMetricValue"]{font-size:1.3rem}
    </style>""", unsafe_allow_html=True)

    db = load_db()

    # Sidebar
    st.sidebar.title("🧠 Brain Tumour Grading")
    st.sidebar.markdown("---")
    page = st.sidebar.selectbox(
        "Navigate",
        ["🧠 Mode 1 — MRI Analysis",
         "🧬 Mode 2 — Clinical Features",
         "📋 Prediction History",
         "ℹ About"],
    )
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Mode 1** — Upload MRI scan\n"
        "→ Segments & grades LGG (Grade 1/2)\n\n"
        "**Mode 2** — Enter mutation data\n"
        "→ Classifies LGG vs GBM\n"
        "→ 84.81% acc · 93.67% recall"
    )
    st.sidebar.markdown("---")
    st.sidebar.warning("⚠ Research system — not for clinical use.")

    # Route
    if   page == "🧠 Mode 1 — MRI Analysis":       page_mode1(db)
    elif page == "🧬 Mode 2 — Clinical Features":   page_mode2(db)
    elif page == "📋 Prediction History":            page_history(db)
    elif page == "ℹ About":                          page_about()


if __name__ == "__main__":
    main()
