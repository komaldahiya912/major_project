"""
model_loader.py — Brain Tumour Grading App

Pipeline 1 — ResNet50-UNet + 4-qubit VQC (3 layers, 24 params)
             MRI scan → tumour segmentation → Grade 1 vs Grade 2

Pipeline 2 — VQC-2  (5-qubit, 2 layers, params shape [2,5,2]=20 values)
             RyRz+CZ feature map / Ry+CNOT ansatz
             Clinical features → LGG vs GBM
             Achieved 84.81% accuracy, 93.67% recall on 862 TCGA patients

IMPORTANT — circuits here must EXACTLY match the training scripts:
  Pipeline 1:  2_quantum_classifier_efficient.py
  Pipeline 2:  A2_train_quantum.py
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
import pennylane as qml
from pennylane import numpy as pnp
from PIL import Image
import gdown


# ════════════════════════════════════════════════════════════════════
#  GOOGLE DRIVE FILE IDs — fill in after running export_vqc2_model.py
#  and uploading the 3 Pipeline-2 files
# ════════════════════════════════════════════════════════════════════

# Pipeline 1 (existing — from minor project)
SEG_MODEL_ID  = "1jHuqYKhHcQIdy-8dji51Mz2QyOh7Iq3R"   # resnet_segmentation_model.pth
QNT_P1_ID     = "1l9FQMMEuPg0TSQzflfCWCzmHNyP2Brgs"   # quantum_classifier_fixed.pth

# Pipeline 2 — REPLACE these after uploading to Drive
VQC2_MODEL_ID   = "YOUR_VQC2_DRIVE_ID"     # models/vqc2_final.pth
SCALER_MIN_ID   = "YOUR_SCALER_MIN_ID"     # models/scaler_min.npy
SCALER_SCALE_ID = "YOUR_SCALER_SCALE_ID"   # models/scaler_scale.npy


# ════════════════════════════════════════════════════════════════════
#  DOWNLOAD HELPERS
# ════════════════════════════════════════════════════════════════════
def _download(file_id, dest):
    if not os.path.exists(dest):
        print(f"⬇  Downloading {os.path.basename(dest)} …")
        gdown.download(f"https://drive.google.com/uc?id={file_id}",
                       dest, quiet=False)
        print(f"✅  {os.path.basename(dest)} ready")
    else:
        print(f"✅  {os.path.basename(dest)} already present")


def download_p1_models(models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    seg  = os.path.join(models_dir, "resnet_segmentation_model.pth")
    qnt  = os.path.join(models_dir, "quantum_classifier_fixed.pth")
    _download(SEG_MODEL_ID, seg)
    _download(QNT_P1_ID,    qnt)
    return seg, qnt


def download_p2_models(models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    vqc2  = os.path.join(models_dir, "vqc2_final.pth")
    smin  = os.path.join(models_dir, "scaler_min.npy")
    sscl  = os.path.join(models_dir, "scaler_scale.npy")
    _download(VQC2_MODEL_ID,   vqc2)
    _download(SCALER_MIN_ID,   smin)
    _download(SCALER_SCALE_ID, sscl)
    return vqc2, smin, sscl


# ════════════════════════════════════════════════════════════════════
#  PIPELINE 1 — SEGMENTATION MODEL (ResNet50-UNet + Attention)
# ════════════════════════════════════════════════════════════════════
class _AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g  = nn.Sequential(nn.Conv2d(F_g,   F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x  = nn.Sequential(nn.Conv2d(F_l,   F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi  = nn.Sequential(nn.Conv2d(F_int, 1,     1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


class ImprovedResUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        bb = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        old_conv = bb.conv1
        bb.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        bb.conv1.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)

        self.conv1   = bb.conv1;   self.bn1     = bb.bn1
        self.relu    = bb.relu;    self.maxpool = bb.maxpool
        self.layer1  = bb.layer1;  self.layer2  = bb.layer2
        self.layer3  = bb.layer3;  self.layer4  = bb.layer4

        self.att4 = _AttentionBlock(1024, 1024, 512)
        self.att3 = _AttentionBlock(512,  512,  256)
        self.att2 = _AttentionBlock(256,  256,  128)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(2048,1024,3,padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True), nn.Dropout2d(0.3))
        self.up2 = nn.Sequential(
            nn.Conv2d(2048,512,3,padding=1),  nn.BatchNorm2d(512),  nn.ReLU(inplace=True), nn.Dropout2d(0.2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
        self.up3 = nn.Sequential(
            nn.Conv2d(1024,256,3,padding=1),  nn.BatchNorm2d(256),  nn.ReLU(inplace=True), nn.Dropout2d(0.1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
        self.up4 = nn.Sequential(
            nn.Conv2d(512,128,3,padding=1),   nn.BatchNorm2d(128),  nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
        self.up5 = nn.Sequential(
            nn.Conv2d(128,64,3,padding=1),    nn.BatchNorm2d(64),   nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
        self.final = nn.Sequential(
            nn.Conv2d(64,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32,1,1))

    def forward(self, x):
        x1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x2 = self.layer1(x1); x3 = self.layer2(x2)
        x4 = self.layer3(x3); x5 = self.layer4(x4)
        d1 = self.up1(x5);   d1 = torch.cat([d1, self.att4(d1, x4)], dim=1)
        d2 = self.up2(d1);   d2 = torch.cat([d2, self.att3(d2, x3)], dim=1)
        d3 = self.up3(d2);   d3 = torch.cat([d3, self.att2(d3, x2)], dim=1)
        d4 = self.up4(d3);   d5 = self.up5(d4)
        return self.final(d5), x5


# ════════════════════════════════════════════════════════════════════
#  PIPELINE 1 — QUANTUM CIRCUIT
#  Matches 2_quantum_classifier_efficient.py exactly:
#  4 qubits, 3 layers, RY encoding, [RY+RZ+CNOT] ansatz, 24 params (1-D)
# ════════════════════════════════════════════════════════════════════
_N1 = 4
_dev1 = qml.device("default.qubit", wires=_N1)

@qml.qnode(_dev1, interface="torch", diff_method="backprop")
def _p1_circuit(inputs, weights):
    # Encoding: RY on each qubit
    for i in range(_N1):
        qml.RY(np.pi * inputs[i], wires=i)
    # Ansatz: 3 layers of [RY+RZ per qubit + circular CNOT]
    for layer in range(3):
        base = layer * _N1 * 2
        for i in range(_N1):
            qml.RY(weights[base + i],        wires=i)
            qml.RZ(weights[base + _N1 + i],  wires=i)
        for i in range(_N1 - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[_N1 - 1, 0])
    return qml.expval(qml.PauliZ(0))


class _P1QuantumClassifier(nn.Module):
    """4-qubit, 3-layer, 24-parameter VQC — matches saved checkpoint."""
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(3 * _N1 * 2) * 0.5)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return torch.stack([_p1_circuit(x[i], self.weights)
                            for i in range(x.shape[0])])


# ════════════════════════════════════════════════════════════════════
#  PIPELINE 1 — BrainTumorPredictor
# ════════════════════════════════════════════════════════════════════
class BrainTumorPredictor:
    """
    Full Pipeline 1 inference.
    1. Load grayscale MRI
    2. ResNet50-UNet segments tumour region (512×512)
    3. Extract 4 features from segmentation mask
    4. 4-qubit VQC classifies Grade 1 vs Grade 2
    """

    def __init__(self, seg_path=None, quantum_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[P1] device = {self.device}")

        if seg_path is None or quantum_path is None:
            seg_path, quantum_path = download_p1_models()

        # Segmentation model
        self.seg = ImprovedResUNet(pretrained=False)
        ckpt = torch.load(seg_path, map_location=self.device, weights_only=False)
        self.seg.load_state_dict(ckpt.get("model_state_dict", ckpt))
        self.seg.to(self.device).eval()
        print(f"[P1] seg model loaded — Dice: {ckpt.get('dice', 'N/A')}")

        # Quantum classifier
        self.qmodel = _P1QuantumClassifier()
        qckpt = torch.load(quantum_path, map_location=self.device, weights_only=False)
        self.qmodel.load_state_dict(qckpt.get("model_state_dict", qckpt))
        self.qmodel.to(self.device).eval()
        print(f"[P1] quantum model loaded — acc: {qckpt.get('accuracy', 'N/A')}")

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def predict(self, image_path: str) -> dict:
        """
        Returns
        -------
        {
            tumor_present    : bool
            tumor_mask       : np.ndarray (512,512) probabilities
            predicted_grade  : int  (1 or 2)
            grade_confidence : float  0-1
            tumor_area       : float  (pixel count)
            segmentation_stats : {mean_prob, std_prob, max_prob, tumor_ratio}
        }
        """
        img = Image.open(image_path).convert("L")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = self.seg(tensor)
            seg_probs  = torch.sigmoid(logits).squeeze().cpu().numpy()

        mean_prob    = float(seg_probs.mean())
        std_prob     = float(seg_probs.std())
        max_prob     = float(seg_probs.max())
        tumor_pixels = int((seg_probs > 0.5).sum())
        tumor_ratio  = tumor_pixels / (512 * 512)

        # 4 features fed to quantum classifier — match training feature set
        features = torch.tensor(
            [mean_prob, std_prob, max_prob, tumor_ratio],
            dtype=torch.float32,
        ).clamp(0.0, 1.0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_raw = self.qmodel(features).item()

        grade_prob = torch.sigmoid(torch.tensor(q_raw)).item()

        return {
            "tumor_present":    tumor_pixels > 10,
            "tumor_mask":       seg_probs,
            "predicted_grade":  2 if grade_prob > 0.5 else 1,
            "grade_confidence": grade_prob,
            "tumor_area":       float(tumor_pixels),
            "segmentation_stats": {
                "mean_prob":   mean_prob,
                "std_prob":    std_prob,
                "max_prob":    max_prob,
                "tumor_ratio": tumor_ratio,
            },
        }


# ════════════════════════════════════════════════════════════════════
#  PIPELINE 2 — VQC-2 CIRCUIT
#  Matches A2_train_quantum.py _vqc2() exactly:
#  5 qubits, 2 layers
#  Feature map  : RY(π·x) + RZ(π·x) per qubit + CZ circular
#  Ansatz       : RY(params[l,i,0]) per qubit + CNOT circular
#  params shape : [N_LAYERS, N_QUBITS, 2]  but only index [:,i,0] used
#  n_params     : 2 × 5 × 2 = 20 total values in state_dict
# ════════════════════════════════════════════════════════════════════
_N2      = 5
_N2_L    = 2
_dev2    = qml.device("default.qubit", wires=_N2)


@qml.qnode(_dev2, interface="torch", diff_method="backprop")
def _vqc2_single(x, params):
    """Single-sample VQC-2 forward. params shape: [N_LAYERS, N_QUBITS, 2]"""
    # Feature map
    for i in range(_N2):
        qml.RY(np.pi * x[i], wires=i)
        qml.RZ(np.pi * x[i], wires=i)
    for i in range(_N2):
        qml.CZ(wires=[i, (i + 1) % _N2])
    # Ansatz
    for l in range(_N2_L):
        for i in range(_N2):
            qml.RY(params[l, i, 0], wires=i)
        for i in range(_N2):
            qml.CNOT(wires=[i, (i + 1) % _N2])
    return qml.expval(qml.PauliZ(0))


class _VQC2Model(nn.Module):
    """
    Wraps VQC-2.  params shape [N_LAYERS, N_QUBITS, 2] = [2,5,2]
    This matches what A2_train_quantum.py's VQCModel saves.
    Uses torch.func.vmap for batched inference.
    """
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(
            torch.randn(_N2_L, _N2, 2, dtype=torch.float64) * 0.5
        )

    def forward(self, x_batch):
        # x_batch: (B, 5)  → (B,)
        return torch.func.vmap(_vqc2_single, in_dims=(0, None))(
            x_batch, self.params
        )


# ════════════════════════════════════════════════════════════════════
#  PIPELINE 2 — VQC2Predictor
# ════════════════════════════════════════════════════════════════════
class VQC2Predictor:
    """
    Full Pipeline 2 inference.
    Feature order (MUST match A1_eda_preprocessing.py output):
        IDH1,  Age_at_diagnosis,  PTEN,  EGFR,  ATRX
    Label encoding: LGG=0, GBM=1
    Scaler: MinMax applied with scaler.data_min_ and scaler.scale_
            (both saved as .npy by A1_eda_preprocessing.py)
    """

    FEATURE_NAMES = ["IDH1", "Age_at_diagnosis", "PTEN", "EGFR", "ATRX"]

    def __init__(self, model_path=None, scaler_min_path=None, scaler_scale_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[P2] device = {self.device}")

        if model_path is None:
            model_path, scaler_min_path, scaler_scale_path = download_p2_models()

        # Scaler (float32 for speed; circuit uses float64 internally)
        self.scaler_min   = np.load(scaler_min_path).astype(np.float32)
        self.scaler_scale = np.load(scaler_scale_path).astype(np.float32)
        print(f"[P2] scaler_min   = {self.scaler_min}")
        print(f"[P2] scaler_scale = {self.scaler_scale}")

        # VQC-2 model
        self.model = _VQC2Model()
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()
        print(f"[P2] VQC-2 loaded — acc: {ckpt.get('mean_metrics',{}).get('acc','N/A')}")

    def predict(self, idh1: int, age: float,
                pten: int, egfr: int, atrx: int) -> dict:
        """
        Parameters
        ----------
        idh1, pten, egfr, atrx : 0 (not mutated) or 1 (mutated)
        age                    : age at diagnosis in years (float)

        Returns
        -------
        {
            predicted_class  : 'LGG' or 'GBM'
            confidence       : probability of predicted class (0-1)
            raw_output       : PauliZ expectation value (-1 to +1)
            lgg_probability  : float 0-1
            gbm_probability  : float 0-1
        }
        """
        # Step 1 — assemble in EXACT training feature order
        raw = np.array([float(idh1), float(age), float(pten),
                        float(egfr), float(atrx)], dtype=np.float32)

        # Step 2 — MinMax scale using SAME params as A1_eda_preprocessing.py
        #  formula: (x - data_min_) / scale_   clipped to [0,1]
        scaled = (raw - self.scaler_min) / (self.scaler_scale + 1e-8)
        scaled = np.clip(scaled, 0.0, 1.0)

        # Step 3 — quantum inference (circuit expects float64)
        x = torch.tensor(scaled, dtype=torch.float64).unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw_out = self.model(x).item()

        # Step 4 — sigmoid → class probabilities
        gbm_prob = 1.0 / (1.0 + math.exp(-raw_out))
        lgg_prob = 1.0 - gbm_prob
        pred_cls  = "GBM" if gbm_prob > 0.5 else "LGG"
        confidence = gbm_prob if pred_cls == "GBM" else lgg_prob

        return {
            "predicted_class": pred_cls,
            "confidence":      round(confidence, 4),
            "raw_output":      round(raw_out,    4),
            "lgg_probability": round(lgg_prob,   4),
            "gbm_probability": round(gbm_prob,   4),
        }
