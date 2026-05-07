# 🏥 BraTS Dataset Registration Guide

## Step 1: Register for BraTS 2020 (Takes ~24 hours for approval)

### 1. Go to Registration Page:
**URL:** https://www.med.upenn.edu/cbica/brats2020/registration.html

### 2. Fill Out the Form:
- Name
- Email (use institutional/university email if possible)
- Institution/Organization
- Country
- Intended use: **"Medical image generation for diagnostic visualization using diffusion models"**

### 3. Accept Terms:
- Agree to use data only for research
- Will cite BraTS papers in publications
- Will not attempt to re-identify patients

### 4. Wait for Approval:
- Usually takes 12-24 hours
- You'll receive an email with download instructions

---

## Step 2: Download the Data (Once Approved)

### What You'll Download:
- **Training Data:** ~300 GB (all annotated cases)
- **File Format:** NIfTI (.nii.gz) - 3D medical imaging format
- **Modalities:** T1, T1ce, T2, FLAIR (we'll use T1ce - contrast enhanced)

### Download Location:
Create a folder: `C:\Pranav Aditya\MP01\BraTS2020\`

---

## Step 3: Dataset Structure (What You'll Get)

```
BraTS2020/
├── Training/
│   ├── BraTS20_Training_001/
│   │   ├── BraTS20_Training_001_t1.nii.gz       # T1 MRI
│   │   ├── BraTS20_Training_001_t1ce.nii.gz     # T1 contrast-enhanced (best for visualization)
│   │   ├── BraTS20_Training_001_t2.nii.gz       # T2 MRI
│   │   ├── BraTS20_Training_001_flair.nii.gz    # FLAIR
│   │   └── BraTS20_Training_001_seg.nii.gz      # SEGMENTATION MASK (tumor location!)
│   ├── BraTS20_Training_002/
│   │   └── ...
│   └── ... (369 patients total)
│
├── survival_info.csv                             # Patient metadata!
│   # Columns: subject_id, age, survival_days, resection_status, tumor_grade
│
└── name_mapping.csv                              # Maps to TCGA identifiers
```

---

## Step 4: What Makes BraTS Perfect for Your Project

### ✅ Exact Tumor Location:
- **Segmentation masks** show EXACTLY where the tumor is (voxel-level precision!)
- 4 tumor regions labeled:
  - **Label 1:** Necrotic/Non-enhancing tumor core
  - **Label 2:** Peritumoral edema
  - **Label 4:** GD-enhancing tumor (active tumor)
  - **Combined:** Whole tumor region

### ✅ Rich Metadata (survival_info.csv):
```csv
subject_id,age,survival_days,resection_status,tumor_grade
BraTS20_Training_001,56.0,365,GTR,HGG
BraTS20_Training_002,68.0,221,STR,HGG
BraTS20_Training_003,45.0,598,GTR,LGG
```

**You get:**
- Patient age
- Tumor grade (HGG = Glioblastoma, LGG = Lower grade)
- Survival days (severity indicator!)
- Resection status (GTR = Gross Total Resection, STR = Subtotal)

### ✅ Your Diagnostic System Can Do:
```python
# Input patient data
patient = {
    'age': 56,
    'tumor_location': 'right_frontal_lobe',  # from segmentation mask
    'tumor_volume': 12500,  # mm³ (from mask)
    'tumor_grade': 'HGG',
    'has_edema': True,
    'has_necrosis': True,
}

# Generate CT scan showing tumor at exact location
generated_scan = model.generate(patient)
```

---

## Step 5: After Download, Run Preprocessing

Once you receive BraTS data, we'll run:
1. **BraTS_Preprocessing.ipynb** - Convert NIfTI to PNG slices
2. **BraTS_Metadata_Extraction.ipynb** - Extract tumor location, size, etc.
3. **BraTS_Conditioned_Model.ipynb** - Train the full diagnostic model

---

## Alternative: BraTS 2021 (Newer, More Data)

If BraTS 2020 registration takes too long, you can also try:
- **BraTS 2021:** https://www.synapse.org/#!Synapse:syn25829067
- **BraTS 2023:** https://www.synapse.org/#!Synapse:syn51514105

All versions have the same structure and metadata!

---

## Required Citations (When You Publish)

You'll need to cite these 3 papers:

1. Menze et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE TMI 2015
2. Bakas et al. "Advancing The Cancer Genome Atlas glioma MRI collections", Nature Scientific Data 2017
3. Bakas et al. "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation", arXiv 2018

---

## Need Help?

Email: brats2020@cbica.upenn.edu

---

# 🚀 Next Steps:

1. ✅ Register at: https://www.med.upenn.edu/cbica/brats2020/registration.html
2. ⏳ Wait for approval email (~24 hours)
3. 📥 Download training data to `C:\Pranav Aditya\MP01\BraTS2020\`
4. 🔧 Run preprocessing notebooks (already created for you!)
5. 🏥 Train your diagnostic visualization model!

---

**In the meantime:** You can continue training with your existing brain CT data using the class-conditional model!
