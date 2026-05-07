# ⚡ INSTANT DOWNLOAD: Kaggle Brain Tumor Dataset

## 🎯 YOU'RE HERE - Let's Download!

I've opened your Kaggle account settings in the browser. Follow these steps:

---

## Step 1: Create Kaggle API Token (2 minutes)

### In the browser window that just opened:

1. **Scroll down** to the "API" section
2. **Click "Create New Token"** button
3. A file `kaggle.json` will download automatically
4. **Move this file** to: `C:\Users\<YourUsername>\.kaggle\kaggle.json`

### Quick Commands (Run in PowerShell):

```powershell
# Create .kaggle directory
mkdir "$env:USERPROFILE\.kaggle" -Force

# Move the downloaded kaggle.json
Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json" -Force
```

---

## Step 2: Download the Dataset (Automatic!)

Once kaggle.json is in place, run this in PowerShell:

```powershell
cd "C:\Pranav Aditya\MP01"

# Download dataset (156 MB - takes ~2-3 minutes)
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset

# Extract
Expand-Archive -Path brain-tumor-mri-dataset.zip -DestinationPath Kaggle_BrainTumor -Force

# Cleanup
Remove-Item brain-tumor-mri-dataset.zip
```

---

## Step 3: Verify Download

You should see:
```
C:\Pranav Aditya\MP01\Kaggle_BrainTumor\
├── Training/
│   ├── glioma_tumor/        (1,321 images)
│   ├── meningioma_tumor/    (1,339 images)
│   ├── no_tumor/            (1,595 images)
│   └── pituitary_tumor/     (1,457 images)
└── Testing/
    ├── glioma_tumor/        (300 images)
    ├── meningioma_tumor/    (306 images)
    ├── no_tumor/            (405 images)
    └── pituitary_tumor/     (300 images)
```

---

## ⚠️ Don't Have a Kaggle Account?

### Create one in 2 minutes:
1. Go to: https://www.kaggle.com/account/login
2. Sign up with Google/GitHub/Email
3. Then follow Step 1 above

---

## 🎯 What We'll Do After Download:

I've already prepared a training notebook that will:
1. Load the 7,023 Kaggle images
2. Train a **class-conditional diffusion model**
3. Generate specific tumor types on demand

### Example:
```python
# Generate glioma tumor scan
generate_diagnostic_scan(
    tumor_type='glioma',
    patient_age=45,
    severity=0.8
)
```

---

## 💡 Why This is Better Than Waiting for BraTS:

✅ **Start training in 5 minutes** (vs 24-hour wait)  
✅ **More images** (7,023 vs 3,690)  
✅ **Smaller download** (156 MB vs 300 GB)  
✅ **No preprocessing needed**  
✅ **Learn the workflow** while waiting for BraTS

---

## 🚀 READY?

Tell me once you've:
1. Created the kaggle.json token
2. Placed it in `.kaggle` folder

Then I'll run the download command for you automatically!
