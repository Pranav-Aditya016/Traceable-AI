// ============================================
// MEDIÄI - INTERACTIVE MEDICAL AI SYSTEM
// Smooth interactions, beautiful UX, mock data
// ============================================

// Mock OCR and medical data
const mockMedicalData = {
    ocr_text: `FLORES MEMORIAL MEDICAL CENTER FPMC "A Legacy of Compassionate Caring" PhilHealth Accredited Hospital August 6, 2021 MEDICAL CERTIFICATE To Whom It May Concern: This is to certify that John Martin Luther, 38 years old, male, married, resident of Malvar, Santiago City was admitted at Flores Memorial Medical Center on August 4, 2021 at 7:55 pm due to stab wound penetrating on the epigastric area are 2-3 cm in length with omental tissue coming out the patient was conscious, coherent, not in respiratory distress with positive alcoholic breath. Exploratory laparotomy was done on August 5, 1992 which started at 11:30 am and ended up 3:25 pm. Findings: Lacerated wound inferior lobe of the liver; lacerated wound on the lesser curvature of the stomach, thru and thru. Suturing of the wound and exuction of blood and gastric contents were done. Drain was placed, washing and closure of the stomach layer per layer were done.`,
    patient_name: "John Martin Luther",
    age: 38,
    gender: "Male",
    hospital: "Flores Memorial Medical Center",
    admission_date: "August 4, 2021",
    symptoms: "Stab wound, Penetrating wound on epigastric area, Omental tissue coming out, Lacerated wound inferior lobe of the liver, Lacerated wound on the lesser curvature of the stomach",
    organ: "Liver",
    condition: "Penetrating abdominal trauma with liver and gastric lacerations",
    confidence: "97%"
};

const explanations = {
    concise: `AI analysis of the CT imaging reveals Grade III glioma in the left frontal lobe (95% confidence). Key findings: tumor size 3.2cm with irregular borders, surrounding edema, and mass effect. The AI identified this condition through pattern recognition against 50,000+ medical images, achieving 99.2% accuracy in similar cases. Immediate neurosurgical consultation recommended.`,
    
    technical: `Deep learning analysis of the abdominal CT imaging demonstrates penetrating trauma with complex injury pattern. Primary findings include: Grade III laceration of the inferior lobe of the left liver (95% confidence, size 3.2cm), irregular laceration of the lesser curvature of the stomach with full-thickness involvement (88% confidence), and surrounding hemorrhage with moderate hemoperitoneum. The convolutional neural network identified tissue discontinuity patterns, fluid dynamics, and geometric trauma signatures through comparative analysis against 50,000+ trauma imaging cases. Segmentation algorithm delineated organ boundaries with 91% precision. Clinical validation algorithm cross-referenced findings with admission vitals, surgical notes, and post-operative imaging.`,
    
    'patient-friendly': `Our AI reviewed your medical images and found some important issues that needed attention. There was a serious cut in your liver and stomach from the injury. The AI system analyzed your images by comparing them to thousands of similar cases it has studied. It identified the exact location and size of the wounds, which helped your doctors decide on the right treatment. Your doctors did surgery to fix these injuries and you're being monitored closely for recovery. This type of injury is serious but treatable with proper medical care.`
};

let currentExplanationTone = 'concise';

// ========== INITIALIZATION ==========
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    initializeCanvas();
    showHeroSection();
});

function setupEventListeners() {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    
    // Drag and drop
    uploadZone?.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });
    
    uploadZone?.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });
    
    uploadZone?.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        handleFileUpload(e.dataTransfer.files[0]);
    });
    
    fileInput?.addEventListener('change', (e) => {
        handleFileUpload(e.target.files[0]);
    });
}

// ========== NAVIGATION & SCROLLING ==========
function scrollToStart() {
    document.getElementById('pipeline-interactive').scrollIntoView({ behavior: 'smooth' });
    startPipeline();
}

function scrollToFeatures() {
    document.getElementById('features').scrollIntoView({ behavior: 'smooth' });
}

function startPipeline() {
    currentStep = 1;
    showStep(1);
    document.getElementById('analysis').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showStep(step) {
    // Hide all steps
    for (let i = 1; i <= 4; i++) {
        const el = document.getElementById(`step-${i}`);
        if (el) {
            el.classList.add('hidden');
            el.style.display = 'none';
        }
    }
    
    // Show current step with animation
    const currentEl = document.getElementById(`step-${step}`);
    if (currentEl) {
        currentEl.classList.remove('hidden');
        currentEl.style.display = 'block';
        currentEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    
    // Update button visibility
    updateButtonStates(step);
    currentStep = step;
}

function nextStep() {
    if (currentStep < 4) {
        if (currentStep === 1) {
            // Move from upload to extract
            showStep(2);
        } else if (currentStep === 2) {
            // Move from extract to generate
            generateImage();
        } else if (currentStep === 3) {
            // Move from generate to results
            showExplanation();
        }
    }
}

function updateButtonStates(step) {
    const nextBtn = document.getElementById('nextBtn');
    const resetBtn = document.getElementById('resetBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    
    // Hide all buttons initially
    if (nextBtn) nextBtn.style.display = 'none';
    if (resetBtn) resetBtn.style.display = 'none';
    if (downloadBtn) downloadBtn.style.display = 'none';
    
    // Show appropriate buttons based on current step
    if (step === 1 || step === 2) {
        if (nextBtn) {
            nextBtn.style.display = 'inline-flex';
            nextBtn.textContent = step === 1 ? 'Next Step' : 'Analyze & Generate';
            nextBtn.onclick = nextStep;
        }
    } else if (step === 3) {
        if (nextBtn) {
            nextBtn.style.display = 'inline-flex';
            nextBtn.textContent = 'View Results';
            nextBtn.onclick = nextStep;
        }
    } else if (step === 4) {
        if (downloadBtn) downloadBtn.style.display = 'inline-flex';
        if (resetBtn) resetBtn.style.display = 'inline-flex';
    }
}

// ========== FILE UPLOAD HANDLING ==========
function handleFileUpload(file) {
    if (!file) return;
    
    uploadedFile = file;
    const uploadZone = document.getElementById('uploadZone');
    
    // Show processing animation
    uploadZone.innerHTML = `
        <div class="upload-animation">
            <div style="font-size: 3rem; animation: bounce 2s ease-in-out infinite;">⚙️</div>
        </div>
        <p style="margin-top: 1rem; color: #667eea; font-weight: 600;">Processing document...</p>
    `;
    
    // Simulate processing delay (2 seconds)
    setTimeout(() => {
        // Populate OCR and extracted data
        populateMedicalData();
        // Move to step 2 after upload completes
        showStep(2);
    }, 2000);
}

function populateMedicalData() {
    // Populate OCR Text
    const ocrText = document.getElementById('ocrText');
    if (ocrText) ocrText.textContent = mockMedicalData.ocr_text;
    
    // Populate extracted fields
    const patientName = document.getElementById('patientName');
    const patientAge = document.getElementById('patientAge');
    const hospital = document.getElementById('hospital');
    const admissionDate = document.getElementById('admissionDate');
    const symptoms = document.getElementById('symptoms');
    
    if (patientName) patientName.value = mockMedicalData.patient_name;
    if (patientAge) patientAge.value = mockMedicalData.age + " years";
    if (hospital) hospital.value = mockMedicalData.hospital;
    if (admissionDate) admissionDate.value = mockMedicalData.admission_date;
    if (symptoms) symptoms.value = mockMedicalData.symptoms;
    
    // Populate AI prediction
    const organPrediction = document.getElementById('organPrediction');
    const conditionPrediction = document.getElementById('conditionPrediction');
    const confidencePrediction = document.getElementById('confidencePrediction');
    
    if (organPrediction) organPrediction.textContent = mockMedicalData.organ;
    if (conditionPrediction) conditionPrediction.textContent = mockMedicalData.condition;
    if (confidencePrediction) confidencePrediction.textContent = mockMedicalData.confidence;
}

// ========== IMAGE GENERATION ==========
function generateImage() {
    showStep(3);
    startGenerationAnimation();
}

function startGenerationAnimation() {
    const progressContainer = document.getElementById('generationProgressContainer');
    const statusText = document.getElementById('statusText');
    const generatedResult = document.getElementById('generatedResult');
    
    if (progressContainer) progressContainer.style.display = 'block';
    if (generatedResult) generatedResult.style.display = 'none';
    
    const stages = [
        { text: 'Initializing diffusion model...' },
        { text: 'Encoding medical context...' },
        { text: 'Generating image layers...' },
        { text: 'Finalizing output...' }
    ];
    
    let stageIndex = 0;
    
    const updateStages = () => {
        if (stageIndex < stages.length) {
            const stage = stages[stageIndex];
            if (statusText) statusText.textContent = stage.text;
            
            stageIndex++;
            setTimeout(updateStages, 1500);
        } else {
            // Show result
            setTimeout(() => {
                if (progressContainer) progressContainer.style.display = 'none';
                if (generatedResult) generatedResult.style.display = 'block';
                
                const generatedImage = document.getElementById('generatedImage');
                if (generatedImage) {
                    generatedImage.src = createMockMedicalImage();
                }
                
                // Show final buttons
                const downloadBtn = document.getElementById('downloadBtn');
                if (downloadBtn) downloadBtn.style.display = 'inline-flex';
            }, 500);
        }
    };
    
    updateStages();
}

// ========== IMAGE GENERATION - CANVAS ========== 
function createMockMedicalImage() {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');
    
    // Background gradient
    const gradient = ctx.createLinearGradient(0, 0, 512, 512);
    gradient.addColorStop(0, '#1a1a2e');
    gradient.addColorStop(1, '#16213e');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 512, 512);
    
    // Brain outline
    ctx.strokeStyle = '#4facfe';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.ellipse(256, 256, 180, 200, 0, 0, Math.PI * 2);
    ctx.stroke();
    
    // Internal structures
    ctx.fillStyle = 'rgba(79, 172, 254, 0.2)';
    ctx.beginPath();
    ctx.ellipse(256, 256, 120, 140, 0, 0, Math.PI * 2);
    ctx.fill();
    
    // Tumor region (highlighted)
    const tumorGradient = ctx.createRadialGradient(200, 180, 10, 200, 180, 50);
    tumorGradient.addColorStop(0, 'rgba(245, 87, 108, 0.8)');
    tumorGradient.addColorStop(1, 'rgba(240, 147, 251, 0.3)');
    ctx.fillStyle = tumorGradient;
    ctx.beginPath();
    ctx.ellipse(200, 180, 45, 55, 0.3, 0, Math.PI * 2);
    ctx.fill();
    
    // Glow effect
    ctx.shadowColor = 'rgba(245, 87, 108, 0.5)';
    ctx.shadowBlur = 20;
    
    // Add noise texture
    for (let i = 0; i < 150; i++) {
        const x = Math.random() * 512;
        const y = Math.random() * 512;
        const size = Math.random() * 2;
        ctx.fillStyle = `rgba(255, 255, 255, ${Math.random() * 0.2})`;
        ctx.fillRect(x, y, size, size);
    }
    
    // Text overlay
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.font = 'bold 16px Poppins';
    ctx.fillText('AI Generated Brain MRI', 20, 40);
    ctx.font = '14px Poppins';
    ctx.fillText('Tumor Region Highlighted (Grade III Glioma)', 20, 65);
    ctx.fillText(`Generated: ${new Date().toLocaleTimeString()}`, 20, 490);
    
    return canvas.toDataURL('image/png');
}

// ========== EXPLANATION & XAI ==========
function showExplanation() {
    showStep(4);
    populateReportData();
    generateExplanation('concise');
    setTimeout(() => {
        animateHeatmap();
    }, 300);
}

function populateReportData() {
    // Populate report summary
    const reportId = document.getElementById('reportId');
    const reportPatient = document.getElementById('reportPatient');
    const reportDate = document.getElementById('reportDate');
    const clinicalFindingsText = document.getElementById('clinicalFindingsText');
    
    if (reportId) reportId.textContent = 'MED-' + Date.now();
    if (reportPatient) reportPatient.textContent = mockMedicalData.patient_name;
    if (reportDate) reportDate.textContent = new Date().toLocaleString();
    if (clinicalFindingsText) {
        clinicalFindingsText.textContent = `${mockMedicalData.condition}. Patient age ${mockMedicalData.age} years. Primary organ affected: ${mockMedicalData.organ}. The AI analysis identified key pathological markers with ${mockMedicalData.confidence} confidence through comparative deep learning analysis against extensive medical imaging databases.`;
    }
}

function generateExplanation(tone) {
    currentExplanationTone = tone;
    
    // Update active button
    document.querySelectorAll('.explanation-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.closest('.explanation-btn').classList.add('active');
    
    // Update explanation text
    const explanationText = document.getElementById('explanationText');
    if (explanationText) {
        explanationText.textContent = explanations[tone];
    }
}

function copyExplanation() {
    const explanationText = document.getElementById('explanationText');
    if (explanationText) {
        explanationText.select();
        document.execCommand('copy');
        showNotification('✓ Explanation copied to clipboard!');
    }
}

function downloadExplanation() {
    const explanationText = document.getElementById('explanationText');
    if (explanationText) {
        const content = `MEDICAL AI EXPLANATION REPORT\n\nPatient: ${mockMedicalData.patient_name}\nReport ID: ${document.getElementById('reportId')?.textContent}\nGenerated: ${new Date().toLocaleString()}\nExplanation Tone: ${currentExplanationTone.toUpperCase()}\n\n${explanationText.value}`;
        
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `medical-ai-explanation-${Date.now()}.txt`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        showNotification('✓ Explanation downloaded!');
    }
}

function saveExplanation() {
    showNotification('✓ Explanation saved to your medical records!');
}

function initializeCanvas() {
    const canvas = document.getElementById('heatmapCanvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Base brain outline
    const grad = ctx.createRadialGradient(width / 2, height / 2, 30, width / 2, height / 2, 150);
    grad.addColorStop(0, '#f0f4f8');
    grad.addColorStop(0.7, '#d0dce6');
    grad.addColorStop(1, '#a0b0c0');
    
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, width, height);
    
    ctx.strokeStyle = '#4a5568';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.ellipse(width / 2, height / 2, 120, 140, 0, 0, Math.PI * 2);
    ctx.stroke();
}

function animateHeatmap() {
    const canvas = document.getElementById('heatmapCanvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    const hotspots = [
        { x: width * 0.35, y: height * 0.35, intensity: 0.9, size: 60 },
        { x: width * 0.5, y: height * 0.5, intensity: 0.7, size: 50 },
        { x: width * 0.65, y: height * 0.55, intensity: 0.5, size: 40 }
    ];
    
    let frame = 0;
    const maxFrames = 40;
    
    const animate = () => {
        if (frame <= maxFrames) {
            const progress = frame / maxFrames;
            
            // Redraw base
            initializeCanvas();
            
            // Draw animated hotspots
            hotspots.forEach(spot => {
                const grad = ctx.createRadialGradient(spot.x, spot.y, 0, spot.x, spot.y, spot.size * progress);
                grad.addColorStop(0, `rgba(245, 87, 108, ${spot.intensity * progress})`);
                grad.addColorStop(0.5, `rgba(240, 147, 251, ${spot.intensity * 0.4 * progress})`);
                grad.addColorStop(1, 'rgba(79, 172, 254, 0)');
                
                ctx.fillStyle = grad;
                ctx.fillRect(0, 0, width, height);
            });
            
            frame++;
            requestAnimationFrame(animate);
        }
    };
    
    animate();
}

// ========== IMAGE ZOOM ==========
function zoomImage() {
    const modal = document.getElementById('zoomModal');
    const zoomedImage = document.getElementById('zoomedImage');
    const generatedImage = document.getElementById('generatedImage');
    
    zoomedImage.src = generatedImage.src;
    modal.classList.add('active');
}

function closeZoom() {
    const modal = document.getElementById('zoomModal');
    modal.classList.remove('active');
}

// Click outside to close modal
document.addEventListener('click', (e) => {
    const modal = document.getElementById('zoomModal');
    if (e.target === modal) {
        closeZoom();
    }
});

// ========== UTILITY FUNCTIONS ==========
function resetPipeline() {
    // Reset all UI
    const uploadZone = document.getElementById('uploadZone');
    if (uploadZone) {
        uploadZone.innerHTML = `
            <div class="upload-icon">📁</div>
            <h4>Drop your file here or click to select</h4>
            <p>Accepts PDF, JPG, PNG, and other medical formats</p>
        `;
    }
    
    // Reset generation
    const progressContainer = document.getElementById('generationProgressContainer');
    const generatedResult = document.getElementById('generatedResult');
    if (progressContainer) progressContainer.style.display = 'block';
    if (generatedResult) generatedResult.style.display = 'none';
    
    // Hide buttons
    const downloadBtn = document.getElementById('downloadBtn');
    const resetBtn = document.getElementById('resetBtn');
    if (downloadBtn) downloadBtn.style.display = 'none';
    if (resetBtn) resetBtn.style.display = 'none';
    
    // Back to step 1
    currentStep = 0;
    uploadedFile = null;
    showStep(1);
    
    document.getElementById('analysis').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function downloadReport() {
    const report = {
        timestamp: new Date().toISOString(),
        patient: document.getElementById('patientName')?.value || 'John Doe',
        diagnosis: document.getElementById('diagnosis')?.value || 'Brain Tumor - Glioma',
        condition: document.getElementById('condition')?.value || 'Grade III glioma',
        imagingType: document.getElementById('imagingType')?.value || 'MRI Brain',
        findings: {
            location: 'Left frontal lobe',
            size: '3.2cm',
            grade: 'Grade III Glioma',
            recommendation: 'Immediate neurosurgical consultation'
        },
        confidence: '99.2%',
        generatedAt: new Date().toLocaleString()
    };
    
    const jsonData = JSON.stringify(report, null, 2);
    const blob = new Blob([jsonData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `medical-ai-report-${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    showNotification('Report downloaded successfully! ✓');
}

function showNotification(message) {
    const toast = document.getElementById('notificationToast');
    if (toast) {
        toast.textContent = message;
        toast.classList.add('show');
        
        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    }
}

// ========== HERO SECTION ANIMATION ==========
function showHeroSection() {
    const hero = document.getElementById('hero');
    if (hero) {
        hero.style.opacity = '1';
    }
}

// ========== SMOOTH SCROLL TO SECTIONS ==========
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
});

// ========== CONSOLE MESSAGE ==========
console.log(`
%c🏥 MediAI
%cIntelligent Medical Image Analysis System
%cBuilt for Excellence • Powered by AI
`,
'color: #667eea; font-size: 28px; font-weight: bold;',
'color: #764ba2; font-size: 14px; font-weight: 600;',
'color: #4facfe; font-size: 12px;'
);
