export type PipelineStatus = 'idle' | 'ocr' | 'predicting' | 'generating' | 'complete' | 'error';

// Mirroring the prompt's detailed schema
export interface ExtractedFields {
  pseudonym: string;
  age: number;
  sex: 'M' | 'F' | 'Other' | 'Unknown';
  symptoms: string[];
}

export interface ConditionPrediction {
  label: string;
  confidence: number;
  model: string;
}

export interface Localization {
  mask: { x: number; y: number }[];
  bbox: [number, number, number, number]; // [x, y, width, height]
  iou?: number; // Optional
}

export interface TextFeature {
  feature: string;
  weight: number;
}

export interface ShapValue {
  feature: string;
  value: string | number;
  shap: number;
}

export interface Provenance {
  generator: string;
  seed: number;
  conditioning?: Record<string, any>;
  createdAt?: string; // ISO string
}

export interface XAI {
  text_features: TextFeature[];
  shap_table: ShapValue[];
  saliencyImageUrl: string; // URL to a heatmap image
  provenance: Provenance;
  naturalLanguageSummary: string;
}

export interface AuditLogEntry {
  user: string;
  action: string;
  timestamp: string; // ISO string
  details?: string;
}

export interface ExplanationMeta {
  authorId: string;
  savedAt: string; // ISO string
  tone: 'Concise' | 'Technical' | 'Patient-friendly';
  source: 'user' | 'server-draft' | 'client-draft';
}

export interface ReportData {
  id: string; // e.g., reportId
  fileName: string;
  ownerId: string;
  uploadPath: string;
  createdAt: string; // ISO string
  
  ocrText: string;
  fields: ExtractedFields;
  
  predictions: {
    organ: string;
    conditions: ConditionPrediction[];
    localization: Localization;
  };

  xai: XAI;

  generatedImageId: string;
  generatedImageUrl: string;
  
  status: 'uploaded' | 'ocr_done' | 'predicted' | 'generated' | 'approved' | 'rejected';
  auditLog: AuditLogEntry[];

  // New fields for Natural Language Explanation
  explanation?: string;
  explanationMeta?: ExplanationMeta;
}
