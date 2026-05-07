// ─── Pipeline ───

export type PipelineStageStatus = 'idle' | 'pending' | 'running' | 'complete' | 'error';

export interface PipelineStep {
  id:     number;
  label:  string;
  model:  string;
  status: PipelineStageStatus;
  result?: string;
  input:  string;
  output: string;
}

// ─── Report Data ───

export interface ExtractedFields {
  pseudonym: string;
  age:       number;
  sex:       'M' | 'F' | 'Other' | 'Unknown';
  symptoms:  string[];
}

export interface Condition {
  label:      string;
  confidence: number;
  model:      string;
  reasoning?: string;
}

export interface Localization {
  bbox:               [number, number, number, number];
  mask:               { x: number; y: number }[];
  region_description: string;
}

export interface TextFeature {
  feature: string;
  weight:  number;
}

export interface ShapEntry {
  feature: string;
  value:   string;
  impact:  number;
}

export interface XAI {
  text_features:          TextFeature[];
  shap_table:             ShapEntry[];
  saliencyImageB64:       string;
  maskImageB64:           string;
  naturalLanguageSummary: string;
  provenance: {
    generator: string;
    xai_model: string;
  };
}

export interface AuditLogEntry {
  action:    string;
  by:        string;
  timestamp: string;
}

export interface ReportData {
  id:                 string;
  fileName:           string;
  createdAt:          string;
  inputType:          'handwritten' | 'printed';
  ocrText:            string;
  fields:             ExtractedFields;
  predictions: {
    organ:              string;
    overall_confidence: number;
    conditions:         Condition[];
    localization:       Localization;
  };
  xai:                XAI;
  generatedImageB64:  string;
  auditLog:           AuditLogEntry[];
  status:             string;
}
