import { GoogleGenAI, Type } from "@google/genai";
import type { ReportData, ExtractedFields, ConditionPrediction, Localization, XAI } from '../types';

const fileToGenerativePart = async (file: File) => {
  const base64EncodedDataPromise = new Promise<string>((resolve) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result.split(',')[1]);
      } else {
        resolve('');
      }
    };
    reader.readAsDataURL(file);
  });
  const data = await base64EncodedDataPromise;
  return {
    inlineData: {
      data,
      mimeType: file.type,
    },
  };
};

const analysisSchema = {
  type: Type.OBJECT,
  properties: {
    ocrText: { type: Type.STRING, description: "Full OCR text." },
    fields: {
      type: Type.OBJECT,
      properties: {
        pseudonym: { type: Type.STRING, description: "Patient pseudonym (e.g., 'Patient A')." },
        age: { type: Type.INTEGER },
        sex: { type: Type.STRING, enum: ['M', 'F', 'Other', 'Unknown'] },
        symptoms: { type: Type.ARRAY, items: { type: Type.STRING } }
      },
      required: ["pseudonym", "age", "sex", "symptoms"]
    },
    predictions: {
      type: Type.OBJECT,
      properties: {
        organ: { type: Type.STRING, description: "Predicted organ." },
        conditions: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              label: { type: Type.STRING },
              confidence: { type: Type.NUMBER },
              model: { type: Type.STRING }
            },
            required: ["label", "confidence", "model"]
          }
        },
        localization: {
          type: Type.OBJECT,
          properties: {
            mask: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: { x: { type: Type.INTEGER }, y: { type: Type.INTEGER } },
                required: ["x", "y"]
              }
            },
            bbox: { type: Type.ARRAY, items: { type: Type.INTEGER }, minItems: 4, maxItems: 4 }
          },
          required: ["mask", "bbox"]
        }
      },
      required: ["organ", "conditions", "localization"]
    },
    xai: {
      type: Type.OBJECT,
      properties: {
        text_features: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              feature: { type: Type.STRING },
              weight: { type: Type.NUMBER }
            },
            required: ["feature", "weight"]
          }
        },
        shap_table: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              feature: { type: Type.STRING },
              value: { type: Type.STRING }, // Use string to accommodate mixed types
              shap: { type: Type.NUMBER }
            },
            required: ["feature", "value", "shap"]
          }
        }
      },
      required: ["text_features", "shap_table"]
    }
  },
  required: ["ocrText", "fields", "predictions", "xai"]
};

type AnalysisResult = {
  ocrText: string;
  fields: ExtractedFields;
  predictions: {
    organ: string;
    conditions: ConditionPrediction[];
    localization: Localization;
  };
  xai: Omit<XAI, 'saliencyImageUrl' | 'provenance' | 'naturalLanguageSummary'>;
};

export const analyzeAndGenerate = async (file: File, onStatusChange: (status: 'predicting' | 'generating') => void): Promise<ReportData> => {
  const apiKey = process.env.GEMINI_API_KEY || process.env.API_KEY;
  if (!apiKey) {
    throw new Error("Gemini API key is not available in the environment.");
  }

  const ai = new GoogleGenAI({ apiKey });
  const imagePart = await fileToGenerativePart(file);
  onStatusChange('predicting');

  const analysisPrompt = `Analyze this medical document. Perform OCR. Extract fields (pseudonym, age, sex, symptoms). Predict the primary organ and up to 2 conditions with confidence. Provide localization (mask polygon and bbox) for the primary condition. Generate XAI artifacts (top 3 text features with weights, SHAP table for all fields). Adhere strictly to the provided JSON schema. The image for the mask and bbox should be assumed to be 512x512 pixels.`;

  const analysisResponse = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: { parts: [imagePart, { text: analysisPrompt }] },
    config: {
      responseMimeType: "application/json",
      responseSchema: analysisSchema
    }
  });

  const analysisResult: AnalysisResult = JSON.parse(analysisResponse.text);

  onStatusChange('generating');
  const primaryCondition = analysisResult.predictions.conditions[0];
  const imagePrompt = `Create a highly detailed, medically accurate synthetic image of a CT scan of a ${analysisResult.predictions.organ}. The patient is a ${analysisResult.fields.age}-year-old ${analysisResult.fields.sex} presenting with symptoms of ${analysisResult.fields.symptoms.join(', ')}. The image must clearly show a well-defined ${primaryCondition.label}. Focus on realistic tissue texture and anatomical accuracy. The image should be 512x512 pixels.`;
  
  const imageResponse = await ai.models.generateContent({
    model: 'gemini-2.5-flash-image',
    contents: {
      parts: [
        {
          text: imagePrompt,
        },
      ],
    },
    config: {
      imageConfig: {
        aspectRatio: "1:1"
      },
    },
  });

  let generatedImageUrl = "https://via.placeholder.com/512.png?text=Generation+Failed";
  if (imageResponse.candidates?.[0]?.content?.parts) {
    for (const part of imageResponse.candidates[0].content.parts) {
      if (part.inlineData) {
        const base64EncodeString: string = part.inlineData.data;
        generatedImageUrl = `data:image/png;base64,${base64EncodeString}`;
        break;
      }
    }
  }

  const explanationPrompt = `Based on this JSON data, generate a plain-English summary for a clinician. Use this template:
  "The system extracted age ({age}), sex ({sex}), and symptoms ({symptoms}) from the document. The classification model ({model}) predicted a {condition} in the {organ} with {confidence}% confidence. Localization places the finding at coordinates [{bbox}]. XAI indicates the prediction relied primarily on: {feature1} ({weight1}), {feature2} ({weight2}). The image below is synthetic, generated by {generator}."
  Data: ${JSON.stringify({
    ...analysisResult.fields,
    organ: analysisResult.predictions.organ,
    condition: primaryCondition.label,
    confidence: (primaryCondition.confidence * 100).toFixed(0),
    model: primaryCondition.model,
    bbox: analysisResult.predictions.localization.bbox.join(', '),
    feature1: analysisResult.xai.text_features[0]?.feature || 'N/A',
    weight1: analysisResult.xai.text_features[0]?.weight.toFixed(2) || 'N/A',
    feature2: analysisResult.xai.text_features[1]?.feature || 'N/A',
    weight2: analysisResult.xai.text_features[1]?.weight.toFixed(2) || 'N/A',
    generator: 'gemini-2.5-flash-image'
  })}`;

  const explanationResponse = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: explanationPrompt
  });
  
  const now = new Date().toISOString();
  return {
    id: `rep_${Date.now()}`,
    fileName: file.name,
    ownerId: 'local-user',
    uploadPath: `local/${file.name}`,
    createdAt: now,
    status: 'generated',
    ...analysisResult,
    generatedImageId: `gen_${Date.now()}`,
    generatedImageUrl,
    xai: {
      ...analysisResult.xai,
      saliencyImageUrl: "https://via.placeholder.com/512.png?text=SALIENCY+MAP+(STUB)", // Placeholder
      provenance: {
        generator: 'gemini-2.5-flash-image',
        seed: Math.floor(Math.random() * 100000),
        conditioning: {
          organ: analysisResult.predictions.organ,
          condition: primaryCondition.label,
        },
        createdAt: now
      },
      naturalLanguageSummary: explanationResponse.text,
    },
    auditLog: [
      { user: 'System', action: 'Upload', timestamp: now },
      { user: 'System', action: 'Analysis complete', timestamp: now },
      { user: 'System', action: 'Image generated', timestamp: now },
    ]
  };
};