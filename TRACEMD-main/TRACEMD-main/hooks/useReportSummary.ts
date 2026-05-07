import { useState, useEffect } from 'react';
import type { ReportData, ExplanationMeta } from '../types';

// Dummy hook to match the prompt's structure for state management.
// In a real Firebase app, this would fetch from Firestore and handle real API calls.
export const useReportSummary = (initialData: ReportData) => {
  const [data, setData] = useState<ReportData | null>(initialData);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setData(initialData);
  }, [initialData]);

  const updateStatus = (newStatus: 'approved' | 'rejected') => {
    if (data) {
        const newAuditLog = [...data.auditLog, {
            user: 'Clinician',
            action: `Result ${newStatus}`,
            timestamp: new Date().toISOString()
        }];
        setData({ ...data, status: newStatus, auditLog: newAuditLog });
        console.log(`Report ${data.id} status updated to: ${newStatus}`);
    }
  };

  const regenerate = async () => {
    if (!data) return;
    setLoading(true);
    setError(null);
    try {
        // This is a MOCK fetch call to a cloud function as requested.
        // It will not actually work but demonstrates the client-side implementation.
        console.log('Simulating call to /regenerate endpoint...');
        // const response = await fetch('/regenerate', {
        //     method: 'POST',
        //     headers: { 'Content-Type': 'application/json' },
        //     body: JSON.stringify({
        //         reportId: data.id,
        //         imageId: data.generatedImageId,
        //         userId: data.ownerId
        //     })
        // });
        // if (!response.ok) throw new Error('Regeneration request failed.');
        
        // Simulate network delay
        await new Promise(resolve => setTimeout(resolve, 1500));

        // Mock response: update image URL and add to audit log
        const newImageUrl = `https://via.placeholder.com/512.png?text=REGENERATED+${Date.now()}`;
        const newAuditLog = [...data.auditLog, {
            user: 'Clinician',
            action: 'Regeneration requested',
            timestamp: new Date().toISOString()
        }];

        setData({ ...data, generatedImageUrl: newImageUrl, auditLog: newAuditLog });
        console.log(`Image for report ${data.id} regenerated.`);

    } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred.';
        setError(errorMessage);
        console.error('Regeneration failed:', errorMessage);
    } finally {
        setLoading(false);
    }
  };

  const saveExplanation = (explanation: string, meta: Omit<ExplanationMeta, 'savedAt' | 'authorId'>) => {
    if (data) {
      const newMeta: ExplanationMeta = {
        ...meta,
        authorId: 'Clinician',
        savedAt: new Date().toISOString(),
      };
      const newAuditLog = [...data.auditLog, {
        user: 'Clinician',
        action: 'Saved explanation',
        timestamp: newMeta.savedAt,
        details: `Tone: ${meta.tone}`
      }];
      setData({ ...data, explanation, explanationMeta: newMeta, auditLog: newAuditLog });
      console.log(`Explanation saved for report ${data.id}.`);
    }
  };


  return { data, loading, error, updateStatus, regenerate, saveExplanation };
};
