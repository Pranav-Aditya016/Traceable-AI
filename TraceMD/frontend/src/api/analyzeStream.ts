import type { ReportData } from '../types';

export type StageEvent = {
  stage:    number;
  status:   'running' | 'complete' | 'error';
  label:    string;
  result:   string;
  payload?: ReportData | { input_type: string };
};

/**
 * SSE stream client for the /api/analyze endpoint.
 * Returns an abort function to cancel the stream.
 */
export function analyzeStream(
  file: File,
  onStageEvent: (event: StageEvent) => void,
  onComplete:   (report: ReportData) => void,
  onError:      (message: string) => void
): () => void {
  const formData = new FormData();
  formData.append('file', file);

  const controller = new AbortController();

  fetch('http://localhost:8000/api/analyze', {
    method: 'POST',
    body:   formData,
    signal: controller.signal,
  }).then(async (response) => {
    if (!response.ok) {
      const text = await response.text().catch(() => 'Unknown error');
      onError(`Server error ${response.status}: ${text}`);
      return;
    }

    if (!response.body) {
      onError('No response body from server.');
      return;
    }

    const reader  = response.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by double-newline
      const normalized = buffer.replace(/\r\n/g, '\n');
      const blocks = normalized.split('\n\n');
      buffer = blocks.pop() ?? '';

      for (const block of blocks) {
        if (!block.trim()) continue;

        // Extract data from SSE block
        let dataStr = '';
        for (const line of block.split('\n')) {
          if (line.startsWith('data:')) {
            dataStr += line.slice(5).trim();
          }
        }

        if (!dataStr) continue;

        try {
          const event: StageEvent = JSON.parse(dataStr);
          onStageEvent(event);

          // Check for final payload (stage 6 complete)
          if (event.stage === 6 && event.status === 'complete' && event.payload) {
            onComplete(event.payload as ReportData);
          }

          // Check for pipeline error
          if (event.status === 'error') {
            onError(event.result || 'Unknown pipeline error');
          }
        } catch (e) {
          console.warn('Failed to parse SSE event:', dataStr.slice(0, 200));
        }
      }
    }
  }).catch((err) => {
    if (err.name !== 'AbortError') {
      onError(`Network error: ${err.message}`);
    }
  });

  return () => controller.abort();
}
