import { expect, test, vi } from 'vitest';
import { RunSocket } from './socket';

class MockWebSocket {
  public static instances: MockWebSocket[] = [];
  public url: string;
  public readyState: number = 0;
  public binaryType: BinaryType = 'blob';
  public sentData: string[] = [];

  public onopen: (() => void) | null = null;
  public onclose: ((event: { code: number; reason: string; wasClean: boolean }) => void) | null = null;
  public onerror: ((error: unknown) => void) | null = null;
  public onmessage: ((event: { data: unknown }) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  public send(data: string) {
    this.sentData.push(data);
  }

  public close() {
    this.readyState = 3; // CLOSED
    this.onclose?.({ code: 1000, reason: 'Normal closure', wasClean: true });
  }

  public simulateOpen() {
    this.readyState = 1; // OPEN
    this.onopen?.();
  }

  public simulateTextMessage(data: string | object) {
    const raw = typeof data === 'string' ? data : JSON.stringify(data);
    this.onmessage?.({ data: raw });
  }

  public simulateBinaryMessage(buffer: ArrayBuffer) {
    this.onmessage?.({ data: buffer });
  }
}

test('RunSocket acks frames with seq and ignores messages without seq', () => {
  MockWebSocket.instances = [];
  const onText = vi.fn();
  const onError = vi.fn();

  new RunSocket('r_1234567890abcdef', {
    wsUrl: 'ws://localhost/api/runs/r_1234567890abcdef/stream',
    WebSocketImpl: MockWebSocket as unknown as typeof WebSocket,
    onText,
    onError,
  });

  const mockWs = MockWebSocket.instances[0];
  mockWs.simulateOpen();

  // Text message with seq
  mockWs.simulateTextMessage({ type: 'hello', seq: 10 });
  expect(mockWs.sentData).toContain(JSON.stringify({ type: 'ack', seq: 10 }));

  // Text message without seq
  mockWs.simulateTextMessage({ type: 'ping' });
  const ackCount = mockWs.sentData.filter(d => d.includes('"type":"ack"')).length;
  expect(ackCount).toBe(1);

  const callCountBeforeMalformed = onText.mock.calls.length;
  // Malformed JSON message
  mockWs.simulateTextMessage('invalid-json-content{');
  expect(onError).toHaveBeenCalled();
  expect(onText.mock.calls.length).toBe(callCountBeforeMalformed);
});
