import { decodeFrame, Frame } from './frames';

export interface RunSocketOptions {
  wsUrl?: string;
  token?: string;
  WebSocketImpl?: typeof WebSocket;
  onOpen?: () => void;
  onClose?: (event: { code: number; reason: string; wasClean: boolean }) => void;
  onError?: (error: unknown) => void;
  onText?: (message: unknown) => void;
  onFrame?: (frame: Frame) => void;
}

export class RunSocket {
  private ws: WebSocket | null = null;
  private closedExplicitly = false;

  constructor(runId: string, options: RunSocketOptions = {}) {
    const {
      wsUrl,
      token,
      WebSocketImpl = WebSocket,
      onOpen,
      onClose,
      onError,
      onText,
      onFrame,
    } = options;

    let url = wsUrl;
    if (!url) {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.host;
      url = `${proto}//${host}/api/runs/${runId}/stream`;
      if (token) {
        url += `?token=${encodeURIComponent(token)}`;
      }
    }

    this.ws = new WebSocketImpl(url);
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      onOpen?.();
    };

    this.ws.onclose = (event) => {
      onClose?.({
        code: event.code,
        reason: event.reason,
        wasClean: event.wasClean,
      });
    };

    this.ws.onerror = (error) => {
      onError?.(error);
    };

    this.ws.onmessage = (event) => {
      try {
        if (event.data instanceof ArrayBuffer) {
          const frame = decodeFrame(event.data);
          if (typeof frame.seq === 'number') {
            this.sendAck(frame.seq);
          }
          onFrame?.(frame);
        } else if (typeof event.data === 'string') {
          let msg: unknown;
          try {
            msg = JSON.parse(event.data);
          } catch (parseErr) {
            onError?.(parseErr);
            return;
          }
          if (msg && typeof msg === 'object') {
            const m = msg as Record<string, unknown>;
            if (typeof m.seq === 'number') {
              this.sendAck(m.seq);
            }
          }
          onText?.(msg);
        }
      } catch (err) {
        onError?.(err);
      }
    };
  }

  public sendAck(seq: number): void {
    this.send({ type: 'ack', seq });
  }

  public sendPace(mode: string, speed: number): void {
    this.send({ type: 'pace', mode, speed });
  }

  public sendSelect(group: string, indices: number[]): void {
    this.send({ type: 'select', group, indices });
  }

  public send(data: unknown): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  public close(): void {
    this.closedExplicitly = true;
    if (this.ws) {
      this.ws.close();
    }
  }

  public get readyState(): number {
    return this.ws ? this.ws.readyState : WebSocket.CLOSED;
  }
}
