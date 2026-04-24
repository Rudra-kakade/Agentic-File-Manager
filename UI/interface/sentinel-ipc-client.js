/**
 * sentinel-ipc-client.js
 * ──────────────────────────────────────────────────────────────────────────
 * Typed IPC client for the SentinelAI UI layer.
 *
 * In Electron: uses Node.js `net` module via the preload bridge.
 * In Tauri:    uses the `invoke` command bridge to Rust IPC helpers.
 * In browser:  stub/mock mode (used during development).
 *
 * Wire protocol (same as all sentinel services):
 *   [ 4-byte LE uint32 length ][ UTF-8 JSON payload ]
 *
 * All public methods are async and throw on transport failure.
 * The caller is responsible for fallback / error display.
 * ──────────────────────────────────────────────────────────────────────────
 */

// ── Constants ────────────────────────────────────────────────────────────
const SOCKET_PATH    = '/run/sentinel/orchestrator.sock';
const CONNECT_TIMEOUT_MS = 5_000;
const MAX_RESULTS    = 20;
const FRAME_HEADER   = 4; // bytes

// ── Transport Detection ───────────────────────────────────────────────────
const IS_ELECTRON = typeof window !== 'undefined' && window.__sentinelBridge !== undefined;
const IS_TAURI    = typeof window !== 'undefined' && window.__TAURI__        !== undefined;
const IS_BROWSER  = !IS_ELECTRON && !IS_TAURI;

// ── Type Definitions (JSDoc) ─────────────────────────────────────────────
/**
 * @typedef {Object} SearchRequest
 * @property {string}   query       Natural-language query string
 * @property {number}   k           Max results to return (1–100)
 * @property {string[]} [fileTypes] Optional extension filter: ["pdf","docx",…]
 */

/**
 * @typedef {Object} SearchResult
 * @property {string} path          Absolute filesystem path
 * @property {string} name          Filename
 * @property {number} score         Normalised similarity [0,1]
 * @property {number} modifiedAt    Unix timestamp (seconds)
 * @property {number} createdAt     Unix timestamp (seconds)
 * @property {number} sizeBytes     File size in bytes
 * @property {string} tier          "graph+vector" | "vector" | "bm25"
 * @property {string} [snippet]     Extracted content preview (may be empty)
 */

/**
 * @typedef {Object} ResultSet
 * @property {SearchResult[]} results
 * @property {number}         queryMs    Total server-side query time
 * @property {string}         tier       Highest tier reached
 */

/**
 * @typedef {Object} SystemAction
 * @property {"open_file"|"move_to_trash"|"set_priority"} action
 * @property {string} path
 * @property {*}      [payload]   Action-specific extra data
 */

/**
 * @typedef {Object} ProgressEvent
 * @property {number} indexed     Files indexed so far
 * @property {number} total       Estimated total (0 = unknown)
 * @property {string} currentPath Last path processed
 * @property {boolean} done       True when bulk indexing is complete
 */

/**
 * @typedef {Object} DaemonStatus
 * @property {boolean} running
 * @property {number}  cpuPercent    System-wide CPU usage (%)
 * @property {number}  memMb         Daemon RSS in MB
 * @property {number}  indexedFiles  Total files in graph
 * @property {boolean} bulkIndexing  True while BulkIndexer is active
 */

// ── Frame Codec ───────────────────────────────────────────────────────────
/**
 * Encode a JS object to a length-prefixed JSON frame (Uint8Array).
 * @param {object} obj
 * @returns {Uint8Array}
 */
function encodeFrame(obj) {
  const json  = JSON.stringify(obj);
  const body  = new TextEncoder().encode(json);
  const frame = new Uint8Array(FRAME_HEADER + body.byteLength);
  const view  = new DataView(frame.buffer);
  view.setUint32(0, body.byteLength, /* littleEndian= */ true);
  frame.set(body, FRAME_HEADER);
  return frame;
}

/**
 * Decode a length-prefixed JSON frame from a DataView.
 * @param {DataView} view
 * @returns {{ obj: object, bytesConsumed: number } | null}
 */
function decodeFrame(view) {
  if (view.byteLength < FRAME_HEADER) return null;
  const len = view.getUint32(0, true);
  if (view.byteLength < FRAME_HEADER + len) return null;
  const bytes  = new Uint8Array(view.buffer, view.byteOffset + FRAME_HEADER, len);
  const obj    = JSON.parse(new TextDecoder().decode(bytes));
  return { obj, bytesConsumed: FRAME_HEADER + len };
}

// ── Electron Preload Contract ─────────────────────────────────────────────
/**
 * Expected shape of window.__sentinelBridge (set in preload.js):
 *
 *   window.__sentinelBridge = {
 *     send: async (socketPath, frameBytes) => responseBytes,
 *     subscribe: (socketPath, onFrame) => unsubscribeFn,
 *   };
 */

// ── Tauri Command Contract ────────────────────────────────────────────────
/**
 * Expected Tauri commands (defined in src-tauri/src/ipc.rs):
 *
 *   sentinel_query(payload: SentinelRequest) -> SentinelResponse
 *   sentinel_action(action: SystemAction) -> bool
 *   sentinel_status() -> DaemonStatus
 *   sentinel_subscribe_progress() -> UnlistenFn (event: ProgressEvent)
 */

// ── Core Client Class ─────────────────────────────────────────────────────
class SentinelClient {
  constructor() {
    this._mode = IS_ELECTRON ? 'electron' : IS_TAURI ? 'tauri' : 'mock';
    console.info(`[SentinelClient] mode=${this._mode}`);
  }

  // ── Search ────────────────────────────────────────────────────────────
  /**
   * Send a natural-language query to the orchestrator and return results.
   * @param {SearchRequest} req
   * @returns {Promise<ResultSet>}
   */
  async search(req) {
    if (req.k === undefined) req.k = MAX_RESULTS;
    req.k = Math.min(req.k, 100); // server clamps at 100 anyway

    switch (this._mode) {
      case 'electron': return this._electronRPC({ type: 'search', ...req });
      case 'tauri':    return this._tauriSearch(req);
      default:         return this._mockSearch(req);
    }
  }

  // ── System Action ─────────────────────────────────────────────────────
  /**
   * Send a whitelisted system action to the daemon.
   * @param {SystemAction} action
   * @returns {Promise<boolean>}   true = success
   */
  async sendAction(action) {
    if (!['open_file', 'move_to_trash', 'set_priority'].includes(action.action)) {
      throw new Error(`Illegal action: ${action.action}`);
    }

    switch (this._mode) {
      case 'electron': return this._electronRPC({ type: 'action', ...action });
      case 'tauri':    return window.__TAURI__.invoke('sentinel_action', { action });
      default:         return this._mockAction(action);
    }
  }

  // ── Status ────────────────────────────────────────────────────────────
  /**
   * Fetch current daemon status (CPU, memory, file count, indexing state).
   * @returns {Promise<DaemonStatus>}
   */
  async getStatus() {
    switch (this._mode) {
      case 'electron': return this._electronRPC({ type: 'status' });
      case 'tauri':    return window.__TAURI__.invoke('sentinel_status');
      default:         return this._mockStatus();
    }
  }

  // ── Progress Stream ───────────────────────────────────────────────────
  /**
   * Subscribe to BulkIndexer ProgressEvents.
   * Calls onEvent for each event; returns an unsubscribe function.
   * @param {(event: ProgressEvent) => void} onEvent
   * @returns {() => void}   Call to stop listening.
   */
  subscribeProgress(onEvent) {
    switch (this._mode) {
      case 'electron': return this._electronSubscribe({ type: 'progress' }, onEvent);
      case 'tauri':    return this._tauriSubscribeProgress(onEvent);
      default:         return this._mockSubscribeProgress(onEvent);
    }
  }

  // ── Electron Transport ────────────────────────────────────────────────
  async _electronRPC(payload) {
    const frame = encodeFrame(payload);
    const respBytes = await window.__sentinelBridge.send(SOCKET_PATH, frame);
    const view  = new DataView(respBytes.buffer);
    const frame2 = decodeFrame(view);
    if (!frame2) throw new Error('Malformed response frame');
    if (frame2.obj.error) throw new Error(frame2.obj.error);
    return frame2.obj;
  }

  _electronSubscribe(initPayload, onEvent) {
    return window.__sentinelBridge.subscribe(SOCKET_PATH, (bytes) => {
      const view = new DataView(bytes.buffer);
      let offset = 0;
      while (offset < bytes.byteLength) {
        const sub = new DataView(bytes.buffer, offset);
        const decoded = decodeFrame(sub);
        if (!decoded) break;
        onEvent(decoded.obj);
        offset += decoded.bytesConsumed;
      }
    });
  }

  // ── Tauri Transport ───────────────────────────────────────────────────
  async _tauriSearch(req) {
    return window.__TAURI__.invoke('sentinel_query', { payload: req });
  }

  _tauriSubscribeProgress(onEvent) {
    let unlisten;
    window.__TAURI__.event.listen('sentinel://progress', (ev) => onEvent(ev.payload))
      .then(fn => { unlisten = fn; });
    return () => unlisten && unlisten();
  }

  // ── Mock / Dev Transport ──────────────────────────────────────────────
  async _mockSearch(req) {
    await _delay(350 + Math.random() * 120);
    return {
      results: [],      // Populated by the UI's own mock data
      queryMs: Math.round(350 + Math.random() * 120),
      tier: 'graph+vector',
    };
  }

  async _mockAction(action) {
    await _delay(200);
    console.info('[mock] action:', action);
    return true;
  }

  async _mockStatus() {
    return {
      running:      true,
      cpuPercent:   Math.round(5 + Math.random() * 8),
      memMb:        1412,
      indexedFiles: 847293,
      bulkIndexing: true,
    };
  }

  _mockSubscribeProgress(onEvent) {
    let count = 847293;
    const id = setInterval(() => {
      count += Math.floor(Math.random() * 500 + 100);
      const done = count >= 1_000_000;
      onEvent({ indexed: Math.min(count, 1_000_000), total: 1_000_000, currentPath: '/home/user/...', done });
      if (done) clearInterval(id);
    }, 1800);
    return () => clearInterval(id);
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────
function _delay(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Singleton Export ──────────────────────────────────────────────────────
const sentinel = new SentinelClient();

// Browser globals for index.html (non-module context)
if (typeof window !== 'undefined') {
  window.sentinel = sentinel;
}

// ES module export (Electron preload / Tauri renderer)
if (typeof module !== 'undefined') {
  module.exports = { SentinelClient, sentinel, encodeFrame, decodeFrame };
}
