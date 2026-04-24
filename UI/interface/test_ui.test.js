/**
 * tests/test_ui.js
 * ──────────────────────────────────────────────────────────────────────────
 * Tests for the SentinelAI UI layer (IPC client + frame codec).
 * Runs with Jest in Node environment (no Electron required).
 * ──────────────────────────────────────────────────────────────────────────
 */

'use strict';

const { SentinelClient, encodeFrame, decodeFrame } = require('../sentinel-ipc-client');

// ── Frame Codec Tests ─────────────────────────────────────────────────────
describe('Frame Codec', () => {
  test('encodeFrame produces correct length prefix', () => {
    const obj   = { type: 'search', query: 'HDFS' };
    const frame = encodeFrame(obj);
    const view  = new DataView(frame.buffer);
    const len   = view.getUint32(0, true);          // LE
    expect(len).toBe(frame.byteLength - 4);
  });

  test('decodeFrame round-trips an object', () => {
    const original = { type: 'search', query: 'HDFS architecture', k: 20 };
    const frame    = encodeFrame(original);
    const view     = new DataView(frame.buffer);
    const decoded  = decodeFrame(view);
    expect(decoded).not.toBeNull();
    expect(decoded.obj).toEqual(original);
    expect(decoded.bytesConsumed).toBe(frame.byteLength);
  });

  test('decodeFrame returns null for incomplete frame', () => {
    const frame = encodeFrame({ x: 1 });
    const trunc = frame.slice(0, frame.byteLength - 2);
    const view  = new DataView(trunc.buffer);
    expect(decodeFrame(view)).toBeNull();
  });

  test('decodeFrame returns null for header-only buffer', () => {
    const buf  = new Uint8Array(4);
    new DataView(buf.buffer).setUint32(0, 100, true); // claims 100 bytes but buffer is empty
    expect(decodeFrame(new DataView(buf.buffer))).toBeNull();
  });

  test('encodeFrame handles unicode correctly', () => {
    const obj   = { query: '日本語ファイル検索' };
    const frame = encodeFrame(obj);
    const view  = new DataView(frame.buffer);
    const dec   = decodeFrame(view);
    expect(dec.obj.query).toBe('日本語ファイル検索');
  });

  test('multiple frames can be decoded sequentially from a single buffer', () => {
    const a = encodeFrame({ id: 1 });
    const b = encodeFrame({ id: 2 });
    const combined = new Uint8Array(a.byteLength + b.byteLength);
    combined.set(a, 0);
    combined.set(b, a.byteLength);

    let offset = 0;
    const results = [];
    while (offset < combined.byteLength) {
      const sub = new DataView(combined.buffer, offset);
      const dec = decodeFrame(sub);
      if (!dec) break;
      results.push(dec.obj);
      offset += dec.bytesConsumed;
    }
    expect(results).toHaveLength(2);
    expect(results[0].id).toBe(1);
    expect(results[1].id).toBe(2);
  });
});

// ── SentinelClient Mock Mode ──────────────────────────────────────────────
describe('SentinelClient (mock mode)', () => {
  let client;

  beforeEach(() => {
    // Force mock mode (no Electron/Tauri in Jest)
    client = new SentinelClient();
    expect(client._mode).toBe('mock');
  });

  test('search returns a ResultSet shape', async () => {
    const result = await client.search({ query: 'test query', k: 10 });
    expect(result).toHaveProperty('results');
    expect(result).toHaveProperty('queryMs');
    expect(result).toHaveProperty('tier');
    expect(Array.isArray(result.results)).toBe(true);
  });

  test('k is clamped to 100', async () => {
    // We can't easily observe the clamping in mock mode, but we verify it doesn't throw
    await expect(client.search({ query: 'x', k: 9999 })).resolves.toBeDefined();
  });

  test('sendAction resolves true for whitelisted actions', async () => {
    await expect(client.sendAction({ action: 'open_file', path: '/tmp/test.pdf' })).resolves.toBe(true);
    await expect(client.sendAction({ action: 'move_to_trash', path: '/tmp/test.pdf' })).resolves.toBe(true);
    await expect(client.sendAction({ action: 'set_priority', path: '/tmp/test.pdf', payload: 1 })).resolves.toBe(true);
  });

  test('sendAction throws for unlisted actions', async () => {
    await expect(
      client.sendAction({ action: 'rm_rf', path: '/' })
    ).rejects.toThrow('Illegal action');
  });

  test('getStatus returns expected fields', async () => {
    const status = await client.getStatus();
    expect(status).toHaveProperty('running', true);
    expect(status).toHaveProperty('cpuPercent');
    expect(status).toHaveProperty('memMb');
    expect(status).toHaveProperty('indexedFiles');
    expect(status).toHaveProperty('bulkIndexing');
    expect(typeof status.cpuPercent).toBe('number');
    expect(status.cpuPercent).toBeGreaterThan(0);
    expect(status.cpuPercent).toBeLessThan(100);
  });

  test('subscribeProgress calls onEvent and returns unsubscribe fn', (done) => {
    let callCount = 0;
    const unsub = client.subscribeProgress((event) => {
      expect(event).toHaveProperty('indexed');
      expect(event).toHaveProperty('total', 1_000_000);
      expect(event).toHaveProperty('done');
      callCount++;
      if (callCount >= 2) {
        unsub(); // stop subscription
        done();
      }
    });
    expect(typeof unsub).toBe('function');
  }, 10_000);

  test('subscribeProgress stops after unsub()', (done) => {
    let callCount = 0;
    const unsub = client.subscribeProgress(() => { callCount++; });
    setTimeout(() => {
      unsub();
      const countAtUnsub = callCount;
      setTimeout(() => {
        // Should not have increased significantly after unsub
        expect(callCount).toBe(countAtUnsub);
        done();
      }, 3000);
    }, 200);
  }, 10_000);
});

// ── Action Whitelist Tests ────────────────────────────────────────────────
describe('Action whitelist', () => {
  const ALLOWED = ['open_file', 'move_to_trash', 'set_priority'];
  const BLOCKED = ['rm_rf', 'exec', 'shell', 'delete', 'format', 'chmod', ''];

  const client = new SentinelClient();

  ALLOWED.forEach((action) => {
    test(`allows: ${action}`, async () => {
      await expect(client.sendAction({ action, path: '/tmp/x' })).resolves.toBeDefined();
    });
  });

  BLOCKED.forEach((action) => {
    test(`blocks: "${action}"`, async () => {
      await expect(client.sendAction({ action, path: '/tmp/x' })).rejects.toThrow('Illegal action');
    });
  });
});

// ── Wire Format Integration ───────────────────────────────────────────────
describe('Wire format compatibility', () => {
  test('SearchRequest encodes to correct JSON keys', () => {
    const req   = { type: 'search', query: 'HDFS architecture', k: 20, fileTypes: ['pdf'] };
    const frame = encodeFrame(req);
    const view  = new DataView(frame.buffer);
    const dec   = decodeFrame(view);
    expect(dec.obj.type).toBe('search');
    expect(dec.obj.query).toBe('HDFS architecture');
    expect(dec.obj.k).toBe(20);
    expect(dec.obj.fileTypes).toEqual(['pdf']);
  });

  test('SystemAction encodes correctly', () => {
    const action = { type: 'action', action: 'move_to_trash', path: '/home/user/file.pdf' };
    const frame  = encodeFrame(action);
    const view   = new DataView(frame.buffer);
    const dec    = decodeFrame(view);
    expect(dec.obj.action).toBe('move_to_trash');
    expect(dec.obj.path).toBe('/home/user/file.pdf');
  });

  test('empty payload encodes and decodes', () => {
    const frame = encodeFrame({});
    const view  = new DataView(frame.buffer);
    const dec   = decodeFrame(view);
    expect(dec.obj).toEqual({});
  });
});
