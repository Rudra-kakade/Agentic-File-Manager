/**
 * electron-preload.js
 * ──────────────────────────────────────────────────────────────────────────
 * Runs in the renderer process's isolated world.
 *
 * Exposes window.__sentinelBridge — a minimal, safe surface for the renderer
 * to communicate with the main process (and through it, the Unix sockets).
 *
 * Only whitelisted channels are exposed. No Node APIs leak into the renderer.
 * ──────────────────────────────────────────────────────────────────────────
 */

'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('__sentinelBridge', {
  /**
   * Send a typed frame to a Unix socket and receive a response.
   * @param {string}     socketPath  e.g. '/run/sentinel/orchestrator.sock'
   * @param {Uint8Array} frameBytes  Already-encoded length-prefixed frame
   * @returns {Promise<Uint8Array>}  Raw response frame bytes
   */
  send: (socketPath, frameBytes) =>
    ipcRenderer.invoke('sentinel:rpc', socketPath, frameBytes),

  /**
   * Begin streaming ProgressEvents from the daemon.
   * Calls onFrame for each event; returns an unsubscribe function.
   * @param {string}   socketPath
   * @param {Function} onFrame
   * @returns {() => void}
   */
  subscribe: (socketPath, onFrame) => {
    ipcRenderer.invoke('sentinel:subscribe_progress');
    const handler = (_event, frame) => onFrame(frame);
    ipcRenderer.on('sentinel:progress', handler);
    return () => ipcRenderer.off('sentinel:progress', handler);
  },

  /**
   * Fetch current daemon status (CPU, mem, file count, indexing state).
   * @returns {Promise<import('./sentinel-ipc-client').DaemonStatus>}
   */
  status: () => ipcRenderer.invoke('sentinel:status'),
});
