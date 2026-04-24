/**
 * electron-main.js
 * ──────────────────────────────────────────────────────────────────────────
 * Electron main process for SentinelAI.
 *
 * Responsibilities:
 *   - Create the BrowserWindow
 *   - Establish persistent Unix socket connections to sentinel services
 *   - Bridge renderer ↔ orchestrator IPC via ipcMain
 *   - Register system tray icon + native menu
 *   - Forward OS-level keyboard shortcut (⌘K / Ctrl+K) to focus window
 *
 * Run: electron electron-main.js
 * ──────────────────────────────────────────────────────────────────────────
 */

'use strict';

const { app, BrowserWindow, Tray, Menu, globalShortcut, nativeImage, ipcMain } = require('electron');
const path   = require('path');
const net    = require('net');
const os     = require('os');
const fs     = require('fs');

// ── Config ────────────────────────────────────────────────────────────────
const SOCKET_DIR       = '/run/sentinel';
const ORCHESTRATOR_SOCK = path.join(SOCKET_DIR, 'orchestrator.sock');
const DAEMON_SOCK       = path.join(SOCKET_DIR, 'daemon.sock');
const PRELOAD_PATH      = path.join(__dirname, 'electron-preload.js');
const INDEX_HTML        = path.join(__dirname, 'index.html');
const TRAY_ICON         = path.join(__dirname, 'assets', 'tray-icon.png');
const FRAME_HEADER      = 4; // bytes — LE uint32 length prefix

// ── Window Reference ──────────────────────────────────────────────────────
let mainWindow = null;
let tray       = null;

// ── App Lifecycle ─────────────────────────────────────────────────────────
app.whenReady().then(() => {
  createWindow();
  createTray();
  registerShortcuts();
});

app.on('window-all-closed', () => {
  // Keep process alive (tray app) — don't quit on window close
});

app.on('activate', () => {
  // macOS dock click — show window if hidden
  if (mainWindow) mainWindow.show();
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
});

// ── Window Creation ───────────────────────────────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    width:           1200,
    height:          720,
    minWidth:        700,
    minHeight:       440,
    backgroundColor: '#0B0C0E',
    titleBarStyle:   'hiddenInset',    // macOS native traffic lights
    frame:           process.platform !== 'darwin', // Windows/Linux: system frame
    show:            false,
    webPreferences: {
      preload:             PRELOAD_PATH,
      contextIsolation:    true,
      nodeIntegration:     false,
      sandbox:             true,
    },
  });

  mainWindow.loadFile(INDEX_HTML);

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  mainWindow.on('close', (e) => {
    // Hide instead of quit so indexing continues in background
    e.preventDefault();
    mainWindow.hide();
  });
}

// ── System Tray ───────────────────────────────────────────────────────────
function createTray() {
  const icon = fs.existsSync(TRAY_ICON)
    ? nativeImage.createFromPath(TRAY_ICON).resize({ width: 16, height: 16 })
    : nativeImage.createEmpty();

  tray = new Tray(icon);
  tray.setToolTip('SentinelAI');

  const menu = Menu.buildFromTemplate([
    { label: 'Open SentinelAI', click: showWindow },
    { type: 'separator' },
    {
      label: 'Services',
      submenu: [
        { label: 'sentinel-daemon',       enabled: false },
        { label: 'sentinel-embedding',    enabled: false },
        { label: 'sentinel-vectorstore',  enabled: false },
        { label: 'sentinel-orchestrator', enabled: false },
      ],
    },
    { type: 'separator' },
    { label: 'Quit SentinelAI', role: 'quit' },
  ]);

  tray.setContextMenu(menu);
  tray.on('double-click', showWindow);
}

function showWindow() {
  if (!mainWindow) return;
  mainWindow.show();
  mainWindow.focus();
}

// ── Global Shortcut ───────────────────────────────────────────────────────
function registerShortcuts() {
  // ⌘Space equivalent — summon from anywhere
  globalShortcut.register('CommandOrControl+Shift+Space', showWindow);
}

// ── Unix Socket IPC Bridge ────────────────────────────────────────────────
/**
 * Send a length-prefixed JSON frame to a Unix socket and await a response.
 * Returns a Buffer with the raw response frame.
 * @param {string} socketPath
 * @param {Buffer} frameBytes
 * @returns {Promise<Buffer>}
 */
function unixRPC(socketPath, frameBytes) {
  return new Promise((resolve, reject) => {
    const client = net.createConnection(socketPath);
    const chunks = [];

    client.setTimeout(8_000);

    client.on('connect', () => {
      client.write(frameBytes);
    });

    client.on('data', (chunk) => {
      chunks.push(chunk);
      // Check if we have a complete frame
      const buf = Buffer.concat(chunks);
      if (buf.length >= FRAME_HEADER) {
        const len = buf.readUInt32LE(0);
        if (buf.length >= FRAME_HEADER + len) {
          client.destroy();
          resolve(buf);
        }
      }
    });

    client.on('timeout', () => {
      client.destroy(new Error('Socket timeout'));
    });

    client.on('error', reject);
  });
}

/**
 * Subscribe to a streaming Unix socket (for ProgressEvents).
 * Calls onFrame for each complete frame received.
 * Returns a function that closes the socket.
 * @param {string} socketPath
 * @param {(frame: object) => void} onFrame
 * @returns {() => void}
 */
function unixSubscribe(socketPath, onFrame) {
  const client = net.createConnection(socketPath);
  let buf = Buffer.alloc(0);

  // Send subscribe request
  const payload = JSON.stringify({ type: 'progress_subscribe' });
  const header  = Buffer.alloc(4);
  header.writeUInt32LE(Buffer.byteLength(payload, 'utf8'), 0);
  client.on('connect', () => client.write(Buffer.concat([header, Buffer.from(payload, 'utf8')])));

  client.on('data', (chunk) => {
    buf = Buffer.concat([buf, chunk]);
    while (buf.length >= FRAME_HEADER) {
      const len = buf.readUInt32LE(0);
      if (buf.length < FRAME_HEADER + len) break;
      const json = buf.slice(FRAME_HEADER, FRAME_HEADER + len).toString('utf8');
      buf = buf.slice(FRAME_HEADER + len);
      try { onFrame(JSON.parse(json)); } catch { /* ignore parse errors */ }
    }
  });

  client.on('error', (err) => {
    console.error('[sentinel-bridge] subscribe error:', err.message);
    setTimeout(() => unixSubscribe(socketPath, onFrame), 3_000); // reconnect
  });

  return () => client.destroy();
}

// ── IPC Main Handlers ─────────────────────────────────────────────────────
// These are invoked by the preload's contextBridge.

ipcMain.handle('sentinel:rpc', async (_event, socketPath, frameBytes) => {
  const buf = await unixRPC(socketPath, Buffer.from(frameBytes));
  // Return as Uint8Array so it can cross the context bridge
  return new Uint8Array(buf);
});

ipcMain.handle('sentinel:status', async () => {
  const payload = JSON.stringify({ type: 'status' });
  const header  = Buffer.alloc(4);
  header.writeUInt32LE(Buffer.byteLength(payload, 'utf8'), 0);
  const frame = Buffer.concat([header, Buffer.from(payload, 'utf8')]);
  const resp  = await unixRPC(DAEMON_SOCK, frame);
  const len   = resp.readUInt32LE(0);
  return JSON.parse(resp.slice(FRAME_HEADER, FRAME_HEADER + len).toString('utf8'));
});

// Progress stream: relay from daemon socket to renderer via WebContents
ipcMain.handle('sentinel:subscribe_progress', (_event) => {
  const unsub = unixSubscribe(DAEMON_SOCK, (frame) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('sentinel:progress', frame);
    }
  });
  // Store so we can clean up on reload
  mainWindow.once('closed', unsub);
});
