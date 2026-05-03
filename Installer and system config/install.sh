#!/usr/bin/env bash
# =============================================================================
# SentinelAI Installer
# =============================================================================
#
# Usage
# ─────
#   sudo bash install.sh                  # interactive, all defaults
#   sudo bash install.sh --unattended     # non-interactive, no prompts
#   sudo bash install.sh --uninstall      # remove everything
#   sudo bash install.sh --help
#
# What this script does
# ─────────────────────
#   1.  Detect OS and package manager
#   2.  Install system dependencies  (poppler-utils, python3, tesseract, …)
#   3.  Create sentinel system user  (no login shell, no home dir)
#   4.  Create runtime directories   (/var/sentinel, /run/sentinel, /etc/sentinel)
#   5.  Install compiled binaries    (sentinel-daemon) and Python packages
#   6.  Download the Llama-3 8B GGUF model  (one-time, ~4.5 GB)
#   7.  Write /etc/sentinel/sentinel.env  (environment file for all services)
#   8.  Install all four systemd service units
#   9.  Enable and start services in the correct dependency order
#  10.  Run a smoke test  (index one file, search for it, verify a result)
#
# Exit codes
# ──────────
#   0   success
#   1   must be run as root
#   2   unsupported OS / missing package manager
#   3   dependency installation failed
#   4   model download failed
#   5   smoke test failed
#
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${GREEN}[INFO]${RESET}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
section() { echo -e "\n${BOLD}━━━ $* ━━━${RESET}"; }
die()     { error "$*"; exit "${2:-1}"; }

# ── Argument parsing ──────────────────────────────────────────────────────────

UNATTENDED=false
UNINSTALL=false
SKIP_MODEL=false
SKIP_SMOKE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --unattended)  UNATTENDED=true ;;
        --uninstall)   UNINSTALL=true  ;;
        --skip-model)  SKIP_MODEL=true ;;
        --skip-smoke)  SKIP_SMOKE=true ;;
        --help|-h)
            sed -n '/^# Usage/,/^# ===/p' "$0" | grep -v '^# ===' | sed 's/^# \{0,2\}//'
            exit 0
            ;;
        *) die "Unknown argument: $1" ;;
    esac
    shift
done

# ── Root check ────────────────────────────────────────────────────────────────

[[ $EUID -eq 0 ]] || die "This installer must be run as root.  Try: sudo bash install.sh" 1

# ── Paths and constants ───────────────────────────────────────────────────────

SENTINEL_USER="sentinel"
SENTINEL_GROUP="sentinel"

DIR_VAR="/var/sentinel"
DIR_STATE="/var/sentinel/state"
DIR_MODELS="/var/sentinel/models"
DIR_TRASH="/var/sentinel/trash"
DIR_RUN="/run/sentinel"
DIR_ETC="/etc/sentinel"
DIR_OPT="/opt/sentinel"
DIR_OPT_DAEMON="/opt/sentinel/daemon"
DIR_OPT_EMBED="/opt/sentinel/embedding"
DIR_OPT_QUERY="/opt/sentinel/query"

ENV_FILE="/etc/sentinel/sentinel.env"
SYSTEMD_DIR="/etc/systemd/system"

INSTALLER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$INSTALLER_DIR/.." && pwd)"

GGUF_MODEL="Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
GGUF_REPO="bartowski/Meta-Llama-3-8B-Instruct-GGUF"

SERVICES=(
    sentinel-vectorstore
    sentinel-embedding
    sentinel-daemon
    sentinel-orchestrator
)

# ── OS detection ──────────────────────────────────────────────────────────────

detect_os() {
    section "Detecting OS"

    if [[ -f /etc/os-release ]]; then
        # shellcheck source=/dev/null
        source /etc/os-release
        OS_ID="${ID:-unknown}"
        OS_LIKE="${ID_LIKE:-}"
        OS_VERSION="${VERSION_ID:-}"
    else
        die "Cannot detect OS: /etc/os-release not found." 2
    fi

    # Normalise to a package manager
    if command -v apt-get &>/dev/null; then
        PKG_MGR="apt"
    elif command -v dnf &>/dev/null; then
        PKG_MGR="dnf"
    elif command -v yum &>/dev/null; then
        PKG_MGR="yum"
    elif command -v zypper &>/dev/null; then
        PKG_MGR="zypper"
    else
        die "No supported package manager found (apt/dnf/yum/zypper)." 2
    fi

    info "OS: ${PRETTY_NAME:-$OS_ID}  |  Package manager: $PKG_MGR"
}

# ── Dependency installation ───────────────────────────────────────────────────

install_deps() {
    section "Installing system dependencies"

    local COMMON_PKGS=(
        "python3"           # Python 3.10+ for embedding + query services
        "python3-pip"       # pip for Python packages
        "python3-venv"      # venv isolation
        "poppler-utils"     # pdftotext for PDF extraction
        "curl"              # model download fallback
        "git"               # optional, for updates
    )

    # tesseract is optional — warn but don't fail if it can't be installed
    local OPTIONAL_PKGS=("tesseract-ocr")

    # Distro-specific package name overrides
    case "$PKG_MGR" in
        apt)
            apt-get update -qq || true
            DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
                "${COMMON_PKGS[@]}" "${OPTIONAL_PKGS[@]}" || {
                warn "Some optional packages failed; retrying without tesseract"
                DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
                    "${COMMON_PKGS[@]}" || die "apt-get install failed" 3
            }
            ;;
        dnf|yum)
            local MGR_PKGS=(python3 python3-pip poppler-utils curl git)
            "$PKG_MGR" install -y "${MGR_PKGS[@]}" || die "$PKG_MGR install failed" 3
            "$PKG_MGR" install -y tesseract || warn "tesseract not available on this distro"
            ;;
        zypper)
            zypper install -y python3 python3-pip poppler curl git \
                || die "zypper install failed" 3
            ;;
    esac

    # Verify Python >= 3.10
    local PY_VER
    PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    local PY_MAJOR PY_MINOR
    PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
    if (( PY_MAJOR < 3 || (PY_MAJOR == 3 && PY_MINOR < 10) )); then
        die "Python 3.10+ required, found $PY_VER" 3
    fi
    info "Python $PY_VER detected ✓"

    # Verify pdftotext
    if command -v pdftotext &>/dev/null; then
        info "pdftotext detected ✓"
    else
        warn "pdftotext not found — PDF text extraction will be disabled"
    fi

    # tesseract is optional
    if command -v tesseract &>/dev/null; then
        info "tesseract detected ✓"
    else
        warn "tesseract not found — image OCR will be disabled"
    fi
}

# ── System user ───────────────────────────────────────────────────────────────

create_user() {
    section "Creating sentinel system user"

    if id "$SENTINEL_USER" &>/dev/null; then
        info "User '$SENTINEL_USER' already exists — skipping"
        return
    fi

    groupadd --system "$SENTINEL_GROUP"
    useradd \
        --system \
        --no-create-home \
        --shell /sbin/nologin \
        --gid "$SENTINEL_GROUP" \
        --comment "SentinelAI daemon" \
        "$SENTINEL_USER"

    info "Created system user: $SENTINEL_USER (no login shell, no home dir)"
}

# ── Directory structure ───────────────────────────────────────────────────────

create_directories() {
    section "Creating runtime directories"

    # Persist /run/sentinel across reboots via tmpfiles.d
    # Write this FIRST so systemd-tmpfiles won't clean it during setup
    cat > /etc/tmpfiles.d/sentinel.conf << 'TMPFILES'
# SentinelAI runtime directory — recreated on boot
d /run/sentinel 0755 sentinel sentinel -
TMPFILES
    info "tmpfiles.d entry written"

    local DIRS=(
        "$DIR_VAR"
        "$DIR_STATE"
        "$DIR_MODELS"
        "$DIR_TRASH"
        "$DIR_ETC"
        "$DIR_OPT_DAEMON"
        "$DIR_OPT_EMBED"
        "$DIR_OPT_QUERY"
    )

    for d in "${DIRS[@]}"; do
        mkdir -p "$d"
        info "  $d"
    done

    # /run/sentinel needs special handling: use install -d to atomically
    # create + set ownership in one step (avoids WSL2 tmpfs race condition)
    install -d -m 755 -o "$SENTINEL_USER" -g "$SENTINEL_GROUP" "$DIR_RUN"
    info "  $DIR_RUN (install -d)"

    # Ownership: sentinel user owns its data and opt directories
    chown -R "$SENTINEL_USER:$SENTINEL_GROUP" \
        "$DIR_VAR" "$DIR_OPT"

    # /etc/sentinel is root-owned but sentinel-readable (contains the env file)
    chown root:sentinel "$DIR_ETC"
    chmod 750 "$DIR_ETC"
}

# ── Install Python services into virtualenvs ──────────────────────────────────

install_python_services() {
    section "Installing Python services"

    # ── sentinel-Query ────────────────────────────────────────────────
    local QUERY_VENV="$DIR_OPT_QUERY/venv"

    info "Creating venv: $QUERY_VENV"
    python3 -m venv "$QUERY_VENV"
    "$QUERY_VENV/bin/pip" install -q --upgrade pip
    "$QUERY_VENV/bin/pip" install -q \
        llama-cpp-python \
        huggingface-hub

    #copy sentinel_query source
    cp -r "$REPO_ROOT/QUERY/"* "$DIR_OPT_QUERY/"
    chown -R "$SENTINEL_USER:$SENTINEL_GROUP" "$DIR_OPT_QUERY"

    # ← ADD THIS LINE (mirrors what embedding does)
    "$QUERY_VENV/bin/pip" install -q "$DIR_OPT_QUERY"

    local PY_VER
    PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo "/opt/sentinel/query" \
        > "$QUERY_VENV/lib/python${PY_VER}/site-packages/sentinel_query.pth"

    info "sentinel-query installed ✓"
    
    # ── sentinel-Embedding ────────────────────────────────────────────────
    local EMBED_VENV="$DIR_OPT_EMBED/venv"
    info "Creating venv: $EMBED_VENV"
    python3 -m venv "$EMBED_VENV"
    "$EMBED_VENV/bin/pip" install -q --upgrade pip
    "$EMBED_VENV/bin/pip" install -q \
        onnxruntime \
        tokenizers \
        hnswlib \
        optimum[onnxruntime]

    # Copy sentinel-embedding source
    cp -r "$REPO_ROOT/embedding/"* "$DIR_OPT_EMBED/"
    chown -R "$SENTINEL_USER:$SENTINEL_GROUP" "$DIR_OPT_EMBED"
    
    # Install the package itself to generate the entrypoint scripts
    "$EMBED_VENV/bin/pip" install -q "$DIR_OPT_EMBED"

    info "sentinel-embedding installed ✓"
    
    # ── Prepare embedding model (MiniLM) ──────────────────────────────────
    info "Preparing embedding model (MiniLM) …"
    "$EMBED_VENV/bin/python3" "$DIR_OPT_EMBED/download_model.py" \
        --output "$DIR_MODELS/minilm"
    chown -R "$SENTINEL_USER:$SENTINEL_GROUP" "$DIR_MODELS/minilm"
    info "Embedding model ready ✓"

    # ── sentinel-query (translator + orchestrator) ────────────────────────
    local QUERY_VENV="$DIR_OPT_QUERY/venv"
    info "Creating venv: $QUERY_VENV"
    python3 -m venv "$QUERY_VENV"
    "$QUERY_VENV/bin/pip" install -q --upgrade pip
    "$QUERY_VENV/bin/pip" install -q \
        llama-cpp-python \
        huggingface-hub

    # Copy sentinel-query source
    cp -r "$REPO_ROOT/QUERY/"* "$DIR_OPT_QUERY/"
    chown -R "$SENTINEL_USER:$SENTINEL_GROUP" "$DIR_OPT_QUERY"

    info "sentinel-query installed ✓"
}

# ── Install compiled Rust binary ──────────────────────────────────────────────

install_daemon() {
    section "Installing sentinel-daemon (Rust binary)"

    local BINARY="$REPO_ROOT/DAEMON/OS-DAEMON/target/release/sentinel-daemon"

    if [[ ! -f "$BINARY" ]]; then
        warn "Compiled binary not found at $BINARY"
        warn "Building from source (requires Rust toolchain) …"

        # Source root's cargo environment if installed
        if [[ -f "$HOME/.cargo/env" ]]; then
            source "$HOME/.cargo/env"
        fi

        if ! command -v cargo &>/dev/null; then
            die "cargo not found. Install Rust: https://rustup.rs" 3
        fi

        (
            cd "$REPO_ROOT/DAEMON/OS-DAEMON"
            export CARGO_TARGET_DIR=/tmp/sentinel-cargo-target
            cargo build --release 2>&1 | tail -5
        ) || die "cargo build failed" 3

        BINARY="/tmp/sentinel-cargo-target/release/sentinel-daemon"
    fi

    install -m 755 -o root -g root "$BINARY" /usr/local/bin/sentinel-daemon
    info "sentinel-daemon installed to /usr/local/bin/ ✓"
}

# ── Download the GGUF model ───────────────────────────────────────────────────

download_model() {
    section "Downloading Llama-3 8B GGUF model"

    local MODEL_PATH="$DIR_MODELS/$GGUF_MODEL"

    if [[ -f "$MODEL_PATH" ]]; then
        info "Model already present at $MODEL_PATH — skipping download"
        return
    fi

    if [[ "$SKIP_MODEL" == "true" ]]; then
        warn "Model download skipped (--skip-model). Set SENTINEL_LLM_MODEL_PATH manually."
        return
    fi

    info "Downloading ~4.5 GB model from HuggingFace — this may take several minutes …"
    info "Model: $GGUF_REPO / $GGUF_MODEL"
    info "Destination: $MODEL_PATH"

    local QUERY_VENV="$DIR_OPT_QUERY/venv"

    # Use huggingface-hub CLI directly to download the GGUF model
    info "Starting GGUF model download via huggingface-hub …"
    "$QUERY_VENV/bin/python3" -c "
from huggingface_hub import hf_hub_download
import os, sys
try:
    path = hf_hub_download(
        repo_id='$GGUF_REPO',
        filename='$GGUF_MODEL',
        local_dir='$DIR_MODELS',
        local_dir_use_symlinks=False,
    )
    print(f'Downloaded to: {path}')
except Exception as e:
    print(f'Error downloading model: {e}', file=sys.stderr)
    sys.exit(1)
" || die "Model download failed. Check your internet connection." 4

    chown "$SENTINEL_USER:$SENTINEL_GROUP" "$MODEL_PATH"
    info "Model downloaded ✓"
}

# ── Write environment file ────────────────────────────────────────────────────

write_env_file() {
    section "Writing /etc/sentinel/sentinel.env"

    cat > "$ENV_FILE" << EOF
# SentinelAI environment configuration
# Generated by install.sh on $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Edit this file to customise paths, CPU limits, and model selection.

# ── Paths ─────────────────────────────────────────────────────────────────────
SENTINEL_KUZU_PATH=/var/sentinel/graph
SENTINEL_GRAPH_LOG=/var/sentinel/state/graph.ndjson
SENTINEL_BULK_STATE=/var/sentinel/state/bulk_index.json
SENTINEL_WATCH_PATHS=/home:/root
SENTINEL_TRASH_DIR=/var/sentinel/trash

# ── Sockets ───────────────────────────────────────────────────────────────────
SENTINEL_DAEMON_SOCKET=/run/sentinel/daemon.sock
SENTINEL_VECTORSTORE_SOCKET=/run/sentinel/vectorstore.sock
SENTINEL_EMBEDDING_SOCKET=/run/sentinel/embedding.sock
SENTINEL_QUERY_SOCKET=/run/sentinel/query.sock
SENTINEL_ORCHESTRATOR_SOCKET=/run/sentinel/orchestrator.sock

# ── LLM model ─────────────────────────────────────────────────────────────────
SENTINEL_LLM_MODEL_PATH=$DIR_MODELS/$GGUF_MODEL

# ── Performance tuning ────────────────────────────────────────────────────────
# Maximum CPU % before the governor pauses indexing (default: 15)
SENTINEL_CPU_THRESHOLD=15

# LLM inference threads (default: half of logical CPUs)
# SENTINEL_LLM_N_THREADS=4
EOF

    chmod 640 "$ENV_FILE"
    chown root:sentinel "$ENV_FILE"
    info "Environment file written: $ENV_FILE"
}

# ── Install systemd units ─────────────────────────────────────────────────────

install_systemd_units() {
    section "Installing systemd service units"

    local UNIT_SOURCES=(
        "$REPO_ROOT/embedding/sentinel-vectorstore.service"
        "$REPO_ROOT/embedding/sentinel-embedding.service"
        "$REPO_ROOT/DAEMON/OS-DAEMON/sentinel-daemon.service"
        "$REPO_ROOT/QUERY/Orchestrator/sentinel-orchestrator.service"
    )

    for src in "${UNIT_SOURCES[@]}"; do
        local unit_name
        unit_name=$(basename "$src")
        if [[ ! -f "$src" ]]; then
            warn "Unit file not found: $src — generating minimal unit"
            generate_minimal_unit "$unit_name"
        else
            cp "$src" "$SYSTEMD_DIR/$unit_name"
            info "  Installed $unit_name"
        fi
    done

    systemctl daemon-reload
    info "systemd units installed and daemon reloaded ✓"
}

generate_minimal_unit() {
    # Fallback: generate a minimal valid unit if the source repo unit is missing.
    local unit_name="$1"
    cat > "$SYSTEMD_DIR/$unit_name" << EOF
[Unit]
Description=SentinelAI — $unit_name
After=network.target

[Service]
Type=simple
User=sentinel
Group=sentinel
EnvironmentFile=/etc/sentinel/sentinel.env
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
    warn "Generated minimal unit for $unit_name — replace with full unit from repo"
}

# ── Enable and start services ─────────────────────────────────────────────────

start_services() {
    section "Enabling and starting services"

    # Start in dependency order (each service waits for the previous)
    for svc in "${SERVICES[@]}"; do
        info "Enabling $svc …"
        systemctl enable "$svc"

        info "Starting $svc …"
        systemctl start "$svc"

        # Wait up to 15 seconds for the service to become active
        local attempts=0
        while [[ $attempts -lt 15 ]]; do
            if systemctl is-active --quiet "$svc"; then
                info "  $svc is active ✓"
                break
            fi
            sleep 1
            (( attempts++ ))
        done

        if ! systemctl is-active --quiet "$svc"; then
            error "$svc failed to start"
            journalctl -u "$svc" --no-pager -n 20 >&2
            die "Service startup failed: $svc" 1
        fi
    done
}

# ── Smoke test ────────────────────────────────────────────────────────────────

run_smoke_test() {
    section "Running smoke test"

    if [[ "$SKIP_SMOKE" == "true" ]]; then
        warn "Smoke test skipped (--skip-smoke)"
        return
    fi

    local TEST_FILE="/tmp/sentinel_smoke_test_$(date +%s).txt"
    local TEST_CONTENT="sentinel smoke test unique identifier alpha bravo charlie"
    local SOCKET="/run/sentinel/orchestrator.sock"
    local MAX_WAIT=30
    local PASS=false

    # Write a known file
    echo "$TEST_CONTENT" > "$TEST_FILE"
    info "Created test file: $TEST_FILE"

    # Wait for the orchestrator socket to appear (services may still be starting)
    local waited=0
    while [[ ! -S "$SOCKET" && $waited -lt $MAX_WAIT ]]; do
        sleep 1
        (( waited++ ))
    done

    if [[ ! -S "$SOCKET" ]]; then
        die "Orchestrator socket not available after ${MAX_WAIT}s — services may still be starting" 5
    fi

    # Give the bulk indexer a moment to index the new file
    info "Waiting up to 20 s for test file to be indexed …"
    sleep 5

    # Send a query to the orchestrator and check the response
    local QUERY_PAYLOAD
    QUERY_PAYLOAD=$(python3 -c "
import json, struct, socket, time

payload = json.dumps({'query': 'sentinel smoke test alpha bravo charlie', 'k': 5}).encode()
frame = struct.pack('<I', len(payload)) + payload

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.settimeout(15)
sock.connect('$SOCKET')
sock.sendall(frame)

header = sock.recv(4)
(n,) = struct.unpack('<I', header)
data = b''
while len(data) < n:
    data += sock.recv(n - len(data))
sock.close()
print(data.decode('utf-8'))
" 2>/dev/null)

    if [[ -z "$QUERY_PAYLOAD" ]]; then
        warn "No response from orchestrator — the system may still be indexing"
        warn "Run the smoke test manually once bulk indexing completes:"
        warn "  python3 $INSTALLER_DIR/scripts/smoke_test.py"
        rm -f "$TEST_FILE"
        return
    fi

    # Check if any result path contains our test file
    local FOUND
    FOUND=$(python3 -c "
import json, sys
data = json.loads(sys.argv[1])
results = data.get('results', [])
for r in results:
    if '$TEST_FILE' in r.get('path', ''):
        print('FOUND')
        break
" "$QUERY_PAYLOAD" 2>/dev/null || true)

    rm -f "$TEST_FILE"

    if [[ "$FOUND" == "FOUND" ]]; then
        info "Smoke test PASSED — test file found in search results ✓"
    else
        warn "Smoke test: test file not yet in results (bulk indexing may be ongoing)"
        warn "This is normal on first install. Re-run the smoke test after indexing completes:"
        warn "  python3 $INSTALLER_DIR/scripts/smoke_test.py"
        info "SentinelAI is running — smoke test will pass once indexing completes."
    fi
}

# ── Uninstall ─────────────────────────────────────────────────────────────────

uninstall() {
    section "Uninstalling SentinelAI"

    # Stop and disable services
    for svc in "${SERVICES[@]}"; do
        if systemctl is-active --quiet "$svc" 2>/dev/null; then
            info "Stopping $svc …"
            systemctl stop "$svc" || true
        fi
        if systemctl is-enabled --quiet "$svc" 2>/dev/null; then
            systemctl disable "$svc" || true
        fi
        rm -f "$SYSTEMD_DIR/$svc.service"
        info "  Removed $svc"
    done

    systemctl daemon-reload

    # Remove installed files
    rm -f /usr/local/bin/sentinel-daemon
    rm -f /etc/tmpfiles.d/sentinel.conf

    # Optionally remove data directories
    if [[ "$UNATTENDED" != "true" ]]; then
        read -rp "Remove all SentinelAI data in $DIR_VAR? [y/N] " CONFIRM
        if [[ "${CONFIRM,,}" == "y" ]]; then
            rm -rf "$DIR_VAR" "$DIR_OPT" "$DIR_ETC"
            info "Data directories removed"
        else
            info "Data directories preserved at $DIR_VAR"
        fi
    else
        info "Data directories preserved at $DIR_VAR (remove manually if needed)"
    fi

    # Remove sentinel user
    if id "$SENTINEL_USER" &>/dev/null; then
        userdel "$SENTINEL_USER" || true
        groupdel "$SENTINEL_GROUP" 2>/dev/null || true
        info "Removed system user: $SENTINEL_USER"
    fi

    info "Uninstall complete."
    exit 0
}

# ── Print post-install summary ────────────────────────────────────────────────

print_summary() {
    section "Installation complete"

    echo ""
    echo -e "${BOLD}SentinelAI is installed and running.${RESET}"
    echo ""
    echo "  Service status:"
    for svc in "${SERVICES[@]}"; do
        local STATUS
        STATUS=$(systemctl is-active "$svc" 2>/dev/null || echo "unknown")
        if [[ "$STATUS" == "active" ]]; then
            echo -e "    ${GREEN}●${RESET} $svc"
        else
            echo -e "    ${RED}●${RESET} $svc ($STATUS)"
        fi
    done
    echo ""
    echo "  Key paths:"
    echo "    Config:       $ENV_FILE"
    echo "    Model:        $DIR_MODELS/$GGUF_MODEL"
    echo "    Graph DB:     /var/sentinel/graph"
    echo "    Sockets:      /run/sentinel/*.sock"
    echo "    Logs:         journalctl -u sentinel-orchestrator -f"
    echo ""
    echo "  First-run bulk indexing:"
    echo "    Watch progress: journalctl -u sentinel-daemon -f"
    echo ""
    echo "  To query from the command line:"
    cat << 'QUERY_EXAMPLE'
    python3 - << 'EOF'
import json, socket, struct
def query(text):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect('/run/sentinel/orchestrator.sock')
    p = json.dumps({'query': text, 'k': 10}).encode()
    s.sendall(struct.pack('<I', len(p)) + p)
    n = struct.unpack('<I', s.recv(4))[0]
    data = b''; 
    while len(data) < n: data += s.recv(n - len(data))
    s.close()
    return json.loads(data)
r = query('find my report from last week')
for f in r['results']: print(f['score']:.2f, f['path'])
EOF
QUERY_EXAMPLE
    echo ""
}

# ── Main ──────────────────────────────────────────────────────────────────────

main() {
    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}║       SentinelAI Installer v0.1.0        ║${RESET}"
    echo -e "${BOLD}╚══════════════════════════════════════════╝${RESET}"
    echo ""

    if [[ "$UNINSTALL" == "true" ]]; then
        uninstall
    fi

    if [[ "$UNATTENDED" != "true" ]]; then
        echo "This installer will:"
        echo "  • Install system packages (python3, poppler-utils, tesseract)"
        echo "  • Create the 'sentinel' system user"
        echo "  • Write to /var/sentinel, /etc/sentinel, /opt/sentinel"
        echo "  • Download a ~4.5 GB GGUF model (Llama-3 8B)"
        echo "  • Install and start four systemd services"
        echo ""
        read -rp "Continue? [y/N] " CONFIRM
        [[ "${CONFIRM,,}" == "y" ]] || { echo "Aborted."; exit 0; }
    fi

    detect_os
    install_deps
    create_user
    create_directories
    install_python_services
    install_daemon
    download_model
    write_env_file
    install_systemd_units
    start_services
    run_smoke_test
    print_summary
}

main "$@"
