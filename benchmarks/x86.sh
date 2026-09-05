#!/bin/sh
# Collect the full x86 measurement set for BENCHMARKS.md (spec T-008).
#
#   sh benchmarks/x86.sh                 # results in benchmarks/results/<arch>-<isa>-<virt>/
#   sh benchmarks/x86.sh -name zen4-box  # choose the directory name yourself
#   sh benchmarks/x86.sh -nopython       # skip the NumPy/PyTorch comparison
#
# The results are meant to be committed to a public repository, so no
# hostname, user name or IP address is recorded; the directory is named
# after the hardware class (e.g. x86_64-avx2-kvm) unless -name is given.
#
# Needs: Linux x86-64, git checkout of the repository, Go (any version with
# GOTOOLCHAIN=auto; go.mod pins the toolchain), python3 with venv for the
# comparison. On a machine without internet access put pre-downloaded
# wheels (numpy, torch and dependencies for linux x86-64 and the local
# Python version) into a directory and pass it as FIBERAI_WHEELS=<dir>.
set -eu

cd "$(dirname "$0")/.."
python=1
name=""
while [ $# -gt 0 ]; do
  case "$1" in
    -nopython) python=0 ;;
    -name) shift; name="$1" ;;
    *) echo "usage: $0 [-name label] [-nopython]" >&2; exit 2 ;;
  esac
  shift
done
if [ -z "$name" ]; then
  flags=$(grep -m1 flags /proc/cpuinfo)
  case " $flags " in
    *" avx512f "*) isa=avx512 ;;
    *" avx2 "*) isa=avx2 ;;
    *) isa=generic ;;
  esac
  virt=$(systemd-detect-virt 2>/dev/null || echo unknown)
  name="$(uname -m)-$isa-$virt"
fi
out="benchmarks/results/$name"
mkdir -p "$out"

log() { printf '%s\n' "$*" | tee -a "$out/run.log"; }
: > "$out/run.log"
log "fiber/ai x86 measurement ($name), $(date -u +%Y-%m-%dT%H:%MZ)"

# --- machine ---------------------------------------------------------------
{
  echo "== CPU =="
  grep -m1 "model name" /proc/cpuinfo | cut -d: -f2
  echo "cores: $(nproc)"
  lscpu | grep -E "^(Socket|NUMA node)\(s\)|Thread\(s\) per core|Model name"
  echo "virt: $(systemd-detect-virt 2>/dev/null || echo unknown)"
  echo "== Flags =="
  f=$(grep -m1 flags /proc/cpuinfo)
  for x in avx2 fma avx512f avx512bw avx512vl avx512dq avx512_vnni avx512_bf16 amx_tile amx_bf16 amx_int8 avx_vnni; do
    case " $f " in *" $x "*) echo "  $x: yes";; *) echo "  $x: NO";; esac
  done
  echo "== Caches =="
  lscpu | grep -E "L1d|L2|L3"
  echo "== Frequency =="
  lscpu | grep -E "MHz" || true
  grep -m1 "cpu MHz" /proc/cpuinfo || true
  echo "== Kernel =="
  uname -srm   # kernel release and arch; -n (host name) is deliberately not used
  echo "== Toolchain =="
  go version
  python3 --version 2>&1 || true
} > "$out/machine.txt" 2>&1
log "machine info -> $out/machine.txt"
grep -E "avx2|avx512f" "$out/machine.txt" | tee -a "$out/run.log"

# --- correctness -----------------------------------------------------------
log "go vet"
go vet ./... > "$out/vet.txt" 2>&1 && log "  ok" || log "  FAILED (see vet.txt)"
log "go test (detected backend)"
go test -count=1 ./... > "$out/test.txt" 2>&1 && log "  ok" || log "  FAILED (see test.txt)"
go test -count=1 ./internal/kernel -run TestInitSelectedImplementation -v 2>&1 | grep -E "active kernel|warnings" | tee -a "$out/run.log" >> "$out/test.txt"
for k in avx2 generic; do
  log "go test FIBERAI_KERNEL=$k"
  FIBERAI_KERNEL=$k go test -count=1 ./internal/... ./tensor/ > "$out/test-$k.txt" 2>&1 && log "  ok" || log "  FAILED (see test-$k.txt)"
done
if grep -q "avx512f: yes" "$out/machine.txt"; then
  log "go test FIBERAI_KERNEL=avx512"
  FIBERAI_KERNEL=avx512 go test -count=1 ./internal/... ./tensor/ > "$out/test-avx512.txt" 2>&1 && log "  ok" || log "  FAILED (see test-avx512.txt)"
fi

# --- throughput ------------------------------------------------------------
log "go bench (detected backend)"
go run ./cmd/bench > "$out/go.md" 2>&1
log "go bench FIBERAI_KERNEL=avx2 -quick"
FIBERAI_KERNEL=avx2 go run ./cmd/bench -quick > "$out/go-avx2.md" 2>&1
log "go bench FIBERAI_KERNEL=generic -quick"
FIBERAI_KERNEL=generic go run ./cmd/bench -quick > "$out/go-generic.md" 2>&1
log "go bench, 1 thread (GEMM efficiency)"
go test ./internal/blas -run x -bench 'Gemm$' -benchtime=1s > "$out/gemm-bench.txt" 2>&1 || true

# --- python comparison -----------------------------------------------------
if [ $python -eq 1 ]; then
  venv="benchmarks/python/.venv"
  if [ ! -x "$venv/bin/python" ]; then
    log "creating $venv and installing numpy + torch (CPU)"
    if ! python3 -m venv "$venv" 2> "$out/venv.err"; then
      log "  python3 -m venv failed (Debian/Ubuntu: sudo apt install python3-venv); rerun, or use -nopython"
      log "  details in $out/venv.err"
      exit 1
    fi
    if [ -n "${FIBERAI_WHEELS:-}" ]; then
      log "  offline install from $FIBERAI_WHEELS"
      if ! "$venv/bin/pip" install -q --no-index --find-links "$FIBERAI_WHEELS" numpy torch > "$out/pip.log" 2>&1; then
        log "  offline pip install failed; see $out/pip.log"
        exit 1
      fi
    else
      "$venv/bin/pip" install -q --upgrade pip > "$out/pip.log" 2>&1 || true
      if ! "$venv/bin/pip" install -q numpy torch --index-url https://download.pytorch.org/whl/cpu >> "$out/pip.log" 2>&1 \
         && ! "$venv/bin/pip" install -q numpy torch >> "$out/pip.log" 2>&1; then
        log "  pip install numpy torch failed; see $out/pip.log (no internet? set FIBERAI_WHEELS=<dir of wheels>) — rerun, or use -nopython"
        exit 1
      fi
    fi
  fi
  if ! "$venv/bin/python" -c "import numpy, torch" 2> "$out/pip.log"; then
    log "  numpy/torch not importable from $venv; see $out/pip.log"
    exit 1
  fi
  {
    echo "== numpy build =="
    "$venv/bin/python" -c "import numpy; numpy.show_config()" 2>&1 | grep -iE "name|blas|lapack|openblas|mkl|version" | head -20
    echo "== torch build =="
    "$venv/bin/python" -c "import torch; print(torch.__version__); print(torch.__config__.parallel_info())" 2>&1
  } > "$out/python-build.txt" 2>&1
  log "python bench"
  "$venv/bin/python" benchmarks/python/bench.py > "$out/python.md" 2>&1 || log "  python bench FAILED (see python.md)"
fi

log "done: $(ls "$out" | tr '\n' ' ')"
