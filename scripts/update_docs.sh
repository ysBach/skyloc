#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/update_docs.sh [options]

Build Sphinx docs for skyloc and optionally commit/push docs source updates.

Options:
  --allow-dirty           Build docs even when the git working tree is dirty.
  --include-autosummary   Stage docs/_autosummary/*.rst (generated stubs).
  --stage                 Stage docs-source changes after build.
  --commit                Commit staged docs-source changes.
  --push                  Push current branch after commit (implies --commit).
  -m, --message <msg>     Commit message (default: "docs: update documentation").
  -h, --help              Show this help.

Notes:
  - HTML output under docs/_build/ is ignored and is never committed.
  - By default this script stages only docs source files, not notebooks.

# 1) Build docs only
scripts/update_docs.sh

# 2) Build + stage docs source changes
scripts/update_docs.sh --stage

# 3) Build + commit [+ push]
scripts/update_docs.sh --commit -m "docs: update docs"

USAGE
}

ALLOW_DIRTY=0
INCLUDE_AUTOSUMMARY=0
DO_COMMIT=0
DO_PUSH=0
DO_STAGE=0
COMMIT_MSG="docs: update documentation"

while (($#)); do
  case "$1" in
    --allow-dirty)
      ALLOW_DIRTY=1
      shift
      ;;
    --include-autosummary)
      INCLUDE_AUTOSUMMARY=1
      shift
      ;;
    --stage)
      DO_STAGE=1
      shift
      ;;
    --commit)
      DO_STAGE=1
      DO_COMMIT=1
      shift
      ;;
    --push)
      DO_STAGE=1
      DO_PUSH=1
      DO_COMMIT=1
      shift
      ;;
    -m|--message)
      if (($# < 2)); then
        echo "error: missing value for $1" >&2
        exit 2
      fi
      COMMIT_MSG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "error: 'uv' not found in PATH" >&2
  exit 1
fi

if [[ ! -d docs ]]; then
  echo "error: docs directory not found" >&2
  exit 1
fi

if ((ALLOW_DIRTY == 0)); then
  if ! git diff --quiet || ! git diff --cached --quiet; then
    cat >&2 <<'MSG'
error: working tree is not clean.
Commit/stash your changes first, or rerun with --allow-dirty.
MSG
    exit 3
  fi
fi

echo "[1/3] Building docs (Sphinx)..."
# Keep uv cache in-repo for portability (useful in sandboxed environments).
export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
mkdir -p "${UV_CACHE_DIR}"

uv run \
  --with sphinx \
  --with sphinx-rtd-theme \
  --with sphinx-copybutton \
  --with numpydoc \
  make -C docs html

if ((DO_STAGE == 1)); then
  echo "[2/3] Staging docs source changes..."
  find docs -maxdepth 1 -type f \( \
    -name "*.rst" -o \
    -name "conf.py" -o \
    -name "Makefile" -o \
    -name "make.bat" -o \
    -name "README.md" \
  \) | sort | while IFS= read -r doc_file; do
    [ -n "$doc_file" ] && git add -- "$doc_file"
  done

  if ((INCLUDE_AUTOSUMMARY == 1)) && [[ -d docs/_autosummary ]]; then
    git add -- docs/_autosummary
  fi
else
  echo "[2/3] Skipping git staging (use --stage, --commit, or --push)."
fi

if ((DO_COMMIT == 1)); then
  if git diff --cached --quiet; then
    echo "No staged docs source changes to commit."
  else
    echo "[3/3] Committing docs changes..."
    git commit -m "$COMMIT_MSG"
  fi
fi

if ((DO_PUSH == 1)); then
  branch="$(git rev-parse --abbrev-ref HEAD)"
  echo "Pushing to origin/${branch}..."
  git push -u origin "$branch"
fi

echo "Done."
