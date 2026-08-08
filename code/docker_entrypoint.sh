#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${VASCFUSION_ROOT:-/opt/vascfusion}"

usage() {
    cat <<'EOF'
VascFusion container commands:

  verify [--require-cuda] [--require-weights] [--data-root /data]
      Validate the runtime, optional CUDA access, dataset layout, and checkpoint
      SHA256 digests. Add --require-initialization when training checkpoints
      should also be mandatory.

  inference --data-root /data --work-root /output/work \
            --output-zip /output/vascfusion.zip [other run_inference.sh options]
      Run all three GAVE2 tasks and package the results.

  shell
      Open an interactive shell for inspection.

The complete checkpoint directory must be mounted at /opt/vascfusion/weights.
Challenge data and an output directory should be mounted separately.
EOF
}

command_name="${1:-help}"
case "${command_name}" in
    help|-h|--help)
        usage
        ;;
    verify)
        shift
        exec python "${APP_ROOT}/code/verify_runtime.py" "$@"
        ;;
    inference)
        shift
        exec bash "${APP_ROOT}/code/run_inference.sh" "$@" --python python
        ;;
    shell)
        shift
        exec /bin/bash "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
