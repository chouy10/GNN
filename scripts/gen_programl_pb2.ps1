$ErrorActionPreference = "Stop"

$ROOT = (Resolve-Path ".").Path
$REPO = (Resolve-Path ".\external\ProGraML").Path
$OUT  = (Resolve-Path ".\src\programl_pb2").Path

Write-Host "ROOT = $ROOT"
Write-Host "REPO = $REPO"
Write-Host "OUT  = $OUT"

# IMPORTANT:
# ProGraML protos import with paths like "programl/proto/..." and
# "programl/third_party/tensorflow/features.proto"
# so -I must be the ProGraML repo root.
python -m grpc_tools.protoc `
  -I "$REPO" `
  --python_out="$OUT" `
  "$REPO\programl\proto\program_graph.proto" `
  "$REPO\programl\proto\util.proto"

Write-Host "`nDone. Generated files:"
Get-ChildItem "$OUT" -Filter "*_pb2.py" | Select-Object Name
