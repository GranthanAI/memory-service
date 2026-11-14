r"""Generate the billing gRPC stubs from proto/billing/billing.proto into app/clients/billing_grpc/.

Run from the repo root (Windows, Git Bash or PowerShell):
    .venv\Scripts\python scripts\gen_billing_proto.py        # or: make proto
Uses the grpcio-tools already in the environment. The generated *_grpc.py imports its sibling with an absolute
name; this script rewrites that to a package-relative import so the stubs work inside app.clients.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "app" / "clients" / "billing_grpc"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "__init__.py").touch()
    cmd = [sys.executable, "-m", "grpc_tools.protoc", f"-I{ROOT / 'proto' / 'billing'}",
           f"--python_out={OUT}", f"--grpc_python_out={OUT}", str(ROOT / "proto" / "billing" / "billing.proto")]
    subprocess.run(cmd, check=True)
    grpc_file = OUT / "billing_pb2_grpc.py"
    text = grpc_file.read_text(encoding="utf-8")
    text = re.sub(r"^import billing_pb2 as billing__pb2$", "from . import billing_pb2 as billing__pb2", text, flags=re.M)
    grpc_file.write_text(text, encoding="utf-8")
    print("generated", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
