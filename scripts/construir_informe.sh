#!/usr/bin/env bash
# Construye docs/avance2/informe.pdf desde informe.md (requiere pandoc y wkhtmltopdf).
set -euo pipefail
export LANG=C.UTF-8 LC_ALL=C.UTF-8
cd "$(dirname "$0")/../docs/avance2"
pandoc informe.md -o informe.pdf --from markdown+pipe_tables --standalone --css estilo.css \
  --pdf-engine=wkhtmltopdf --pdf-engine-opt=--enable-local-file-access \
  --pdf-engine-opt=--margin-top --pdf-engine-opt=18mm --pdf-engine-opt=--margin-bottom --pdf-engine-opt=18mm \
  --pdf-engine-opt=--margin-left --pdf-engine-opt=18mm --pdf-engine-opt=--margin-right --pdf-engine-opt=18mm
echo "OK: docs/avance2/informe.pdf"
