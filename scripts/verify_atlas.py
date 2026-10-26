#!/usr/bin/env python3
"""Verifica que los IDs de MITRE ATLAS usados por el proyecto existan en la versión vigente.

Uso:
    python scripts/verify_atlas.py --fetch              # descarga la versión vigente
    python scripts/verify_atlas.py ATLAS-2026.09.yaml   # usa un archivo local

Nota: dist/v6/ATLAS-latest.yaml del repositorio oficial no es el catálogo, solo contiene
el nombre del archivo de la versión vigente (por ejemplo ATLAS-2026.09.yaml).

Sale con código 1 si algún ID falta o si su nombre cambió respecto a EXPECTED_*.
"""
import argparse
import sys
import urllib.request

import yaml

BASE = "https://raw.githubusercontent.com/mitre-atlas/atlas-data/main/dist/v6/"

EXPECTED_TECHNIQUES = {
    "AML.T0051": "LLM Prompt Injection",
    "AML.T0051.000": "Direct",
    "AML.T0051.001": "Indirect",
    "AML.T0051.002": "Triggered",
    "AML.T0054": "LLM Jailbreak",
    "AML.T0056": "Extract LLM System Prompt",
    "AML.T0057": "LLM Data Leakage",
    "AML.T0068": "LLM Prompt Obfuscation",
    "AML.T0069.002": "System Prompt",
    "AML.T0053": "AI Agent Tool Invocation",
    "AML.T0086": "Exfiltration via AI Agent Tool Invocation",
    "AML.T0101": "Data Destruction via AI Agent Tool Invocation",
    "AML.T0110": "AI Agent Tool Poisoning",
    "AML.T0010.005": "AI Agent Tool",
    "AML.T0070": "RAG Poisoning",
    "AML.T0071": "False RAG Entry Injection",
    "AML.T0077": "LLM Response Rendering",
    "AML.T0129": "Triggers in Multimodal Inputs",
}

EXPECTED_MITIGATIONS = {
    "AML.M0008": "Validate AI Model",
    "AML.M0020": "Generative AI Guardrails",
    "AML.M0021": "Generative AI Guidelines",
    "AML.M0022": "Generative AI Model Alignment",
    "AML.M0024": "AI Telemetry Logging",
    "AML.M0026": "Privileged AI Agent Permissions Configuration",
    "AML.M0027": "Single-User AI Agent Permissions Configuration",
    "AML.M0028": "AI Agent Tools Permissions Configuration",
    "AML.M0029": "Human In-the-Loop for AI Agent Actions",
    "AML.M0030": "Restrict AI Agent Tool Invocation on Untrusted Data",
    "AML.M0032": "Segmentation of AI Agent Components",
    "AML.M0033": "Input and Output Validation for AI Agent Components",
    "AML.M0035": "AI Red Team",
}


def fetch_yaml() -> str:
    pointer = urllib.request.urlopen(BASE + "ATLAS-latest.yaml", timeout=60).read().decode().strip()
    return urllib.request.urlopen(BASE + pointer, timeout=120).read().decode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="archivo ATLAS-YYYY.MM.yaml local")
    ap.add_argument("--fetch", action="store_true", help="descargar la versión vigente")
    args = ap.parse_args()
    if not args.path and not args.fetch:
        ap.error("indica un archivo o usa --fetch")

    text = fetch_yaml() if args.fetch else open(args.path, encoding="utf-8").read()
    data = yaml.safe_load(text)
    techniques, mitigations = data["techniques"], data["mitigations"]
    tactics, relationships = data["tactics"], data["relationships"]

    # índices inversos: técnica -> mitigaciones y técnica -> tácticas
    mitigated_by, tactics_of = {}, {}
    for source, rels in relationships.items():
        for kind, items in rels.items():
            for rel in items:
                if kind == "mitigates":
                    mitigated_by.setdefault(rel["target"], set()).add(source)
                elif kind == "achieves" and source.startswith("AML.T"):
                    tactics_of.setdefault(source, set()).add(rel["target"])

    col = data["collection"]
    print(f"ATLAS {col['version']} (modificado {col['modified-date']}), formato {data['format-version']}")
    errors = 0

    print("\nTécnicas")
    for tid, expected in EXPECTED_TECHNIQUES.items():
        tech = techniques.get(tid)
        if tech is None:
            print(f"  FALTA   {tid} (esperado: {expected})")
            errors += 1
            continue
        flag = "OK     " if tech["name"] == expected else "RENOMBRADO"
        errors += flag != "OK     "
        names = ", ".join(tactics[t]["name"] for t in sorted(tactics_of.get(tid, [])) if t in tactics)
        mits = ", ".join(sorted(mitigated_by.get(tid, []))) or "sin mitigaciones vinculadas"
        print(f"  {flag} {tid}: {tech['name']} | {tech.get('maturity', '?')} | {names} | {mits}")

    print("\nMitigaciones")
    for mid, expected in EXPECTED_MITIGATIONS.items():
        mit = mitigations.get(mid)
        if mit is None:
            print(f"  FALTA   {mid} (esperado: {expected})")
            errors += 1
            continue
        flag = "OK     " if mit["name"] == expected else "RENOMBRADO"
        errors += flag != "OK     "
        print(f"  {flag} {mid}: {mit['name']}")

    print(f"\n{'Todo coincide.' if not errors else f'{errors} discrepancia(s): revisa el mapeo.'}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
