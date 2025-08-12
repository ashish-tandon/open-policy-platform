#!/usr/bin/env python3
import sys, os, yaml, glob

ROOT=os.path.dirname(os.path.dirname(__file__))
INV=os.path.join(ROOT, 'docs', 'reference', 'services.inventory.yaml')

required_files_http = ['Dockerfile', 'README.md']
required_files_any = ['Dockerfile', 'README.md']

def exists(path):
    return os.path.exists(path)

def load_inventory():
    with open(INV, 'r') as f:
        return yaml.safe_load(f)

def check_service(svc):
    path = os.path.join(ROOT, svc['path'])
    issues=[]
    # files
    files = required_files_http if svc['kind'].startswith('http') else required_files_any
    for rf in files:
        if not exists(os.path.join(path, rf)):
            issues.append(f"missing file: {rf}")
    # contracts (heuristics)
    contracts = set(svc.get('contracts', []))
    if 'openapi' in contracts:
        # expect dist/openapi.(json|yaml) or script reference
        has_openapi = bool(glob.glob(os.path.join(path, 'dist', 'openapi.*'))) or exists(os.path.join(ROOT, 'scripts', 'export-openapi.sh'))
        if not has_openapi:
            issues.append('missing openapi artifact or export capability')
    # health/ready/metrics cannot be validated without running; mark informational
    return issues

def main():
    inv = load_inventory()
    all_issues=[]
    for svc in inv['services']:
        issues=check_service(svc)
        if issues:
            all_issues.append({'service': svc['name'], 'path': svc['path'], 'issues': issues})
    if all_issues:
        print('Service validation report:')
        for e in all_issues:
            print(f"- {e['service']} ({e['path']}):")
            for i in e['issues']:
                print(f"  * {i}")
        # report-only mode
        sys.exit(0)
    else:
        print('All services satisfy baseline file/contract checks (report-only).')

if __name__ == '__main__':
    main()