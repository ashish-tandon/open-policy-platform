#!/usr/bin/env python3
import yaml, os, concurrent.futures, subprocess, sys

ROOT=os.path.dirname(os.path.dirname(__file__))
INV=os.path.join(ROOT, 'docs', 'reference', 'services.inventory.yaml')


def load_inventory():
    with open(INV, 'r') as f:
        return yaml.safe_load(f)

def run_service_task(svc):
    path=os.path.join(ROOT, svc['path'])
    # Placeholder: validate presence
    ok=os.path.exists(path)
    return svc['name'], ok

def main():
    inv=load_inventory()
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs=[ex.submit(run_service_task, s) for s in inv['services']]
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())
    for name, ok in results:
        print(f"{name}: {'OK' if ok else 'MISSING'}")
    sys.exit(0)

if __name__=='__main__':
    main()