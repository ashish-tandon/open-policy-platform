#!/usr/bin/env python3
import os, sys
ROOT=os.path.dirname(os.path.dirname(__file__))

def scaffold(path):
    # dry-run print only
    print(f"Would scaffold: {path} (README.md, Dockerfile, src/, tests/)")

def main():
    if len(sys.argv)<2:
        print('usage: agent_service.py <service-path>')
        sys.exit(1)
    path=os.path.join(ROOT, sys.argv[1])
    if not os.path.exists(path):
        scaffold(path)
    else:
        print(f"Exists: {path}")

if __name__=='__main__':
    main()