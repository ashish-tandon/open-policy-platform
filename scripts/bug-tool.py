#!/usr/bin/env python3
import sys, yaml, uuid

BUG_FILE='scripts/bug-register.yaml'

def load():
    try:
        with open(BUG_FILE, 'r') as f:
            return yaml.safe_load(f) or {'bugs': []}
    except FileNotFoundError:
        return {'bugs': []}

def save(db):
    with open(BUG_FILE, 'w') as f:
        yaml.safe_dump(db, f, sort_keys=False)

def list_bugs(db):
    for b in db['bugs']:
        print(f"{b['id']}: [{b['status']}] {b['service']} - {b['title']} ({b.get('priority','')})")

def add_bug(db, service, title, priority='medium'):
    bid = f"BUG-{uuid.uuid4().hex[:6].upper()}"
    db['bugs'].append({'id': bid, 'service': service, 'title': title, 'status': 'open', 'priority': priority})
    save(db)
    print(bid)

def close_bug(db, bug_id):
    for b in db['bugs']:
        if b['id'] == bug_id:
            b['status'] = 'closed'
            save(db)
            print('closed')
            return
    print('not found', file=sys.stderr)
    sys.exit(1)

if __name__ == '__main__':
    db = load()
    if len(sys.argv) < 2:
        print('usage: bug-tool.py list | add <service> <title> [priority] | close <bug_id>')
        sys.exit(2)
    cmd = sys.argv[1]
    if cmd == 'list':
        list_bugs(db)
    elif cmd == 'add':
        add_bug(db, sys.argv[2], ' '.join(sys.argv[3:-1]) if len(sys.argv)>4 else sys.argv[3], sys.argv[-1] if len(sys.argv)>4 else 'medium')
    elif cmd == 'close':
        close_bug(db, sys.argv[2])
    else:
        print('unknown command', file=sys.stderr); sys.exit(2)