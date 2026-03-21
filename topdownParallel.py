from mpi4py import MPI
import argparse
import random
from collections import defaultdict
from time import time

TAG = 1201

OP_INIT_LEAF = "init_leaf"
OP_INIT_INTERNAL_ABOVE = "init_internal_above"
OP_SET_ROOT = "set_root"

OP_INS = "ins"
OP_INS_REPLY = "ins_reply"
OP_FIND = "find"
OP_FIND_REPLY = "find_reply"
OP_SPLIT_NOTIFY = "split_notify"

OP_STOP = "stop"

LOG_MODE = "splits"


def log(rank, msg, level="all"):
    if LOG_MODE == "none":
        return
    if level == "all" and LOG_MODE != "all":
        return
    print(f"[Rank {rank}] {msg}", flush=True)


def lower_bound(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


class Node:
    def __init__(self, is_leaf, keys=None, values=None, children=None):
        self.is_leaf = is_leaf
        self.keys = [] if keys is None else keys
        self.values = values
        self.children = children
        self.left = None
        self.right = None
        self.high_key = None  # None => open-ended rightmost sibling


class LevelStore:
    def __init__(self, B):
        self.B = B
        self.max_degree = 2 ** B
        self.max_leaf = self.max_degree - 1
        self.nodes = {}
        self.next_id = 1

    def new_leaf(self):
        nid = self.next_id
        self.next_id += 1
        self.nodes[nid] = Node(True, [], [], None)
        return nid

    def new_internal(self):
        nid = self.next_id
        self.next_id += 1
        self.nodes[nid] = Node(False, [], None, [])
        return nid

    def init_internal_above(self, child_id):
        nid = self.new_internal()
        n = self.nodes[nid]
        n.children = [child_id]
        n.keys = []
        n.high_key = None
        return nid

    def update_leaf_high_key(self, nid):
        n = self.nodes[nid]
        if n.right is None and n.keys:
            n.high_key = None
        elif n.keys:
            n.high_key = n.keys[-1]
        else:
            n.high_key = None

    def update_internal_high_key(self, nid):
        n = self.nodes[nid]
        if n.right is None:
            n.high_key = None
        elif n.keys:
            n.high_key = n.keys[-1]
        else:
            n.high_key = None

    def child_for_key(self, nid, k):
        n = self.nodes[nid]
        i = lower_bound(n.keys, k)
        if i >= len(n.children):
            i = len(n.children) - 1
        return n.children[i]

    def chase_right(self, nid, k):
        cur = nid
        while True:
            n = self.nodes[cur]
            if n.high_key is not None and k > n.high_key and n.right is not None:
                cur = n.right
                continue
            return cur

    def leaf_contains(self, nid, k):
        n = self.nodes[nid]
        i = lower_bound(n.keys, k)
        return i < len(n.keys) and n.keys[i] == k

    def insert_in_leaf(self, nid, k, v):
        n = self.nodes[nid]
        i = lower_bound(n.keys, k)
        if i < len(n.keys) and n.keys[i] == k:
            n.values[i] = v
        else:
            n.keys.insert(i, k)
            n.values.insert(i, v)
        self.update_leaf_high_key(nid)

    def split_leaf(self, nid):
        n = self.nodes[nid]
        mid = len(n.keys) // 2
        rid = self.new_leaf()
        r = self.nodes[rid]

        r.keys = n.keys[mid:]
        r.values = n.values[mid:]
        n.keys = n.keys[:mid]
        n.values = n.values[:mid]

        r.left = nid
        r.right = n.right
        if n.right is not None and n.right in self.nodes:
            self.nodes[n.right].left = rid
        n.right = rid

        sep = r.keys[0]
        n.high_key = sep
        self.update_leaf_high_key(rid)
        return sep, rid

    def split_internal(self, nid):
        n = self.nodes[nid]
        m = len(n.children)
        mid_child = m // 2
        sep_index = mid_child - 1
        sep = n.keys[sep_index]

        rid = self.new_internal()
        r = self.nodes[rid]
        r.children = n.children[mid_child:]
        r.keys = n.keys[mid_child:]
        r.right = n.right
        if n.right is not None and n.right in self.nodes:
            self.nodes[n.right].left = rid
        n.right = rid
        r.left = nid

        n.children = n.children[:mid_child]
        n.keys = n.keys[:sep_index]

        n.high_key = sep
        self.update_internal_high_key(rid)
        return sep, rid

    def is_full_for_topdown(self, nid):
        n = self.nodes[nid]
        if n.is_leaf:
            return len(n.keys) >= self.max_leaf
        return len(n.children) >= self.max_degree



def init_fixed_height_tree(comm, size):
    comm.send({"op": OP_INIT_LEAF}, dest=1, tag=TAG)
    root_id = comm.recv(source=1, tag=TAG)["root_id"]
    for r in range(2, size):
        comm.send({"op": OP_INIT_INTERNAL_ABOVE, "child_id": root_id}, dest=r, tag=TAG)
        root_id = comm.recv(source=r, tag=TAG)["root_id"]
    root_rank = size - 1
    for r in range(1, size):
        comm.send({"op": OP_SET_ROOT, "root_rank": root_rank, "root_id": root_id}, dest=r, tag=TAG)
    return root_rank, root_id



def apply_split_install(store, nid, rep, orphans, rank):
    n = store.nodes[nid]
    left_id = rep["left_child_id"]
    right_id = rep["right_node"]
    sep = rep["split_key"]

    if right_id is None:
        return True
    if left_id not in n.children:
        orphans[left_id].append(rep)
        return False

    pos = n.children.index(left_id)
    if pos + 1 < len(n.children) and n.children[pos + 1] == right_id:
        return True

    n.children.insert(pos + 1, right_id)
    n.keys.insert(pos, sep)
    store.update_internal_high_key(nid)
    log(rank, f"Installed child split at node={nid}, sep={sep}, op_id={rep['op_id']}", "splits")
    return True



def flush_orphans(store, nid, orphans, rank):
    changed = True
    while changed:
        changed = False
        n = store.nodes[nid]
        for left_id in list(orphans.keys()):
            if left_id in n.children:
                payloads = orphans.pop(left_id)
                for rep in payloads:
                    apply_split_install(store, nid, rep, orphans, rank)
                    changed = True



def processor(comm, rank, B):
    store = LevelStore(B)
    root_rank = 1
    root_id = None

    pending_ins = {}
    pending_find = {}
    orphan_splits = defaultdict(list)

    while True:
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=TAG)
        op = msg["op"]

        if op == OP_STOP:
            return

        if op == OP_SET_ROOT:
            root_rank = msg["root_rank"]
            root_id = msg["root_id"]
            continue

        if op == OP_INIT_LEAF:
            rid = store.new_leaf()
            comm.send({"root_id": rid}, dest=0, tag=TAG)
            continue

        if op == OP_INIT_INTERNAL_ABOVE:
            rid = store.init_internal_above(msg["child_id"])
            comm.send({"root_id": rid}, dest=0, tag=TAG)
            continue

        if op == OP_SPLIT_NOTIFY:
            nid = msg["node_id"]
            if nid in store.nodes:
                apply_split_install(store, nid, msg, orphan_splits, rank)
                flush_orphans(store, nid, orphan_splits, rank)
            continue

        if op == OP_INS:
            op_id = msg["op_id"]
            k = msg["key"]
            v = msg["value"]
            nid = msg["node_id"]
            reply_to = msg["reply_to"]

            if nid not in store.nodes:
                comm.send({"op": OP_INS_REPLY, "op_id": op_id, "ok": False}, dest=reply_to, tag=TAG)
                continue

            nid = store.chase_right(nid, k)
            n = store.nodes[nid]

            if store.is_full_for_topdown(nid):
                if rank == root_rank:
                    raise RuntimeError("Top fixed root overflowed. Increase MPI ranks or B.")
                if n.is_leaf:
                    sep, rid = store.split_leaf(nid)
                    log(rank, f"LEAF SPLIT at node={nid} -> right={rid}, split_key={sep}, op_id={op_id}", "splits")
                else:
                    sep, rid = store.split_internal(nid)
                    log(rank, f"INTERNAL SPLIT at node={nid} -> right={rid}, split_key={sep}, op_id={op_id}", "splits")

                comm.send(
                    {
                        "op": OP_SPLIT_NOTIFY,
                        "node_id": msg["parent_node_id"],
                        "left_child_id": nid,
                        "right_node": rid,
                        "split_key": sep,
                        "op_id": op_id,
                    },
                    dest=rank + 1,
                    tag=TAG,
                )

                nid = store.chase_right(nid, k)
                n = store.nodes[nid]

            if n.is_leaf:
                store.insert_in_leaf(nid, k, v)
                comm.send({"op": OP_INS_REPLY, "op_id": op_id, "ok": True}, dest=reply_to, tag=TAG)
                continue

            child = store.child_for_key(nid, k)
            pending_ins[op_id] = {"reply_to": reply_to}
            comm.send(
                {
                    "op": OP_INS,
                    "op_id": op_id,
                    "key": k,
                    "value": v,
                    "node_id": child,
                    "parent_node_id": child if rank - 1 == 1 else child,
                    "reply_to": rank,
                },
                dest=rank - 1,
                tag=TAG,
            )
            continue

        if op == OP_INS_REPLY:
            op_id = msg["op_id"]
            ctx = pending_ins.pop(op_id)
            comm.send(msg, dest=ctx["reply_to"], tag=TAG)
            continue

        if op == OP_FIND:
            op_id = msg["op_id"]
            k = msg["key"]
            nid = msg["node_id"]
            reply_to = msg["reply_to"]

            if nid not in store.nodes:
                comm.send({"op": OP_FIND_REPLY, "op_id": op_id, "found": False}, dest=reply_to, tag=TAG)
                continue

            nid = store.chase_right(nid, k)
            n = store.nodes[nid]
            if n.is_leaf:
                found = store.leaf_contains(nid, k)
                comm.send({"op": OP_FIND_REPLY, "op_id": op_id, "found": found}, dest=reply_to, tag=TAG)
                continue

            child = store.child_for_key(nid, k)
            pending_find[op_id] = {"reply_to": reply_to}
            comm.send(
                {"op": OP_FIND, "op_id": op_id, "key": k, "node_id": child, "reply_to": rank},
                dest=rank - 1,
                tag=TAG,
            )
            continue

        if op == OP_FIND_REPLY:
            op_id = msg["op_id"]
            ctx = pending_find.pop(op_id)
            comm.send(msg, dest=ctx["reply_to"], tag=TAG)
            continue



def run_insert_phase(comm, root_rank, root_id, keys, window):
    next_op_id = 1
    next_idx = 0
    outstanding = 0
    inserted = []

    while next_idx < len(keys) or outstanding > 0:
        while next_idx < len(keys) and outstanding < window:
            k = keys[next_idx]
            comm.send(
                {
                    "op": OP_INS,
                    "op_id": next_op_id,
                    "key": k,
                    "value": f"v{k}",
                    "node_id": root_id,
                    "parent_node_id": -1,
                    "reply_to": 0,
                },
                dest=root_rank,
                tag=TAG,
            )
            inserted.append(k)
            next_idx += 1
            next_op_id += 1
            outstanding += 1

        rep = comm.recv(source=root_rank, tag=TAG)
        if rep["op"] != OP_INS_REPLY:
            raise RuntimeError(f"Unexpected message during insert phase: {rep}")
        outstanding -= 1

    return inserted



def validate_inserts(comm, root_rank, root_id, inserted, window):
    unique_keys = sorted(set(inserted))
    next_op_id = 1
    idx = 0
    outstanding = 0
    found = 0
    missing = []

    while idx < len(unique_keys) or outstanding > 0:
        while idx < len(unique_keys) and outstanding < window:
            k = unique_keys[idx]
            comm.send(
                {"op": OP_FIND, "op_id": next_op_id, "key": k, "node_id": root_id, "reply_to": 0},
                dest=root_rank,
                tag=TAG,
            )
            idx += 1
            next_op_id += 1
            outstanding += 1

        rep = comm.recv(source=root_rank, tag=TAG)
        if rep["op"] != OP_FIND_REPLY:
            raise RuntimeError(f"Unexpected validation message: {rep}")
        check_key = unique_keys[found + len(missing)]
        if rep["found"]:
            found += 1
        else:
            missing.append(check_key)
        outstanding -= 1

    return unique_keys, found, missing



def server(comm, size, inserts, key_range, seed, B, window, random_order, validate):
    if size < 2:
        raise SystemExit("Need >=2 ranks")

    root_rank, root_id = init_fixed_height_tree(comm, size)

    rng = random.Random(seed)
    keys = list(range(inserts))
    if key_range > inserts:
        keys = rng.sample(range(key_range), inserts)
    if random_order:
        rng.shuffle(keys)

    inserted = run_insert_phase(comm, root_rank, root_id, keys, window)

    if validate:
        unique_keys, found, missing = validate_inserts(comm, root_rank, root_id, inserted, window)
        print(f"[rank0] validate-inserts unique={len(unique_keys)} found={found} missing={len(missing)}")
        if missing:
            print(f"[rank0] first missing keys: {missing[:10]}")
            print("[rank0] insert validation FAILED")
        else:
            print("[rank0] insert validation OK")

    print(f"[rank0] summary inserts={inserts} deletes=0 removed=0 window={window} root_rank={root_rank}")

    for r in range(1, size):
        comm.send({"op": OP_STOP}, dest=r, tag=TAG)



def main():
    global LOG_MODE

    ap = argparse.ArgumentParser()
    ap.add_argument("--inserts", type=int, default=10000)
    ap.add_argument("--deletes", type=int, default=0)
    ap.add_argument("--key-range", type=int, default=10000000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--B", type=int, default=6)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--validate-inserts", action="store_true")
    ap.add_argument("--log", choices=["none", "splits", "all"], default="splits")
    args = ap.parse_args()

    if args.deletes != 0:
        raise SystemExit("This safe top-down article-style version currently supports inserts/find validation only. Use --deletes 0.")

    LOG_MODE = args.log

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    if rank == 0:
        start=time()
        server(comm, size, args.inserts, args.key_range, args.seed, args.B, args.window, args.random, args.validate_inserts)
        print("Time: ",time()-start)
    else:
        processor(comm, rank, args.B)


if __name__ == "__main__":
    main()


