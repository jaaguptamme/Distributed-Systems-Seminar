from mpi4py import MPI
import argparse
import random
from time import time

TAG = 1009

OP_INIT_LEAF = "init_leaf"
OP_INIT_INTERNAL_ABOVE = "init_internal_above"
OP_SET_ROOT = "set_root"

OP_INS = "insert_bottomup"
OP_INS_REPLY = "insert_reply"

OP_DEL = "delete_key"
OP_DEL_REPLY = "delete_reply"

OP_STOP = "shutdown"

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
        self.right = None # right sibling on the same rank
        self.high_key = None  # exclusive upper bound; None means +inf


class LevelStore:
    def __init__(self, B):
        self.B = B
        self.max_degree = 2 ** B
        self.max_leaf = 2 ** B
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

    def move_right_for_key(self, nid, k):
        while True:
            n = self.nodes[nid]
            if n.high_key is None or k < n.high_key or n.right is None:
                return nid
            nid = n.right

    def insert_leaf(self, nid, k, v):
        nid = self.move_right_for_key(nid, k)
        n = self.nodes[nid]
        i = lower_bound(n.keys, k)

        if i < len(n.keys) and n.keys[i] == k:
            n.values[i] = v
        else:
            n.keys.insert(i, k)
            n.values.insert(i, v)

        return nid

    def delete_leaf(self, nid, k):
        nid = self.move_right_for_key(nid, k)
        n = self.nodes[nid]
        i = lower_bound(n.keys, k)

        if i < len(n.keys) and n.keys[i] == k:
            n.keys.pop(i)
            n.values.pop(i)
            return nid, True

        return nid, False

    def choose_child(self, nid, k):
        nid = self.move_right_for_key(nid, k)
        n = self.nodes[nid]
        i = lower_bound(n.keys, k)
        if i >= len(n.children):
            i = len(n.children) - 1
        child = n.children[i]
        return nid, child

    def split_leaf_if_needed(self, nid):
        n = self.nodes[nid]

        if (not n.is_leaf) or len(n.keys) <= self.max_leaf:
            return None, None

        mid = len(n.keys) // 2

        rid = self.new_leaf()
        r = self.nodes[rid]

        r.keys = n.keys[mid:]
        r.values = n.values[mid:]
        n.keys = n.keys[:mid]
        n.values = n.values[:mid]

        split_key = r.keys[0]

        # B-link maintenance: left keeps old upper bound only up to split_key.
        r.right = n.right
        r.high_key = n.high_key
        n.right = rid
        n.high_key = split_key

        return split_key, rid

    def split_internal_if_needed(self, nid):
        n = self.nodes[nid]

        if n.is_leaf or len(n.children) <= self.max_degree:
            return None, None

        m = len(n.children)
        mid = m // 2

        rid = self.new_internal()
        r = self.nodes[rid]

        promote = n.keys[mid - 1]

        r.children = n.children[mid:]
        r.keys = n.keys[mid:]

        n.children = n.children[:mid]
        n.keys = n.keys[:mid - 1]

        r.right = n.right
        r.high_key = n.high_key
        n.right = rid
        n.high_key = promote

        return promote, rid

    def install_child_split(self, nid, child_id, split_key, right_node):
        n = self.nodes[nid]
        pos = n.children.index(child_id)
        n.children.insert(pos + 1, right_node)
        n.keys.insert(pos, split_key)


def init_fixed_height_tree(comm, size):
    comm.send({"op": OP_INIT_LEAF}, dest=1, tag=TAG)
    rep = comm.recv(source=1, tag=TAG)
    current_root_id = rep["root_id"]

    for r in range(2, size):
        comm.send({"op": OP_INIT_INTERNAL_ABOVE, "child_id": current_root_id}, dest=r, tag=TAG)
        rep = comm.recv(source=r, tag=TAG)
        current_root_id = rep["root_id"]

    root_rank = size - 1
    root_id = current_root_id

    for r in range(1, size):
        comm.send({"op": OP_SET_ROOT, "root_rank": root_rank, "root_id": root_id}, dest=r, tag=TAG)

    return root_rank, root_id


def processor(comm, rank, B):
    store = LevelStore(B)

    root_rank = 1
    root_id = None

    pending_insert = {}
    pending_delete = {}

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
            log(rank, f"Init leaf root id={rid}", "all")
            comm.send({"root_id": rid}, dest=0, tag=TAG)
            continue

        if op == OP_INIT_INTERNAL_ABOVE:
            rid = store.new_internal()
            n = store.nodes[rid]
            n.children = [msg["child_id"]]
            n.keys = []
            log(rank, f"Init internal above child={msg['child_id']} -> id={rid}", "all")
            comm.send({"root_id": rid}, dest=0, tag=TAG)
            continue

        if op == OP_INS:
            op_id = msg["op_id"]
            k = msg["key"]
            v = msg["value"]
            nid = msg["node_id"]
            reply_to = msg["reply_to"]

            nid = store.move_right_for_key(nid, k)
            n = store.nodes[nid]

            if n.is_leaf:
                actual_leaf = store.insert_leaf(nid, k, v)
                log(rank, f"Insert leaf node={actual_leaf}, key={k}, op_id={op_id}", "all")

                sep, rid = store.split_leaf_if_needed(actual_leaf)
                if sep is not None:
                    log(rank, f"LEAF SPLIT at node={actual_leaf} -> right={rid}, split_key={sep}, op_id={op_id}", "splits")

                comm.send(
                    {
                        "op": OP_INS_REPLY,
                        "op_id": op_id,
                        "split_key": sep,
                        "right_node": rid,
                        "child_id": actual_leaf,
                    },
                    dest=reply_to,
                    tag=TAG,
                )
                continue

            actual_node, child = store.choose_child(nid, k)
            pending_insert[op_id] = {
                "reply_to": reply_to,
                "node_id": actual_node,
                "child_id": child,
            }

            log(rank, f"Forward insert op_id={op_id} from node={actual_node} to child={child} for key={k}", "all")

            comm.send(
                {
                    "op": OP_INS,
                    "op_id": op_id,
                    "key": k,
                    "value": v,
                    "node_id": child,
                    "reply_to": rank,
                },
                dest=rank - 1,
                tag=TAG,
            )
            continue

        if op == OP_INS_REPLY:
            op_id = msg["op_id"]
            if op_id not in pending_insert:
                raise RuntimeError(f"Missing pending insert context for op_id={op_id} on rank={rank}")

            ctx = pending_insert.pop(op_id)
            nid = ctx["node_id"]
            child_id = ctx["child_id"]
            reply_to = ctx["reply_to"]

            # The parent itself may have shifted right because of an earlier split.
            n_actual = store.nodes[nid]
            if n_actual.high_key is not None and msg["split_key"] is not None and msg["split_key"] >= n_actual.high_key and n_actual.right is not None:
                nid = store.move_right_for_key(nid, msg["split_key"])

            sep = msg.get("split_key")
            rid = msg.get("right_node")
            actual_child = msg.get("child_id", child_id)

            if rid is not None:
                # The original child may have moved to the right if parent split before this reply was handled.
                parent = store.nodes[nid]
                if actual_child not in parent.children:
                    probe = nid
                    while True:
                        p = store.nodes[probe]
                        if actual_child in p.children:
                            nid = probe
                            parent = p
                            break
                        if p.right is None:
                            raise RuntimeError(
                                f"Child {actual_child} not found while finishing insert op_id={op_id} on rank={rank}"
                            )
                        probe = p.right

                store.install_child_split(nid, actual_child, sep, rid)
                log(rank, f"Installed child split at node={nid}, sep={sep}, op_id={op_id}", "splits")

            sep2, rid2 = store.split_internal_if_needed(nid)
            if sep2 is not None:
                if rank == root_rank:
                    raise RuntimeError(
                        "Fixed top root overflowed. Increase MPI ranks or use larger B."
                    )
                log(rank, f"INTERNAL SPLIT at node={nid} -> right={rid2}, split_key={sep2}, op_id={op_id}", "splits")

            comm.send(
                {
                    "op": OP_INS_REPLY,
                    "op_id": op_id,
                    "split_key": sep2,
                    "right_node": rid2,
                    "child_id": nid,
                },
                dest=reply_to,
                tag=TAG,
            )
            continue

        if op == OP_DEL:
            op_id = msg["op_id"]
            k = msg["key"]
            nid = msg["node_id"]
            reply_to = msg["reply_to"]

            nid = store.move_right_for_key(nid, k)
            n = store.nodes[nid]

            if n.is_leaf:
                actual_leaf, removed = store.delete_leaf(nid, k)
                log(rank, f"Delete leaf node={actual_leaf}, key={k}, removed={removed}, op_id={op_id}", "all")

                comm.send(
                    {
                        "op": OP_DEL_REPLY,
                        "op_id": op_id,
                        "removed": removed,
                    },
                    dest=reply_to,
                    tag=TAG,
                )
                continue

            actual_node, child = store.choose_child(nid, k)
            pending_delete[op_id] = {
                "reply_to": reply_to,
                "node_id": actual_node,
                "child_id": child,
            }

            log(rank, f"Forward delete op_id={op_id} from node={actual_node} to child={child} for key={k}", "all")

            comm.send(
                {
                    "op": OP_DEL,
                    "op_id": op_id,
                    "key": k,
                    "node_id": child,
                    "reply_to": rank,
                },
                dest=rank - 1,
                tag=TAG,
            )
            continue

        if op == OP_DEL_REPLY:
            op_id = msg["op_id"]
            if op_id not in pending_delete:
                raise RuntimeError(f"Missing pending delete context for op_id={op_id} on rank={rank}")
            ctx = pending_delete.pop(op_id)

            comm.send(
                {
                    "op": OP_DEL_REPLY,
                    "op_id": op_id,
                    "removed": msg["removed"],
                },
                dest=ctx["reply_to"],
                tag=TAG,
            )
            continue


def run_insert_phase(comm, root_rank, root_id, keys, window):
    next_op_id = 1
    next_key_index = 0
    outstanding = 0
    inserted = []

    while next_key_index < len(keys) or outstanding > 0:
        while next_key_index < len(keys) and outstanding < window:
            k = keys[next_key_index]
            comm.send(
                {
                    "op": OP_INS,
                    "op_id": next_op_id,
                    "key": k,
                    "value": f"v{k}",
                    "node_id": root_id,
                    "reply_to": 0,
                },
                dest=root_rank,
                tag=TAG,
            )
            inserted.append(k)
            next_key_index += 1
            next_op_id += 1
            outstanding += 1

        rep = comm.recv(source=root_rank, tag=TAG)
        if rep["op"] != OP_INS_REPLY:
            raise RuntimeError(f"Unexpected message at server during insert phase: {rep}")
        if rep["right_node"] is not None:
            raise RuntimeError("Fixed top root overflowed. Increase MPI ranks or use larger B.")
        outstanding -= 1

    return inserted


def run_delete_phase(comm, root_rank, root_id, delete_keys, window):
    next_op_id = 1
    next_key_index = 0
    outstanding = 0
    removed_count = 0

    while next_key_index < len(delete_keys) or outstanding > 0:
        while next_key_index < len(delete_keys) and outstanding < window:
            k = delete_keys[next_key_index]
            comm.send(
                {
                    "op": OP_DEL,
                    "op_id": next_op_id,
                    "key": k,
                    "node_id": root_id,
                    "reply_to": 0,
                },
                dest=root_rank,
                tag=TAG,
            )
            next_key_index += 1
            next_op_id += 1
            outstanding += 1

        rep = comm.recv(source=root_rank, tag=TAG)
        if rep["op"] != OP_DEL_REPLY:
            raise RuntimeError(f"Unexpected message at server during delete phase: {rep}")
        if rep["removed"]:
            removed_count += 1
        outstanding -= 1

    return removed_count


def server(comm, size, B, inserts, deletes, key_range, use_random, seed, window, validate):
    root_rank, root_id = init_fixed_height_tree(comm, size)

    keys = list(range(inserts))
    rng = random.Random(seed)
    if key_range > inserts:
        keys = rng.sample(range(key_range), inserts)
    if use_random:
        rng.shuffle(keys)
        log(0, f"Random insert order enabled (seed={seed})", "splits")

    inserted = run_insert_phase(comm, root_rank, root_id, keys, window)

    if validate:
        print("[rank0] validate-inserts skipped (not implemented in bottomupParallel.py)")

    delete_count = min(deletes, len(inserted))
    delete_keys = inserted[:delete_count]
    removed_count = run_delete_phase(comm, root_rank, root_id, delete_keys, window)

    print(
        f"[rank0] summary inserts={inserts} deletes={delete_count} "
        f"removed={removed_count} window={window} root_rank={root_rank}"
    )

    for r in range(1, size):
        comm.send({"op": OP_STOP}, dest=r, tag=TAG)


def main():
    global LOG_MODE

    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=5)
    ap.add_argument("--inserts", type=int, default=1000)
    ap.add_argument("--deletes", type=int, default=0)
    ap.add_argument("--key-range", type=int, default=10000000)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--log", choices=["none", "splits", "all"], default="splits")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validate-inserts", "--validate-insert", dest="validate_inserts", action="store_true")
    args = ap.parse_args()

    if args.window < 1:
        raise SystemExit("--window must be >= 1")

    LOG_MODE = args.log

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    if size < 2:
        raise SystemExit("Need >=2 ranks")

    if rank == 0:
        start=time()
        server(
            comm,
            size,
            args.B,
            args.inserts,
            args.deletes,
            args.key_range,
            args.random,
            args.seed,
            args.window,
            args.validate_inserts,
        )
        print("Time: ",time()-start)
    else:
        processor(comm, rank, args.B)


if __name__ == "__main__":
    main()



