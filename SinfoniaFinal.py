import argparse, bisect, random
from collections import deque
from mpi4py import MPI
from time import time

INS, INS_BATCH, DEL, FIND, READ, PREP, COMM, ABRT, STOP = "INS", "INS_BATCH", "DEL", "FIND", "READ", "PREP", "COMM", "ABRT", "STOP"  # message types
OK, REDIR, SPLIT, MERGE, YES, NO, ACK, FOUND = "OK", "REDIR", "SPLIT", "MERGE", "YES", "NO", "ACK", "FOUND"  # response types
OP_TAG, CTRL_TAG = 1, 2  # separate data-path ops from control/transaction traffic


def log_ok(mode, category):
    return mode == "all" or (mode == "splits" and category == "splits")


class Leaf:
    def __init__(self, page_id, high_key, right_link=None, keys=None, values=None, version=0, lock=None):
        self.page_id = page_id  # local page id
        self.high_key = high_key  # max key in this leaf
        self.right_link = right_link  # pointer to next leaf
        self.keys = keys or []  # keys stored in leaf
        self.values = values or []  # values stored in leaf
        self.version = version  # page version
        self.lock = lock  # transaction holding lock


class IndexLeaf:
    def __init__(self, entries):
        self.entries = entries  # [(high_key, rank, page_id), ...] sorted by high_key
        self.max_keys = [entry[0] for entry in entries]

class InternalNode:
    def __init__(self, children):
        self.children = children
        self.max_keys = [child.max_keys[-1] for child in children]


class Server:
    def __init__(self, rank, comm, leaf_capacity):
        self.rank = rank  # server rank
        self.comm = comm  # MPI communicator
        self.leaf_capacity = leaf_capacity  # max keys per leaf
        self.pages = {}  # local pages
        self.next_page_id = 1  # next local page id
        self.head_page_id = self.new_leaf(10**18).page_id  # first leaf page id
        self.staged_transactions = {}  # prepared transactions

    def new_leaf(self, high_key, right_link=None, page_id=None, keys=None, values=None, version=0):
        if page_id is None:
            page_id = self.next_page_id
            self.next_page_id += 1
        self.next_page_id = max(self.next_page_id, page_id + 1)
        leaf = Leaf(page_id, high_key, right_link, keys, values, version)
        self.pages[page_id] = leaf
        return leaf

    def find_leaf(self, key, start_page_id=None):
        if start_page_id is None:
            start_page_id = self.head_page_id

        leaf = self.pages[start_page_id]
        while key > leaf.high_key and leaf.right_link is not None:
            next_rank, next_page_id = leaf.right_link
            if next_rank != self.rank:
                return leaf
            leaf = self.pages[next_page_id]
        return leaf

    def insert_local(self, leaf, key, value):
        insert_index = bisect.bisect_left(leaf.keys, key)
        if insert_index < len(leaf.keys) and leaf.keys[insert_index] == key:
            leaf.values[insert_index] = value
            return False
        leaf.keys.insert(insert_index, key)
        leaf.values.insert(insert_index, value)
        return True

    def delete_local(self, leaf, key):
        delete_index = bisect.bisect_left(leaf.keys, key)
        if delete_index < len(leaf.keys) and leaf.keys[delete_index] == key:
            leaf.keys.pop(delete_index)
            leaf.values.pop(delete_index)

    def split_plan(self, leaf, num_servers):
        middle = len(leaf.keys) // 2
        left_keys, right_keys = leaf.keys[:middle], leaf.keys[middle:]
        left_values, right_values = leaf.values[:middle], leaf.values[middle:]
        target_rank = 1 + (self.rank % num_servers)

        return {
            "left": {
                "rank": self.rank,
                "pid": leaf.page_id,
                "exp": leaf.version,
                "page": {
                    "pid": leaf.page_id,
                    "keys": left_keys,
                    "vals": left_values,
                    "high": left_keys[-1],
                    "right": None,
                    "ver": leaf.version,
                },
            },
            "right": {
                "rank": target_rank,
                "keys": right_keys,
                "vals": right_values,
                "high": leaf.high_key,
                "right": leaf.right_link,
            },
        }

    def merge_plan(self, leaf, right_leaf):
        merged_keys = leaf.keys + right_leaf.keys
        merged_values = leaf.values + right_leaf.values

        return {
            "rank": self.rank,
            "pid": leaf.page_id,
            "exp": leaf.version,
            "removed": {
                "rank": self.rank,
                "pid": right_leaf.page_id,
            },
            "page": {
                "pid": leaf.page_id,
                "keys": merged_keys,
                "vals": merged_values,
                "high": right_leaf.high_key,
                "right": right_leaf.right_link,
                "ver": leaf.version,
            },
        }

    def prepare(self, transaction_id, operations):
        locked_pages = []
        for operation in sorted(operations, key=lambda item: item["pid"]):
            page_id, expected_version = operation["pid"], operation["exp"]
            if expected_version != -1:
                page = self.pages.get(page_id)
                if page is None or (page.lock is not None and page.lock != transaction_id):
                    for locked_page_id in locked_pages:
                        self.pages[locked_page_id].lock = None
                    return {"k": NO}
                page.lock = transaction_id
                locked_pages.append(page_id)

        for operation in operations:
            page_id, expected_version = operation["pid"], operation["exp"]
            if expected_version == -1 and page_id in self.pages:
                for locked_page_id in locked_pages:
                    self.pages[locked_page_id].lock = None
                return {"k": NO}
            if expected_version != -1 and self.pages[page_id].version != expected_version:
                for locked_page_id in locked_pages:
                    self.pages[locked_page_id].lock = None
                return {"k": NO}

        self.staged_transactions[transaction_id] = operations
        return {"k": YES}

    def commit(self, transaction_id):
        operations = self.staged_transactions.pop(transaction_id, [])
        for operation in operations:
            if operation["t"] == "PUT":
                page_data = operation["page"]
                old_lock = self.pages[page_data["pid"]].lock if page_data["pid"] in self.pages else None
                self.pages[page_data["pid"]] = Leaf(
                    page_data["pid"],
                    page_data["high"],
                    tuple(page_data["right"]) if page_data["right"] is not None else None,
                    list(page_data["keys"]),
                    list(page_data["vals"]),
                    page_data["ver"] + 1,
                    old_lock,
                )
                self.next_page_id = max(self.next_page_id, page_data["pid"] + 1)

        for page in self.pages.values():
            if page.lock == transaction_id:
                page.lock = None
        return {"k": ACK}

    def abort(self, transaction_id):
        self.staged_transactions.pop(transaction_id, None)
        for page in self.pages.values():
            if page.lock == transaction_id:
                page.lock = None
        return {"k": ACK}

    def loop(self):
        num_servers = self.comm.Get_size() - 1
        min_keys = max(1, self.leaf_capacity // 2)

        while True:
            status = MPI.Status()
            message = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            reply_tag = status.Get_tag()
            message_type = message["t"]

            if message_type == STOP:
                self.comm.send({"k": ACK}, dest=0, tag=reply_tag)
                return
            if message_type == PREP:
                self.comm.send(self.prepare(message["tid"], message["ops"]), dest=0, tag=reply_tag)
                continue
            if message_type == COMM:
                self.comm.send(self.commit(message["tid"]), dest=0, tag=reply_tag)
                continue
            if message_type == ABRT:
                self.comm.send(self.abort(message["tid"]), dest=0, tag=reply_tag)
                continue

            if message_type == READ:
                start_page_id = message.get("start_pid")
                leaf = self.pages[start_page_id]
                self.comm.send({"k": FOUND, "keys": list(leaf.keys)}, dest=0, tag=reply_tag)
                continue

            if message_type == INS_BATCH:
                items = message["items"]
                start_page_id = message.get("start_pid")
                first_key = items[0][0]
                leaf = self.find_leaf(first_key, start_page_id)
                done = 0

                while done < len(items):
                    batch_key, batch_value = items[done]

                    if leaf.high_key is not None and batch_key > leaf.high_key and leaf.right_link is not None:
                        self.comm.send({"k": REDIR, "to": leaf.right_link, "done": done, "req": message.get("req")}, dest=0, tag=reply_tag)
                        break

                    insert_index = bisect.bisect_left(leaf.keys, batch_key)
                    key_exists = insert_index < len(leaf.keys) and leaf.keys[insert_index] == batch_key
                    if not key_exists and len(leaf.keys) >= self.leaf_capacity:
                        self.comm.send({"k": SPLIT, "plan": self.split_plan(leaf, num_servers), "done": done, "req": message.get("req")}, dest=0, tag=reply_tag)
                        break

                    self.insert_local(leaf, batch_key, batch_value)
                    done += 1

                if done == len(items):
                    self.comm.send({"k": OK, "done": done, "req": message.get("req")}, dest=0, tag=reply_tag)
                continue

            key = message["key"]
            start_page_id = message.get("start_pid")
            leaf = self.find_leaf(key, start_page_id)

            if leaf.high_key is not None and key > leaf.high_key and leaf.right_link is not None:
                self.comm.send({"k": REDIR, "to": leaf.right_link, "req": message.get("req")}, dest=0, tag=reply_tag)
                continue

            if message_type == FIND:
                find_index = bisect.bisect_left(leaf.keys, key)
                found = find_index < len(leaf.keys) and leaf.keys[find_index] == key
                self.comm.send({"k": FOUND, "found": found, "req": message.get("req")}, dest=0, tag=reply_tag)
                continue

            if message_type == INS:
                value = message["val"]
                insert_index = bisect.bisect_left(leaf.keys, key)
                key_exists = insert_index < len(leaf.keys) and leaf.keys[insert_index] == key

                # Predictive split: if a new key would overflow the leaf, do not mutate the page first.
                # Let rank0 commit the split transaction, then retry the insert against the updated route.
                if not key_exists and len(leaf.keys) >= self.leaf_capacity:
                    self.comm.send({"k": SPLIT, "plan": self.split_plan(leaf, num_servers), "req": message.get("req")}, dest=0, tag=reply_tag)
                    continue

                self.insert_local(leaf, key, value)
                self.comm.send({"k": OK, "req": message.get("req")}, dest=0, tag=reply_tag)
                continue

            if message_type == DEL:
                self.delete_local(leaf, key)

                if leaf.right_link is not None and len(leaf.keys) < min_keys:
                    next_rank, next_page_id = leaf.right_link

                    if next_rank == self.rank:
                        right_leaf = self.pages[next_page_id]

                        if len(leaf.keys) + len(right_leaf.keys) <= self.leaf_capacity:
                            self.comm.send({"k": MERGE, "plan": self.merge_plan(leaf, right_leaf), "req": message.get("req")}, dest=0, tag=reply_tag)
                            continue

                self.comm.send({"k": OK, "req": message.get("req")}, dest=0, tag=reply_tag)
                continue


class Coordinator:
    def __init__(self, comm, log_mode, leaf_capacity):
        self.comm = comm  # MPI communicator
        self.log_mode = log_mode  # logging mode
        self.world_size = comm.Get_size()  # total process count
        self.transaction_id = 1000  # next transaction id
        self.next_page_id = {rank: 1_000_000_000 for rank in range(1, self.world_size)}  # page ids by rank
        self.route_index = [(10**18, 1, 1)]  # sorted leaf directory: (high_key, rank, page_id)
        self.index_fanout = max(8, min(leaf_capacity, 64))  # coordinator-side internal-node fanout
        self.index_root = None  # coordinator-side B+-tree internal nodes over route_index
        self.rebuild_index_tree()
        self.splits = 0  # split counter
        self.redirects = 0  # redirect counter
        self.tx_commits = 0  # committed tx count
        self.tx_aborts = 0  # aborted tx count
        self.next_request_id = 1  # next async op request id
        self.leaf_capacity = leaf_capacity
        self.mutating_ops_per_leaf = 1  # robustness first: avoid split livelock on the same leaf

    def rpc(self, rank, message):
        self.comm.send(message, dest=rank, tag=CTRL_TAG)
        return self.comm.recv(source=rank, tag=CTRL_TAG)

    def alloc_pid(self, rank):
        new_page_id = self.next_page_id[rank]
        self.next_page_id[rank] += 1
        return new_page_id

    def rebuild_index_tree(self):
        entries = list(self.route_index)
        if not entries:
            self.index_root = None
            return

        level = [IndexLeaf(entries[i:i + self.index_fanout]) for i in range(0, len(entries), self.index_fanout)]
        while len(level) > 1:
            level = [InternalNode(level[i:i + self.index_fanout]) for i in range(0, len(level), self.index_fanout)]
        self.index_root = level[0]

    def find_start_leaf(self, key):
        if self.index_root is None:
            raise RuntimeError('empty route index')

        node = self.index_root
        while isinstance(node, InternalNode):
            child_index = bisect.bisect_left(node.max_keys, key)
            if child_index >= len(node.children):
                child_index = len(node.children) - 1
            node = node.children[child_index]

        entry_index = bisect.bisect_left(node.max_keys, key)
        if entry_index >= len(node.entries):
            entry_index = len(node.entries) - 1
        _, rank, page_id = node.entries[entry_index]
        return rank, page_id

    def update_route_index_after_split(self, left_info, right_info, new_page_id):
        old_entry = None
        for entry in self.route_index:
            if entry[1] == left_info["rank"] and entry[2] == left_info["pid"]:
                old_entry = entry
                break

        if old_entry is not None:
            self.route_index.remove(old_entry)

        left_high = left_info["page"]["high"]
        right_high = right_info["high"]

        bisect.insort(self.route_index, (left_high, left_info["rank"], left_info["pid"]))
        bisect.insort(self.route_index, (right_high, right_info["rank"], new_page_id))
        self.rebuild_index_tree()

    def update_route_index_after_merge(self, merge_info):
        merged_rank = merge_info["rank"]
        merged_pid = merge_info["pid"]
        merged_high = merge_info["page"]["high"]

        removed_rank = merge_info["removed"]["rank"]
        removed_pid = merge_info["removed"]["pid"]

        old_left = None
        old_right = None

        for entry in self.route_index:
            if entry[1] == merged_rank and entry[2] == merged_pid:
                old_left = entry
            if entry[1] == removed_rank and entry[2] == removed_pid:
                old_right = entry

        if old_left is not None:
            self.route_index.remove(old_left)
        if old_right is not None:
            self.route_index.remove(old_right)

        bisect.insort(self.route_index, (merged_high, merged_rank, merged_pid))
        self.rebuild_index_tree()

    def txn(self, operations_by_rank):
        self.transaction_id += 1
        current_tid = self.transaction_id
        ranks = sorted(operations_by_rank)

        for rank in ranks:
            if self.rpc(rank, {"t": PREP, "tid": current_tid, "ops": operations_by_rank[rank]})["k"] != YES:
                self.tx_aborts += 1
                for abort_rank in ranks:
                    self.rpc(abort_rank, {"t": ABRT, "tid": current_tid})
                return False

        for rank in ranks:
            self.rpc(rank, {"t": COMM, "tid": current_tid})

        self.tx_commits += 1
        return True

    def _handle_insert_result(self, state, result):
        if result["k"] == OK:
            return True

        if result["k"] == REDIR:
            self.redirects += 1
            state["current_rank"], state["start_page_id"] = result["to"]
            state["hops"] += 1
            return False

        if result["k"] == SPLIT:
            self.splits += 1
            left_info, right_info = result["plan"]["left"], result["plan"]["right"]
            new_page_id = self.alloc_pid(right_info["rank"])

            left_page = left_info["page"]
            left_page["right"] = (right_info["rank"], new_page_id)

            right_page = {
                "pid": new_page_id,
                "keys": right_info["keys"],
                "vals": right_info["vals"],
                "high": right_info["high"],
                "right": right_info["right"],
                "ver": 0,
            }

            if log_ok(self.log_mode, "splits"):
                print(
                    f"[rank0] SPLIT key={state['key']} left=({left_info['rank']},{left_info['pid']}) "
                    f"-> right=({right_info['rank']},{new_page_id})"
                )

            ok = self.txn({
                left_info["rank"]: [{"t": "PUT", "pid": left_info["pid"], "exp": left_info["exp"], "page": left_page}],
                right_info["rank"]: [{"t": "PUT", "pid": new_page_id, "exp": -1, "page": right_page}],
            })
            if ok:
                self.update_route_index_after_split(left_info, right_info, new_page_id)
            else:
                state["retries"] += 1

            state["current_rank"], state["start_page_id"] = self.find_start_leaf(state["key"])
            state["hops"] = 0
            return False

        raise RuntimeError(f"unexpected insert response: {result}")

    def _handle_delete_result(self, state, result):
        if result["k"] == OK:
            return True

        if result["k"] == REDIR:
            self.redirects += 1
            state["current_rank"], state["start_page_id"] = result["to"]
            state["hops"] += 1
            return False

        if result["k"] == MERGE:
            merge_info = result["plan"]
            ok = self.txn({
                merge_info["rank"]: [{
                    "t": "PUT",
                    "pid": merge_info["pid"],
                    "exp": merge_info["exp"],
                    "page": merge_info["page"],
                }]
            })
            if not ok:
                state["retries"] += 1

            self.update_route_index_after_merge(merge_info)
            state["current_rank"], state["start_page_id"] = self.find_start_leaf(state["key"])
            state["hops"] = 0
            return False

        raise RuntimeError(f"unexpected delete response: {result}")

    def alloc_req(self):
        request_id = self.next_request_id
        self.next_request_id += 1
        return request_id

    def _run_ops_concurrent(self, operations, kind, max_hops=100000, max_retries=200, window=None):
        if not operations:
            return

        max_parallel = max(1, window if window is not None else (self.world_size - 1))

        pending = deque()
        for op in operations:
            state = {
                "kind": kind,
                "key": op["key"],
                "value": op.get("value"),
                "retries": 0,
                "hops": 0,
            }
            state["current_rank"], state["start_page_id"] = self.find_start_leaf(state["key"])
            pending.append(state)

        active = {}
        active_per_rank = {rank: 0 for rank in range(1, self.world_size)}
        active_leaf_counts = {}  # only one mutating op per known leaf to avoid split storms
        per_rank_limit = max(1, (max_parallel + (self.world_size - 2)) // max(1, self.world_size - 1))

        while pending or active:
            launched = True
            while launched and len(active) < max_parallel and pending:
                launched = False
                pending_count = len(pending)
                for _ in range(pending_count):
                    state = pending.popleft()
                    rank = state["current_rank"]

                    if state["retries"] >= max_retries:
                        raise RuntimeError(f"{kind} failed after too many retries for key={state['key']}")
                    if state["hops"] >= max_hops:
                        raise RuntimeError(f"{kind} exceeded max hops for key={state['key']}")

                    if active_per_rank[rank] >= per_rank_limit:
                        pending.append(state)
                        continue

                    leaf_key = (rank, state["start_page_id"])
                    if kind != "find" and state["start_page_id"] is not None and active_leaf_counts.get(leaf_key, 0) >= self.mutating_ops_per_leaf:
                        pending.append(state)
                        continue

                    message_type = INS if kind == "insert" else DEL if kind == "delete" else FIND
                    request_id = self.alloc_req()
                    message = {"t": message_type, "key": state["key"], "req": request_id}
                    if kind == "insert":
                        message["val"] = state["value"]
                    if state["start_page_id"] is not None:
                        message["start_pid"] = state["start_page_id"]

                    self.comm.send(message, dest=rank, tag=OP_TAG)
                    active[request_id] = (state, leaf_key)
                    active_per_rank[rank] += 1
                    if kind != "find" and state["start_page_id"] is not None:
                        active_leaf_counts[leaf_key] = active_leaf_counts.get(leaf_key, 0) + 1
                    launched = True
                    if len(active) >= max_parallel:
                        break

            if not active:
                raise RuntimeError(f"deadlock while scheduling {kind} operations")

            status = MPI.Status()
            result = self.comm.recv(source=MPI.ANY_SOURCE, tag=OP_TAG, status=status)
            rank = status.Get_source()
            request_id = result.get("req")
            if request_id not in active:
                raise RuntimeError(f"unknown {kind} response req={request_id}: {result}")
            state, leaf_key = active.pop(request_id)
            active_per_rank[rank] -= 1
            if kind != "find" and state["start_page_id"] is not None:
                remaining = active_leaf_counts.get(leaf_key, 0) - 1
                if remaining > 0:
                    active_leaf_counts[leaf_key] = remaining
                else:
                    active_leaf_counts.pop(leaf_key, None)

            if kind == "insert":
                done = self._handle_insert_result(state, result)
            elif kind == "delete":
                done = self._handle_delete_result(state, result)
            else:
                if result["k"] == FOUND:
                    state["found"] = result["found"]
                    done = True
                elif result["k"] == REDIR:
                    self.redirects += 1
                    state["current_rank"], state["start_page_id"] = result["to"]
                    state["hops"] += 1
                    done = False
                else:
                    raise RuntimeError(f"unexpected find response: {result}")

            if not done:
                pending.append(state)

    def find_many(self, keys, max_hops=100000, max_retries=200, window=None):
        if not keys:
            return {}

        max_parallel = max(1, window if window is not None else (self.world_size - 1))

        pending = deque()
        for key in keys:
            state = {"key": key, "retries": 0, "hops": 0}
            state["current_rank"], state["start_page_id"] = self.find_start_leaf(key)
            pending.append(state)

        active = {}
        active_per_rank = {rank: 0 for rank in range(1, self.world_size)}
        per_rank_limit = max(1, (max_parallel + (self.world_size - 2)) // max(1, self.world_size - 1))
        results = {}

        while pending or active:
            launched = True
            while launched and len(active) < max_parallel and pending:
                launched = False
                pending_count = len(pending)
                for _ in range(pending_count):
                    state = pending.popleft()
                    rank = state["current_rank"]

                    if state["retries"] >= max_retries:
                        raise RuntimeError(f"find failed after too many retries for key={state['key']}")
                    if state["hops"] >= max_hops:
                        raise RuntimeError(f"find exceeded max hops for key={state['key']}")

                    if active_per_rank[rank] >= per_rank_limit:
                        pending.append(state)
                        continue

                    request_id = self.alloc_req()
                    message = {"t": FIND, "key": state["key"], "req": request_id}
                    if state["start_page_id"] is not None:
                        message["start_pid"] = state["start_page_id"]
                    self.comm.send(message, dest=rank, tag=OP_TAG)
                    active[request_id] = state
                    active_per_rank[rank] += 1
                    launched = True
                    if len(active) >= max_parallel:
                        break

            if not active:
                raise RuntimeError("deadlock while scheduling find operations")

            status = MPI.Status()
            result = self.comm.recv(source=MPI.ANY_SOURCE, tag=OP_TAG, status=status)
            rank = status.Get_source()
            request_id = result.get("req")
            if request_id not in active:
                raise RuntimeError(f"unknown find response req={request_id}: {result}")
            state = active.pop(request_id)
            active_per_rank[rank] -= 1

            if result["k"] == FOUND:
                results[state["key"]] = result["found"]
            elif result["k"] == REDIR:
                self.redirects += 1
                state["current_rank"], state["start_page_id"] = result["to"]
                state["hops"] += 1
                pending.append(state)
            else:
                raise RuntimeError(f"unexpected find response: {result}")

        return results

    def collect_all_keys(self):
        all_keys = []
        for _, rank, page_id in self.route_index:
            result = self.rpc(rank, {"t": READ, "start_pid": page_id})
            all_keys.extend(result.get("keys", []))
        return set(all_keys)

    def _dedup_last_wins(self, items):
        seen = set()
        prepared = []
        for key, value in reversed(items):
            if key in seen:
                continue
            seen.add(key)
            prepared.append((key, value))
        prepared.reverse()
        return prepared

    def _prepare_insert_batches(self, items, batch_size):
        prepared = self._dedup_last_wins(items)
        if not prepared:
            return []

        keys = [key for key, _ in prepared]
        min_key = min(keys)
        max_key = max(keys)
        if min_key == max_key:
            return [prepared[i:i + batch_size] for i in range(0, len(prepared), batch_size)]

        bucket_count = max(8, 8 * (self.world_size - 1))
        width = max(1, ((max_key - min_key) + bucket_count) // bucket_count)
        buckets = [[] for _ in range(bucket_count)]
        for key, value in prepared:
            bucket_index = min(bucket_count - 1, (key - min_key) // width)
            buckets[bucket_index].append((key, value))

        bucket_batches = []
        for bucket in buckets:
            if not bucket:
                bucket_batches.append([])
                continue
            bucket.sort(key=lambda item: item[0])
            bucket_batches.append([bucket[i:i + batch_size] for i in range(0, len(bucket), batch_size)])

        batches = []
        more = True
        while more:
            more = False
            for chunk_list in bucket_batches:
                if chunk_list:
                    batches.append(chunk_list.pop(0))
                    more = True
        return batches

    def _handle_insert_batch_result(self, state, result):
        done = result.get("done", 0)
        if done:
            state["items"] = state["items"][done:]

        if not state["items"]:
            return True

        if result["k"] == OK:
            return True

        if result["k"] == REDIR:
            self.redirects += 1
            state["current_rank"], state["start_page_id"] = result["to"]
            state["hops"] += 1
            return False

        if result["k"] == SPLIT:
            self.splits += 1
            left_info, right_info = result["plan"]["left"], result["plan"]["right"]
            new_page_id = self.alloc_pid(right_info["rank"])

            left_page = left_info["page"]
            left_page["right"] = (right_info["rank"], new_page_id)

            right_page = {
                "pid": new_page_id,
                "keys": right_info["keys"],
                "vals": right_info["vals"],
                "high": right_info["high"],
                "right": right_info["right"],
                "ver": 0,
            }

            if log_ok(self.log_mode, "splits"):
                print(
                    f"[rank0] SPLIT key={state['items'][0][0]} left=({left_info['rank']},{left_info['pid']}) "
                    f"-> right=({right_info['rank']},{new_page_id})"
                )

            ok = self.txn({
                left_info["rank"]: [{"t": "PUT", "pid": left_info["pid"], "exp": left_info["exp"], "page": left_page}],
                right_info["rank"]: [{"t": "PUT", "pid": new_page_id, "exp": -1, "page": right_page}],
            })
            if ok:
                self.update_route_index_after_split(left_info, right_info, new_page_id)
            else:
                state["retries"] += 1

            state["current_rank"], state["start_page_id"] = self.find_start_leaf(state["items"][0][0])
            state["hops"] = 0
            return False

        raise RuntimeError(f"unexpected insert-batch response: {result}")

    def _run_insert_batches(self, batches, max_hops=100000, max_retries=200):
        if not batches:
            return

        pending = deque()
        for batch_items in batches:
            rank, page_id = self.find_start_leaf(batch_items[0][0])
            pending.append({
                "items": batch_items,
                "current_rank": rank,
                "start_page_id": page_id,
                "retries": 0,
                "hops": 0,
            })

        while pending:
            state = pending.popleft()
            while True:
                if state["retries"] >= max_retries:
                    raise RuntimeError(f"insert batch failed after too many retries near key={state['items'][0][0]}")
                if state["hops"] >= max_hops:
                    raise RuntimeError(f"insert batch exceeded max hops near key={state['items'][0][0]}")

                request_id = self.alloc_req()
                message = {
                    "t": INS_BATCH,
                    "items": state["items"],
                    "key": state["items"][0][0],
                    "req": request_id,
                }
                if state["start_page_id"] is not None:
                    message["start_pid"] = state["start_page_id"]
                self.comm.send(message, dest=state["current_rank"], tag=OP_TAG)

                result = self.comm.recv(source=state["current_rank"], tag=OP_TAG)
                if result.get("req") != request_id:
                    raise RuntimeError(f"unknown insert-batch response req={result.get('req')}: {result}")

                done = self._handle_insert_batch_result(state, result)
                if done:
                    break

    def insert_many(self, items, max_hops=100000, max_retries=200, window=None):
        batch_size = max(64, min(1024, self.leaf_capacity))
        batches = self._prepare_insert_batches(items, batch_size=batch_size)
        self._run_insert_batches(batches, max_hops=max_hops, max_retries=max_retries)

    def delete_many(self, keys, max_hops=100000, max_retries=200, window=None):
        operations = [{"key": key} for key in keys]
        self._run_ops_concurrent(operations, "delete", max_hops=max_hops, max_retries=max_retries, window=window)

    def index_height(self):
        if self.index_root is None:
            return 0
        height = 1
        node = self.index_root
        while isinstance(node, InternalNode):
            height += 1
            node = node.children[0]
        return height

    def internal_node_count(self):
        if self.index_root is None or isinstance(self.index_root, IndexLeaf):
            return 0
        count = 0
        stack = [self.index_root]
        while stack:
            node = stack.pop()
            if isinstance(node, InternalNode):
                count += 1
                stack.extend(node.children)
        return count

    def index_leaf_node_count(self):
        if self.index_root is None:
            return 0
        if isinstance(self.index_root, IndexLeaf):
            return 1
        count = 0
        stack = [self.index_root]
        while stack:
            node = stack.pop()
            if isinstance(node, InternalNode):
                stack.extend(node.children)
            else:
                count += 1
        return count

    def root_children(self):
        if self.index_root is None:
            return 0
        if isinstance(self.index_root, InternalNode):
            return len(self.index_root.children)
        return len(self.index_root.entries)

    def stop(self):
        for rank in range(1, self.world_size):
            self.rpc(rank, {"t": STOP})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inserts", type=int, default=1000)  # number of inserts
    ap.add_argument("--deletes", type=int, default=0)  # number of deletes
    ap.add_argument("--key-range", type=int, default=10000)  # random key range
    ap.add_argument("--seed", type=int, default=42)  # random seed
    ap.add_argument("--leaf-capacity", "--B", dest="leaf_capacity", type=int, default=32)  # max keys per leaf
    ap.add_argument("--window", type=int, default=0)  # max concurrent requests from rank0; 0 => all servers
    ap.add_argument("--validate-inserts", "--validate-insert", dest="validate_inserts", action="store_true")  # validate inserted keys at the end
    ap.add_argument("--log", choices=["none", "splits", "all"], default="splits")  # logging mode
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if comm.Get_size() < 2:
        raise SystemExit("Need >=2 ranks")
    if args.window < 0:
        raise SystemExit("--window must be >= 0")

    if rank == 0:
        start=time()
        rng = random.Random(args.seed)
        coordinator = Coordinator(comm, args.log, args.leaf_capacity)
        inserted_keys = []
        items = []

        for insert_index in range(args.inserts):
            key = rng.randrange(args.key_range)
            inserted_keys.append(key)
            items.append((key, f"v{insert_index}"))

        window = args.window or (comm.Get_size() - 1)
        coordinator.insert_many(items, window=window)

        delete_count = min(args.deletes, len(inserted_keys))
        delete_keys = []
        for _ in range(delete_count):
            delete_keys.append(inserted_keys[rng.randrange(len(inserted_keys))])
        coordinator.delete_many(delete_keys, window=window)

        if args.validate_inserts:
            remaining = sorted(set(inserted_keys) - set(delete_keys))
            present = coordinator.collect_all_keys()
            present_set = set(present)
            found = sum(1 for key in remaining if key in present_set)
            missing = [key for key in remaining if key not in present_set]
            print(f"[rank0] validate-inserts unique={len(remaining)} present_pages={len(present)} found={found} missing={len(missing)}")
            if missing:
                print(f"[rank0] first missing keys: {missing[:10]}")
                print("[rank0] insert validation FAILED")
            else:
                print("[rank0] insert validation OK")

        print(
            f"[rank0] summary inserts={args.inserts} deletes={delete_count} window={window} "
            f"B={args.leaf_capacity} splits={coordinator.splits} redirects={coordinator.redirects} "
            f"tx_commits={coordinator.tx_commits} tx_aborts={coordinator.tx_aborts} "
            f"route_leaves={len(coordinator.route_index)} index_leaf_nodes={coordinator.index_leaf_node_count()} "
            f"internal_nodes={coordinator.internal_node_count()} root_children={coordinator.root_children()} index_height={coordinator.index_height()} fanout={coordinator.index_fanout}"
        )
        coordinator.stop()
        print("Time: ",time()-start)
    else:
        Server(rank, comm, args.leaf_capacity).loop()


if __name__ == "__main__":
    main()



