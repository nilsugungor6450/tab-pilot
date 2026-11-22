"""Runs TabPFN in a separate process so the game loop and inference do not compete for the GIL.

Protocol over stdin/stdout (pickle): ("fit", X, y) -> number of context rows, ("probs", X) -> (N, 5).
"""
import os
import pickle
import select
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


class Brain:
    def __init__(self):
        self.proc = subprocess.Popen([sys.executable, os.path.join(HERE, "brain.py")],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=HERE)
        self.busy = False

    def request(self, *msg):
        if self.busy:
            self.result()
        pickle.dump(msg, self.proc.stdin)
        self.proc.stdin.flush()
        self.busy = True

    def ready(self):
        return self.busy and bool(select.select([self.proc.stdout], [], [], 0)[0])

    def result(self):
        res = pickle.load(self.proc.stdout)
        self.busy = False
        if isinstance(res, Exception):
            raise res
        return res

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.terminate()
        except Exception:
            pass


def serve():
    proto = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    from policy import TabPFNPolicy
    import tabpfn

    pol = None
    inp = sys.stdin.buffer
    while True:
        try:
            msg = pickle.load(inp)
        except EOFError:
            break
        try:
            if msg[0] == "fit":
                pol = TabPFNPolicy().fit(msg[1], msg[2])
                res = pol.n_context
            elif msg[0] == "probs":
                res = pol.probs(msg[1])
            else:
                res = ValueError(f"unknown command {msg[0]}")
        except Exception as e:
            res = e
        pickle.dump(res, proto)
        proto.flush()


if __name__ == "__main__":
    serve()
