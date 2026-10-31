"""
Session logic for the stream processor, kept free of Spark so it can be unit-tested.

Rule: events of one user_session arrive in Kafka-offset order. A gap of more than
GAP_MIN minutes ends the session; later events of that user_session are ignored.
When a session reaches its K-th event with no purchase among the first K, a "pred"
row is emitted; when a predicted session ends, a "close" row with the label follows.
"""
import math

import pandas as pd

K = 3
GAP_MIN = 30
GAP_MS = GAP_MIN * 60 * 1000
RETENTION_MS = 7 * 24 * 3600 * 1000      # keep a closed session as a tombstone for 7 days of event time
OPEN, CLOSED = 0, 1

OUTPUT_SCHEMA = ("kind string, session_id string, t_event long, t_pred long, n_events int, "
                 "f_c double, f_avg_price double, f_max_price double, f_dur double, y int")
STATE_SCHEMA = ("status int, n int, t_first long, t_last long, t_k long, c int, ps double, pc int, "
                "pmax double, buy_k boolean, y boolean, predicted boolean")
OUT_COLS = ["kind", "session_id", "t_event", "t_pred", "n_events",
            "f_c", "f_avg_price", "f_max_price", "f_dur", "y"]
FEATURES = ["c", "avg_price", "max_price", "dur"]


def make_features(c, ps, pc, pmax, dur_s):
    """The four model features, identical to the v1 pipeline."""
    return {
        "c": float(c),
        "avg_price": math.log1p(ps / pc) if pc > 0 else 0.0,
        "max_price": math.log1p(pmax),
        "dur": math.log1p(max(dur_s, 0.0)),
    }


class Session:
    __slots__ = ("status", "n", "t_first", "t_last", "t_k", "c", "ps", "pc", "pmax", "buy_k", "y", "predicted")

    def __init__(self, values=None):
        if values is None:
            values = (OPEN, 0, 0, 0, 0, 0, 0.0, 0, 0.0, False, False, False)
        for name, v in zip(self.__slots__, values):
            setattr(self, name, v)

    def to_tuple(self):
        return tuple(getattr(self, name) for name in self.__slots__)

    def row(self, kind, sid):
        x = make_features(self.c, self.ps, self.pc, self.pmax, (self.t_k - self.t_first) / 1000)
        t_event = self.t_k if kind == "pred" else self.t_last + GAP_MS
        y = int(self.y) if kind == "close" else -1
        return (kind, sid, t_event, self.t_k, self.n, x["c"], x["avg_price"], x["max_price"], x["dur"], y)

    def add(self, ts, et, price, out, sid):
        """Apply one event. Returns False if the event is ignored."""
        if self.status == CLOSED:
            return False
        if self.n > 0 and ts - self.t_last > GAP_MS:
            self.close(out, sid)
            return False
        self.n += 1
        if self.n == 1:
            self.t_first = ts
        self.t_last = ts
        if self.n <= K:
            if et == "purchase":
                self.buy_k = True
            else:
                if et == "cart":
                    self.c += 1
                if price == price and price > 0:       # not NaN and positive
                    self.ps += price
                    self.pc += 1
                    self.pmax = max(self.pmax, price)
            if self.n == K:
                self.t_k = ts
                if not self.buy_k:
                    self.predicted = True
                    out.append(self.row("pred", sid))
        elif et == "purchase":
            self.y = True
        return True

    def close(self, out, sid):
        if self.status == OPEN and self.predicted:
            out.append(self.row("close", sid))
        self.status = CLOSED


def update_session(key, pdfs, state):
    """Function for applyInPandasWithState (event-time timeout)."""
    sid = key[0]
    out = []
    s = Session(tuple(state.get)) if state.exists else Session()
    if state.hasTimedOut:
        s.close(out, sid)
    else:
        pdf = pd.concat(list(pdfs), ignore_index=True).sort_values("kafka_offset", kind="stable")
        for ts, et, price in zip(pdf["ts_ms"].tolist(), pdf["event_type"].tolist(), pdf["price"].tolist()):
            s.add(int(ts), et, price, out, sid)

    wm = state.getCurrentWatermarkMs()
    if s.status == OPEN and s.t_last + GAP_MS <= wm:   # the gap is already over in event time
        s.close(out, sid)
    if s.status == OPEN:
        state.update(s.to_tuple())
        state.setTimeoutTimestamp(s.t_last + GAP_MS)
    elif s.t_last + RETENTION_MS > wm:
        state.update(s.to_tuple())
        state.setTimeoutTimestamp(s.t_last + RETENTION_MS)
    else:
        state.remove()
    yield pd.DataFrame(out, columns=OUT_COLS)
