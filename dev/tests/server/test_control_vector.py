import json
import unittest

from dev.tests.server_fixtures import FakeRuntime, Harness, Plan
from server import protocol as wire
from server import request_fields

CHAT = ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}]})
COMPLETION = ("/v1/completions", {"prompt": "Hi"})
SKIP = wire.RequestFlag.SKIP_CONTROL_VECTOR
KEY = bytes(range(32))


class ControlVectorTests(unittest.TestCase):
    def harness(self, *plans, **options):
        runtime = FakeRuntime(*plans)
        harness = Harness(runtime, **options)
        self.addCleanup(harness.close)
        return harness, runtime

    def send(self, harness, path, body, fields):
        return harness.request("POST", path, {**body, "max_tokens": 1, **fields})

    def test_a_request_leaves_the_vector_out_under_its_stock_salt(self):
        harness, runtime = self.harness(
            *(Plan([[4]], reason="length") for _ in range(8)), control_vector_key=KEY
        )
        for path, body in (CHAT, COMPLETION):
            for fields in (
                {},
                {"control_vector": True},
                {"control_vector": False},
                {"control_vector": False, "cache_salt": "s"},
            ):
                with self.subTest(path=path, fields=fields):
                    status, _, payload = self.send(harness, path, body, fields)
                    self.assertEqual(status, 200, payload)
        frames = [request.frame for request in runtime.requests]
        self.assertEqual([frame.flags for frame in frames], [0, 0, SKIP, SKIP] * 2)
        projected, stock = frames[0].cache_salt, frames[2].cache_salt
        # A projected prompt never shares cache with a stock one, nor with a
        # request's own salt; one that leaves the vector out keeps its salt.
        self.assertEqual(frames[1].cache_salt, projected)
        self.assertEqual(stock, 0)
        self.assertNotIn(projected, (0, request_fields.cache_salt({"cache_salt": "s"})))
        self.assertEqual(
            frames[3].cache_salt, request_fields.cache_salt({"cache_salt": "s"})
        )

    def test_without_a_vector_only_false_is_accepted(self):
        harness, runtime = self.harness(Plan([[4]], reason="length"))
        status, _, payload = self.send(harness, *CHAT, {"control_vector": False})
        self.assertEqual(status, 200, payload)
        self.assertEqual(
            (runtime.requests[0].frame.flags, runtime.requests[0].frame.cache_salt),
            (0, 0),
        )
        for value, message in (
            (True, "control_vector needs a server started with --control-vector"),
            (1, "control_vector must be a boolean"),
            ("false", "control_vector must be a boolean"),
        ):
            with self.subTest(value=value):
                status, _, payload = self.send(
                    harness, *CHAT, {"control_vector": value}
                )
                self.assertEqual(status, 400)
                self.assertEqual(json.loads(payload)["error"]["message"], message)
        self.assertEqual(len(runtime.requests), 1)


if __name__ == "__main__":
    unittest.main()
