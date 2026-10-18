import importlib.util
import contextlib
import io
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("reference", Path(__file__).resolve().parents[1]/"tools/run_qsa_vllm_reference.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class ReferenceTest(unittest.TestCase):
    def test_token_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"ids"
            path.write_text("tokens=3\ntoken_ids=7,11,13\n")
            self.assertEqual(reference.read_tokens(path), [7, 11, 13])
            for value in ("tokens=2\ntoken_ids=7,11,13\n", "tokens=3\ntoken_ids=7,-1,13\n",
                          "tokens=1\ntoken_ids=7\n", "not a token list"):
                path.write_text(value)
                with self.assertRaises(ValueError):
                    reference.read_tokens(path)

    def test_target_ce(self):
        data = [None, {11: SimpleNamespace(logprob=-2.), 99: SimpleNamespace(logprob=-1.)},
                {13: SimpleNamespace(logprob=-3.)}]
        self.assertEqual(reference.loss_rows([7, 11, 13], data), [(0, 11, 2.), (1, 13, 3.)])
        for bad in (None, data[:2], [None, None, data[2]], [None, {99: data[1][11]}, data[2]]):
            with self.assertRaises(ValueError):
                reference.loss_rows([7, 11, 13], bad)
        for invalid in (math.nan, math.inf, .1):
            data[1][11].logprob = invalid
            with self.assertRaises(ValueError):
                reference.loss_rows([7, 11, 13], data)

    def test_guard_kills_only_created_container(self):
        cid, image = "1"*64, "sha256:"+"a"*64
        recorded = []

        def inspect(command, **kwargs):
            recorded.append(command)
            if command[0] == "nvidia-smi": return ""
            if command[:3] == ["docker", "image", "inspect"]: return image+"\n"
            if command[:2] == ["docker", "create"]: return cid+"\n"
            if command[:2] == ["docker", "inspect"]:
                self.assertEqual(command[-1], cid)
                return "false\n" if "Running" in command[3] else "137\n"
            self.fail(f"unexpected command {command}")

        class Client:
            status = None
            def poll(self): return self.status
            def wait(self, timeout): self.status = 0; return 0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root/"model"; model.mkdir()
            (model/"model.safetensors.index.json").write_text("{}")
            tokens = root/"tokens"; tokens.write_text("tokens=2\ntoken_ids=7,11\n")
            args = SimpleNamespace(model=str(model), tokens=str(tokens), output=str(root/"output"),
                                   n_ctx=256, prefix=2, image="test", batch=128)
            status = io.StringIO()
            with contextlib.redirect_stdout(status), \
                 patch.object(reference, "available_kib", side_effect=[101*1024**2, 23*1024**2]), \
                 patch.object(reference.subprocess, "check_output", side_effect=inspect), \
                 patch.object(reference.subprocess, "Popen", return_value=Client()) as start, \
                 patch.object(reference.subprocess, "run") as mutation:
                self.assertEqual(reference.host(args), 137)
                self.assertEqual(start.call_args.args[0], ["docker", "start", "--attach", cid])
                self.assertEqual([call.args[0] for call in mutation.call_args_list],
                                 [["docker", "kill", cid], ["docker", "rm", cid]])
            self.assertIn("reference_memory_guard=stop", status.getvalue())
            create = next(command for command in recorded if command[:2] == ["docker", "create"])
            self.assertIn("--read-only", create)
            self.assertEqual(create[create.index("--network")+1], "none")
            mounts = [create[i+1] for i, value in enumerate(create) if value == "--mount"]
            self.assertEqual(sum(value.endswith(",readonly") for value in mounts), 3)


if __name__ == "__main__":
    unittest.main()
