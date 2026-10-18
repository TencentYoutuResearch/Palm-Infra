"""Check process-local failure semantics without GPU or real memory pressure."""
import subprocess
import sys
import unittest
import os

BINARY = sys.argv.pop(1)


class WatchdogExitTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux terminal restoration")
    def test_emergency_restores_line_editing_terminal(self):
        import pty
        import termios
        master, slave = pty.openpty()
        try:
            original = termios.tcgetattr(slave)
            result = subprocess.run([BINARY, "--drop-raw"], stdin=slave,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, timeout=10)
            self.assertEqual(result.returncode, 137)
            self.assertEqual(termios.tcgetattr(slave), original)
        finally:
            os.close(master)
            os.close(slave)

    def test_stop_only_model_child(self):
        for argument, readable in (("--drop", "readable=1"),
                                   ("--unreadable", "readable=0")):
            with self.subTest(argument=argument):
                result = subprocess.run([BINARY, argument], capture_output=True,
                                        text=True, timeout=10)
                self.assertEqual(result.returncode, 137)
                self.assertIn("watchdog child ready", result.stdout)
                self.assertIn("stopping this model process", result.stderr)
                self.assertIn(readable, result.stderr)
                # This test runner survived both failures; no group signal.


if __name__ == "__main__":
    unittest.main()
