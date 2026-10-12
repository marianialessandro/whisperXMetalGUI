from pathlib import Path
import socket
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.offline_worker import deny_python_network, OfflineNetworkError


class OfflineWorkerTests(unittest.TestCase):
    def test_python_network_is_denied_before_socket_or_dns_and_restored(self):
        original = socket.create_connection
        with deny_python_network():
            for call in (lambda: socket.create_connection(("example.invalid", 443)), lambda: socket.getaddrinfo("example.invalid", 443)):
                with self.assertRaises(OfflineNetworkError) as error:
                    call()
                self.assertEqual(str(error.exception), "offline_network_denied")
            with socket.socket() as connection:
                with self.assertRaises(OfflineNetworkError):
                    connection.connect(("127.0.0.1", 1))
        self.assertIs(socket.create_connection, original)

    def test_hook_cleanup_is_guaranteed_on_failure(self):
        original = socket.socket.connect
        with self.assertRaises(RuntimeError):
            with deny_python_network():
                raise RuntimeError("controlled fixture failure")
        self.assertIs(socket.socket.connect, original)


if __name__ == "__main__":
    unittest.main()
