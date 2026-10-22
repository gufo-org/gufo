"""The staging-pressure workload must refuse missing reclamation evidence."""

import unittest

from publication_wait import reclaim_evidence


class PublicationReclaimTests(unittest.TestCase):
    def test_retains_actual_byte_pressure_reclamation(self):
        segment = (
            "cache schema=component-cache-v1 event=retention action=2 reason=1 "
            "tokens=84 rank=3 unique_bytes_freed=133100000 ledger_bytes=200000000\n"
        )
        self.assertEqual(reclaim_evidence(segment)[0]["tokens"], "84")

    def test_rejects_publication_without_freed_retained_ram(self):
        for segment in (
            "",
            "cache event=published tokens=55286 payload_bytes=588936280\n",
            "cache event=retention action=0 reason=0 unique_bytes_freed=133100000\n",
            "cache event=retention action=2 reason=3 unique_bytes_freed=133100000\n",
            "cache event=retention action=2 reason=1 unique_bytes_freed=0\n",
        ):
            with self.subTest(segment=segment), self.assertRaises(AssertionError):
                reclaim_evidence(segment)


if __name__ == "__main__":
    unittest.main()
