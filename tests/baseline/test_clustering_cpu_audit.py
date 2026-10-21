from concurrent.futures import ThreadPoolExecutor
import json
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pyannote.audio.pipelines.clustering as clustering

from tools.baseline.clustering_cpu_audit import clustering_cpu_audit, targets


class ClusteringCPUAuditTests(unittest.TestCase):
    def work(self):
        features = np.array([[0., 0.], [0., .1], [2., 2.], [2., 2.1]])
        tree = clustering.linkage(features, method="centroid", metric="euclidean")
        groups = clustering.fcluster(tree, .5, criterion="distance")
        predicted = clustering.KMeans(n_clusters=2, n_init=3, random_state=42).fit_predict(features)
        assigned = clustering.linear_sum_assignment(np.array([[1., 2., 3.], [3., 2., 1.]]), maximize=True)
        return tree, groups, predicted, *assigned

    def test_optional_pairwise_audit_distinguishes_linkage_from_pdist(self):
        features = np.array([[0., 0.], [0., .1], [2., 2.], [2., 2.1]])
        from scipy.spatial.distance import pdist
        condensed = pdist(features)
        with clustering_cpu_audit(records := [], include_pairwise=True):
            expected = clustering.linkage(features, method="centroid", metric="euclidean")
        by_name = {row["operation"]: row for row in records}
        self.assertEqual(len(by_name), 5)
        self.assertEqual(by_name["ahc_pairwise_distances"]["calls_started"], 1)
        self.assertEqual(by_name["ahc_linkage"]["calls_started"], 1)
        with clustering_cpu_audit(records := [], include_pairwise=True):
            actual = clustering.linkage(condensed, method="centroid", metric="euclidean")
        by_name = {row["operation"]: row for row in records}
        self.assertEqual(by_name["ahc_pairwise_distances"]["calls_started"], 0)
        self.assertEqual(by_name["ahc_linkage"]["calls_started"], 1)
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        with self.assertRaisesRegex(ValueError, "pairwise_flag_invalid"):
            with clustering_cpu_audit([], include_pairwise=1):
                pass
        self.assertIsNone(sys.getprofile())

    def test_original_results_and_function_identities_preserved(self):
        expected = self.work()
        originals = (clustering.linkage, clustering.fcluster, clustering.KMeans.fit_predict, clustering.linear_sum_assignment)
        with clustering_cpu_audit(records := []):
            actual = self.work()
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(originals, (clustering.linkage, clustering.fcluster, clustering.KMeans.fit_predict, clustering.linear_sum_assignment))
        self.assertIsNone(sys.getprofile())
        self.assertEqual(len(records), 4)
        for row in records:
            self.assertEqual(row["calls_started"], 1)
            self.assertEqual(row["calls_returned_or_unwound"], 1)
            self.assertEqual(row["c_exceptions"], 0)
            self.assertEqual(row["incomplete_calls"], 0)
            self.assertTrue(row["observation_complete"])
            self.assertGreater(row["total_wall_ns"], 0)
            self.assertFalse(row["exclusive_cpu_time"])
            self.assertFalse(row["gpu_dispatch_verified"])
        self.assertNotIn("features", json.dumps(records))

    def test_background_thread_and_empty_scope_do_not_receive_credit(self):
        with ThreadPoolExecutor(max_workers=1) as executor, clustering_cpu_audit(records := []):
            executor.submit(self.work).result()
        self.assertTrue(all(row["calls_started"] == 0 for row in records))
        with clustering_cpu_audit(empty := []):
            pass
        self.assertTrue(all(row["total_wall_ns"] == 0 for row in empty))

    def test_original_c_error_and_python_unwind_counted_without_messages(self):
        with clustering_cpu_audit(records := []):
            with self.assertRaises(ValueError):
                clustering.linear_sum_assignment(np.full((2, 3), np.nan))
            with self.assertRaises(ValueError):
                clustering.fcluster(np.ones((1, 4)), .5, criterion="secret-invalid-criterion")
        solver = next(row for row in records if row["operation"] == "linear_sum_assignment")
        self.assertEqual(solver["c_exceptions"], 1)
        self.assertEqual(solver["calls_returned_or_unwound"], 1)
        python = next(row for row in records if row["operation"] == "ahc_fcluster")
        self.assertEqual(python["calls_returned_or_unwound"], 1)
        self.assertFalse(python["python_exception_status_available"])
        self.assertNotIn("secret", json.dumps(records))

    def test_existing_profiler_nesting_and_failure_restore(self):
        def existing(frame, event, arg):
            pass
        sys.setprofile(existing)
        try:
            with self.assertRaisesRegex(ValueError, "profiler_already_active"):
                with clustering_cpu_audit([]):
                    pass
            self.assertIs(sys.getprofile(), existing)
        finally:
            sys.setprofile(None)
        with self.assertRaisesRegex(RuntimeError, "private-exception"):
            with clustering_cpu_audit(records := []):
                with self.assertRaisesRegex(ValueError, "profiler_already_active"):
                    with clustering_cpu_audit([]):
                        pass
                self.work()
                raise RuntimeError("private-exception")
        self.assertIsNone(sys.getprofile())
        self.assertTrue(all(row["incomplete_calls"] == 0 for row in records))
        self.assertNotIn("private-exception", json.dumps(records))

    def test_runtime_source_and_reference_drift_fail_before_installation(self):
        for records in (None, (), [{}]):
            with self.assertRaisesRegex(ValueError, "empty_records_required"):
                with clustering_cpu_audit(records):
                    pass
        with patch("tools.baseline.clustering_cpu_audit.importlib.metadata.version", return_value="changed"), self.assertRaisesRegex(ValueError, "runtime_not_supported"):
            targets()
        with patch("tools.baseline.clustering_cpu_audit.inspect.getsource", return_value="changed"), self.assertRaisesRegex(ValueError, "source_not_supported"):
            targets()
        with patch.object(clustering, "linkage", object()), self.assertRaisesRegex(ValueError, "reference_changed"):
            targets()
        self.assertIsNone(sys.getprofile())

    def test_displaced_profiler_cannot_claim_complete_observation(self):
        with clustering_cpu_audit(records := []):
            self.work()
            sys.setprofile(None)
            self.work()
        self.assertIsNone(sys.getprofile())
        self.assertTrue(all(row["calls_started"] == 1 for row in records))
        self.assertTrue(all(not row["observation_complete"] for row in records))


if __name__ == "__main__":
    unittest.main()
