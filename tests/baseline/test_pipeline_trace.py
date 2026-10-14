from pathlib import Path
import sys
import unittest
import copy
import subprocess
import signal
import tempfile
import wave
import io
import json
import ctypes
import os
import uuid
from unittest.mock import patch
from unittest.mock import Mock
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.tracing import audio_clip, pipeline_entry, pipeline_review, pipeline_trace
from tools.tracing.start_notification import StartNotification


class PipelineTraceTests(unittest.TestCase):
    def test_real_native_notification_waits_for_post_and_cancels_descriptor(self):
        with StartNotification("org.whisperx.test." + uuid.uuid4().hex) as notification:
            descriptor = notification.descriptor
            with self.assertRaisesRegex(ValueError, "notification_timeout"): notification.wait(timeout=0.001)
            notification.library.notify_post.argtypes = [ctypes.c_char_p]
            notification.library.notify_post.restype = ctypes.c_uint32
            self.assertEqual(notification.library.notify_post(notification.name.encode()), 0)
            notification.wait(timeout=1)
        notification.close()
        with self.assertRaises(OSError): os.fstat(descriptor)

    def test_native_recorder_ready_required_and_early_exit_or_timeout_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recorder.log"
            tracer = Mock()
            tracer.poll.return_value = None
            path.write_text("Ctrl-C to stop the recording")
            pipeline_trace.wait_recorder_ready(tracer, path)
            tracer.poll.return_value = 2
            with self.assertRaisesRegex(ValueError, "exited_before_attach"): pipeline_trace.wait_recorder_ready(tracer, path)
            tracer.poll.return_value = None
            path.write_text("Starting recorder")
            ticks = iter([0, 1])
            with self.assertRaisesRegex(ValueError, "attach_timeout"): pipeline_trace.wait_recorder_ready(tracer, path, timeout=1, clock=lambda: next(ticks))

    def test_controlled_start_requires_parent_release_and_keeps_legacy_input(self):
        request = {"quality": "Rapida", "language": "it", "voices": False}
        arguments = ["--source-root", "/fixture", "--spans", "/fixture/spans.json", "--run-id", "fixture", "--ready-delay", "0"]
        for controlled in (False, True):
            content = json.dumps(request) + ("\ntrace_attached\ntrace_saved\n" if controlled else "")
            with patch.object(pipeline_entry.sys, "stdin", io.StringIO(content)), patch.object(pipeline_entry.os, "setpgrp"), patch.object(pipeline_entry, "print"), patch.object(pipeline_entry.adapter, "main", return_value=0) as main:
                self.assertEqual(pipeline_entry.main(arguments + (["--controlled-start"] if controlled else [])), 0)
                main.assert_called_once()
        with patch.object(pipeline_entry.sys, "stdin", io.StringIO(json.dumps(request) + "\ninvalid\n")), patch.object(pipeline_entry.os, "setpgrp"), patch.object(pipeline_entry, "print"), patch.object(pipeline_entry.adapter, "main") as main:
            with self.assertRaisesRegex(ValueError, "release_invalid"): pipeline_entry.main(arguments + ["--controlled-start"])
            main.assert_not_called()

    def test_diagnostic_audio_prefix_preserves_original_pcm_and_exclusive_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source.wav", root / "clip.wav"
            payload = b"\x01\x02" * 32000
            with wave.open(str(source), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes(payload)
            original = source.read_bytes()
            report = audio_clip.prefix(source, target, 1)
            self.assertEqual(source.read_bytes(), original)
            with wave.open(str(target), "rb") as reader: self.assertEqual(reader.readframes(reader.getnframes()), payload[:32000])
            self.assertEqual(report["selected_frames"], 16000)
            self.assertFalse(report["full_recording_trace"])
            self.assertFalse(report["quality_sample"])
            with self.assertRaises(FileExistsError): audio_clip.prefix(source, target, 1)
            for value in (0, 121, True, 1.5):
                with self.assertRaises(ValueError): audio_clip.prefix(source, root / "unused.wav", value)

    def test_recorder_auto_exit_is_allowed_to_save_without_second_interrupt(self):
        tracer = Mock()
        tracer.wait.return_value = 0
        self.assertEqual(pipeline_trace.finish_recorder(tracer), 0)
        tracer.send_signal.assert_not_called()

    def test_recorder_is_stopped_before_live_target_is_released(self):
        tracer = Mock()
        tracer.poll.return_value = None
        tracer.wait.return_value = 0
        self.assertEqual(pipeline_trace.finish_recorder(tracer, target_alive=True), 0)
        tracer.send_signal.assert_called_once_with(signal.SIGINT)

    def test_recorder_still_running_after_grace_receives_one_stop_signal(self):
        tracer = Mock()
        tracer.wait.side_effect = [subprocess.TimeoutExpired("recorder", 20), 0]
        tracer.poll.return_value = None
        self.assertEqual(pipeline_trace.finish_recorder(tracer), 0)
        tracer.send_signal.assert_called_once_with(signal.SIGINT)

    def test_recorder_exiting_at_grace_boundary_is_not_signalled_again(self):
        tracer = Mock()
        tracer.wait.side_effect = [subprocess.TimeoutExpired("recorder", 20), 0]
        tracer.poll.return_value = 0
        self.assertEqual(pipeline_trace.finish_recorder(tracer), 0)
        tracer.send_signal.assert_not_called()

    def fixture(self):
        toc = ET.fromstring('<trace-toc><run><info><target><process pid="401" type="attached"/></target></info></run></trace-toc>')
        columns = "start duration process channel-name state".split()
        schema = "<schema name='metal-gpu-intervals'>" + "".join("<col><mnemonic>" + column + "</mnemonic></col>" for column in columns) + "</schema>"
        foreign = '<row><start id="1">10</start><duration id="2">20</duration><process id="3"><pid id="4">602</pid></process><channel id="5">Compute</channel><state id="6">Active</state></row>'
        target = '<row><start ref="1"/><duration ref="2"/><process><pid>401</pid></process><channel ref="5"/><state ref="6"/></row>'
        gpu = ET.fromstring("<trace-query-result><node>" + schema + foreign + target + "</node></trace-query-result>")
        spans = {"run_id": "actual", "pid": 401, "timing_kind": "synchronized_completion", "spans": [{"stage": stage, "status": "synchronized"} for stage in ("decode", "asr_mlx", "alignment", "export_write", "diarization", "speaker_assignment")]}
        controller = {"status": "diagnostic_pipeline_completed", "pid": 401, "child_returncode": 0, "run_id": "actual", "voices": True}
        return toc, gpu, spans, controller

    def test_pipeline_process_activity_never_certifies_operators_or_other_apps(self):
        result = pipeline_review.review(*self.fixture())
        self.assertTrue(result["gpu_activity_observed"])
        self.assertEqual(result["target_compute_intervals"], [{"start_ns": 10, "duration_ns": 20}])
        self.assertEqual(result["foreign_process_rows_excluded"], 1)
        self.assertFalse(result["individual_operator_dispatch_verified"])
        self.assertFalse(result["phase_interval_attribution_verified"])
        self.assertTrue(all(item["status"] == "unverified" for item in result["operator_ledger"]))
        self.assertEqual(len(result["operator_ledger"]), 26)

    def test_wrong_target_phase_run_or_failed_child_is_rejected(self):
        for kind in ("pid", "phase", "run", "exit"):
            toc, gpu, spans, controller = self.fixture()
            if kind == "pid": controller["pid"] = 999
            if kind == "phase": spans["spans"] = spans["spans"][:-1]
            if kind == "run": controller["run_id"] = "other"
            if kind == "exit": controller["child_returncode"] = 2
            with self.assertRaises(ValueError): pipeline_review.review(toc, gpu, spans, controller)

    def test_foreign_only_activity_and_broken_references_fail_closed(self):
        toc, gpu, spans, controller = self.fixture()
        gpu.findall(".//row")[1][2].find("pid").text = "602"
        self.assertFalse(pipeline_review.review(toc, gpu, spans, controller)["gpu_activity_observed"])
        gpu.findall(".//row")[1][2].find("pid").text = "401"
        gpu.findall(".//row")[1][0].set("ref", "missing")
        with self.assertRaises(ValueError): pipeline_review.review(toc, gpu, spans, controller)


    def test_native_unattributed_sentinel_is_excluded_and_missing_process_pid_still_rejected(self):
        toc, gpu, spans, controller = self.fixture()
        row = copy.deepcopy(gpu.findall(".//row")[1])
        row[2] = ET.Element("sentinel")
        gpu.find("node").append(row)
        report = pipeline_review.review(toc, gpu, spans, controller)
        self.assertEqual(report["unattributed_process_rows_excluded"], 1)
        self.assertEqual(len(report["target_compute_intervals"]), 1)
        row[2] = ET.Element("process")
        with self.assertRaisesRegex(ValueError, "reference_invalid"): pipeline_review.review(toc, gpu, spans, controller)


if __name__ == "__main__":
    unittest.main()
