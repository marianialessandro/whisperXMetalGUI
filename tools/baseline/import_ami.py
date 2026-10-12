"""Import targeted AMI manual XML as unreviewed independent dataset references, never certified ground truth."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import corpus, manifest


def xml_member(archive, name):
    try:
        entry = archive.getinfo(name)
    except KeyError:
        raise corpus.CorpusError("ami_member_missing") from None
    if not 0 < entry.file_size <= 4 * 1024 * 1024 or entry.flag_bits & 1:
        raise corpus.CorpusError("ami_xml_budget_invalid")
    data = archive.read(entry)
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise corpus.CorpusError("ami_xml_entities_forbidden")
    try:
        return ET.fromstring(data), hashlib.sha256(data).hexdigest()
    except ET.ParseError:
        raise corpus.CorpusError("ami_xml_invalid") from None


def interval(element, first, last, duration):
    try:
        start, end = float(element.attrib[first]), float(element.attrib[last])
    except (KeyError, ValueError):
        raise corpus.CorpusError("ami_interval_invalid") from None
    if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= duration:
        raise corpus.CorpusError("ami_interval_invalid")
    return start, end


def import_manual(path, digest, meeting, audio_digest, duration):
    if not corpus.valid_digest(digest) or not corpus.valid_digest(audio_digest) or not re.fullmatch(r"[A-Z]{2}[0-9]{4}[a-z]?", meeting) or not math.isfinite(duration) or duration <= 0 or manifest.sha256(path) != digest:
        raise corpus.CorpusError("ami_identity_invalid")
    words, turns, sources, speakers, punctuation = [], [], [], {}, []
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) > 10000 or len({item.filename for item in archive.infolist()}) != len(archive.infolist()):
            raise corpus.CorpusError("ami_archive_invalid")
        metadata, checksum = xml_member(archive, "corpusResources/meetings.xml")
        sources.append({"member": "corpusResources/meetings.xml", "sha256": checksum})
        selected = [item for item in metadata.findall("meeting") if item.get("observation") == meeting]
        if len(selected) != 1:
            raise corpus.CorpusError("ami_meeting_missing")
        for item in selected[0].findall("speaker"):
            agent, identity = item.get("nxt_agent"), item.get("global_name")
            if not identity or not agent or agent in speakers:
                raise corpus.CorpusError("ami_speaker_identity_missing")
            speakers[agent] = "ami_person_" + hashlib.sha256(identity.encode()).hexdigest()[:16]
        if not speakers:
            raise corpus.CorpusError("ami_speaker_identity_missing")
        for agent, speaker in sorted(speakers.items()):
            for kind in ("words", "segments"):
                name = kind + "/" + meeting + "." + agent + "." + kind + ".xml"
                content, checksum = xml_member(archive, name)
                sources.append({"member": name, "sha256": checksum})
                if kind == "words":
                    for item in content.findall("w"):
                        text = "".join(item.itertext()).strip()
                        if not text: continue
                        if item.get("punc") == "true":
                            punctuation.append({"text": text, "speaker": speaker, "source_id": item.get("{http://nite.sourceforge.net/}id")})
                            continue
                        start, end = interval(item, "starttime", "endtime", duration)
                        words.append({"word": text, "start": start, "end": end, "speaker": speaker, "source_id": item.get("{http://nite.sourceforge.net/}id")})
                else:
                    for item in content.findall("segment"):
                        start, end = interval(item, "transcriber_start", "transcriber_end", duration)
                        turns.append({"speaker": speaker, "start": start, "end": end, "source_id": item.get("{http://nite.sourceforge.net/}id")})
    words.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))
    turns.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))
    return {"schema_version": 1, "recording_id": "ami_" + meeting, "source_group": "ami_" + meeting.rstrip("abcdefghijklmnopqrstuvwxyz"), "speaker_ids": sorted(speakers.values()), "audio_sha256": audio_digest, "duration": duration, "provenance": "independent_manual_dataset", "source": "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip", "archive_sha256": digest, "members": sources, "transcript": " ".join(item["word"] for item in words), "transcript_order": "word_start_time_ties_end_time_speaker_requires_overlap_policy_review", "words": words, "turns": turns, "turn_basis": "manual_transcriber_segment_boundaries_requires_speech_activity_adjudication", "punctuation": punctuation, "uem": [[0, duration]], "reviewed": False, "reviewers": [], "uncertainties": [{"start": 0, "end": duration, "reason": "local_word_turn_overlap_uem_adjudication_pending"}], "quality_certified": False, "full_baseline_gate_passed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--meeting", required=True)
    parser.add_argument("--audio-sha256", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New private reference JSON; imported references remain unreviewed and cannot pass the baseline gate.")
    args = parser.parse_args(argv)
    try:
        reference = import_manual(args.archive, args.archive_sha256, args.meeting, args.audio_sha256, args.duration)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output: json.dump(reference, output, indent=2)
        print(json.dumps({"status": "manual_references_imported_unreviewed", "recording_id": reference["recording_id"], "words": len(reference["words"]), "turns": len(reference["turns"]), "speakers": len(reference["speaker_ids"]), "full_baseline_gate_passed": False}))
        return 0
    except corpus.CorpusError as error:
        print(json.dumps({"status": "prerequisite_missing", "code": str(error)}))
        return 2
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        print(json.dumps({"status": "prerequisite_missing", "code": "ami_import_failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
