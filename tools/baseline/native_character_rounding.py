"""Trial-only deferred character values; optional resident Metal scaling preserves original separate binary64 multiply/add semantics."""

import struct
import numpy as np
from tools.baseline.native_ctc import ResidentCharacterSegments


class CharacterRoundBatch:
    def __init__(self, segments, context, records):
        self.segments, self.context, self.records = segments, context, records
        self.values, self.scale_key = None, None

    def materialize(self, ratio, offset):
        if type(ratio) is not float or type(offset) not in (int, float): raise ValueError("native_character_scale_type_not_supported")
        if isinstance(self.segments, ResidentCharacterSegments) and type(offset) is int and abs(offset) > 1 << 53: raise ValueError("native_resident_scale_type_invalid")
        key = struct.pack("dd", ratio, float(offset))
        if self.values is not None:
            if key != self.scale_key: raise ValueError("native_character_scale_changed")
            return
        resident = isinstance(self.segments, ResidentCharacterSegments)
        if resident:
            rounded, record = self.context.scale_round_segments(self.segments, ratio, offset)
        else:
            values = np.asarray([(start * ratio + offset, end * ratio + offset, score) for start, end, score in self.segments], dtype=np.float64)
            rounded, record = self.context.round3(values.reshape(-1))
        self.values, self.scale_key = rounded.reshape(-1, 3), key
        record.update(characters=len(self.segments), scaling="metal_binary64_separate_multiply_add" if resident else "original_cpu_python_float_operations", rounding="metal_exact_fixed_decimal_three")
        self.records.append(record)

    def result(self, index, field, digits):
        if type(digits) is not int or digits != 3: raise ValueError("native_character_round_digits_not_supported")
        if self.values is None: raise ValueError("native_character_round_scale_not_ready")
        return float(self.values[index, field])


class RoundedCharacterValue:
    def __init__(self, batch, index, field): self.batch, self.index, self.field = batch, index, field

    def __round__(self, ndigits=None): return self.batch.result(self.index, self.field, ndigits)


class ScaledCharacterCoordinate:
    def __init__(self, batch, index, field, ratio): self.batch, self.index, self.field, self.ratio = batch, index, field, ratio

    def __add__(self, offset):
        self.batch.materialize(self.ratio, offset)
        return RoundedCharacterValue(self.batch, self.index, self.field)


class CharacterCoordinate:
    def __init__(self, batch, index, field): self.batch, self.index, self.field = batch, index, field

    def __mul__(self, ratio): return ScaledCharacterCoordinate(self.batch, self.index, self.field, ratio)


def rounded_character_segments(alignment, segments, transcript, context, records):
    if not segments or len(segments) != len(transcript) or not isinstance(records, list): raise ValueError("native_character_round_metadata_invalid")
    batch = CharacterRoundBatch(segments, context, records)
    return [alignment.Segment(transcript[i], CharacterCoordinate(batch, i, 0), CharacterCoordinate(batch, i, 1), RoundedCharacterValue(batch, i, 2)) for i in range(len(segments))]
