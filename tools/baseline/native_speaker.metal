// The existing CTC shader precedes this source and supplies exact bounded binary64 arithmetic.
struct SpeakerDimensions { uint turns; uint queries; uint speakers; uint nearest; };

kernel void speaker_stable_order(device const ulong2* turns [[buffer(0)]], constant SpeakerDimensions& d [[buffer(1)]], device uint* order [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index >= d.turns) return;
    uint rank = 0;
    ulong start = turns[index].x;
    for (uint j = 0; j < d.turns; ++j) rank += turns[j].x < start || (turns[j].x == start && j < index);
    order[rank] = index;
}

kernel void speaker_assign(device const ulong2* turns [[buffer(0)]], constant SpeakerDimensions& d [[buffer(1)]], device const uint* order [[buffer(2)]], device const uint* labels [[buffer(3)]], device const ulong2* queries [[buffer(4)]], device ulong* sums [[buffer(5)]], device int* output [[buffer(6)]], uint query [[thread_position_in_grid]]) {
    if (query >= d.queries) return;
    device ulong* totals = sums + query * d.speakers;
    for (uint s = 0; s < d.speakers; ++s) totals[s] = 0;
    ulong2 q = queries[query];
    for (uint j = 0; j < d.turns; ++j) {
        uint index = order[j];
        ulong2 t = turns[index];
        if (min(q.y, t.y) > max(q.x, t.x)) {
            ulong intersection = add_signed_double(min(q.y, t.y), max(q.x, t.x) ^ (1ul << 63));
            totals[labels[index]] = add_positive_double(totals[labels[index]], intersection);
        }
    }
    int winner = -1;
    ulong best = 0;
    for (uint j = 0; j < d.turns; ++j) {
        uint index = order[j];
        ulong2 t = turns[index];
        if (min(q.y, t.y) > max(q.x, t.x) && (winner < 0 || totals[labels[index]] > best)) {
            winner = int(labels[index]);
            best = totals[labels[index]];
        }
    }
    if (winner < 0 && d.nearest) {
        ulong midpoint = divide_positive_double(add_positive_double(q.x, q.y), 2);
        ulong distance = ~0ul;
        for (uint j = 0; j < d.turns; ++j) {
            uint index = order[j];
            ulong2 t = turns[index];
            ulong middle = divide_positive_double(add_positive_double(t.x, t.y), 2);
            ulong delta = add_signed_double(middle, midpoint ^ (1ul << 63)) & 0x7ffffffffffffffful;
            if (delta < distance) { distance = delta; winner = int(labels[index]); }
        }
    }
    output[query] = winner;
}
