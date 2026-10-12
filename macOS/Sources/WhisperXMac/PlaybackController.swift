import AVFoundation
import Observation
import SwiftUI

@MainActor @Observable
final class PlaybackController {
    var projectID: UUID?
    var title = L10n.tr("Nessun progetto selezionato")
    var currentTime = 0.0
    var duration = 0.0
    var isPlaying = false
    var isSeeking = false
    var volume: Double = 1 { didSet { player.volume = Float(volume) } }
    let player = AVPlayer()
    private var timer: Timer?

    init() {
        timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.updateTime() }
        }
    }

    func select(_ item: MediaItem?) {
        guard let item else {
            player.pause()
            player.replaceCurrentItem(with: nil)
            projectID = nil
            title = L10n.tr("Nessun progetto selezionato")
            currentTime = 0
            duration = 0
            isPlaying = false
            return
        }
        title = item.title
        duration = item.durationSeconds
        guard projectID != item.id else { return }
        player.pause()
        player.replaceCurrentItem(with: AVPlayerItem(url: item.url))
        projectID = item.id
        currentTime = 0
        isPlaying = false
    }

    func toggle() {
        guard projectID != nil else { return }
        if isPlaying { player.pause(); isPlaying = false }
        else {
            if duration > 0 && currentTime >= duration - 0.1 { seek(0) }
            player.play()
            isPlaying = true
        }
    }

    func pause() { player.pause(); isPlaying = false }

    func seek(_ seconds: Double) {
        currentTime = max(0, min(seconds, duration > 0 ? duration : seconds))
        player.seek(to: CMTime(seconds: currentTime, preferredTimescale: 600))
    }

    private func updateTime() {
        guard projectID != nil else { return }
        let actualDuration = player.currentItem?.duration.seconds ?? 0
        if actualDuration.isFinite && actualDuration > 0 { duration = actualDuration }
        if !isSeeking {
            let seconds = player.currentTime().seconds
            if seconds.isFinite { currentTime = seconds }
        }
        isPlaying = player.rate > 0
    }
}

func timeLabel(_ seconds: Double) -> String {
    guard seconds.isFinite else { return "0:00" }
    let value = max(0, Int(seconds))
    return value >= 3600 ? String(format: "%d:%02d:%02d", value / 3600, value / 60 % 60, value % 60) : String(format: "%d:%02d", value / 60, value % 60)
}

struct PlaybackBar: View {
    @Bindable var playback: PlaybackController

    var body: some View {
        HStack(spacing: 12) {
            HStack(spacing: 12) {
                Button { playback.seek(playback.currentTime - 15) } label: { Image(systemName: "gobackward.15") }.help(L10n.tr("Indietro di 15 secondi"))
                Button { playback.toggle() } label: { Image(systemName: playback.isPlaying ? "pause.fill" : "play.fill").font(.title3) }.help(playback.isPlaying ? L10n.tr("Pausa") : L10n.tr("Riproduci"))
                Button { playback.seek(playback.currentTime + 15) } label: { Image(systemName: "goforward.15") }.help(L10n.tr("Avanti di 15 secondi"))
            }.buttonStyle(.plain).disabled(playback.projectID == nil)
            VStack(spacing: 2) {
                Text(playback.title).font(.caption.weight(.medium)).lineLimit(1)
                HStack(spacing: 6) {
                    Text(timeLabel(playback.currentTime)).frame(width: 43, alignment: .trailing)
                    Slider(value: Binding(get: { min(playback.currentTime, max(playback.duration, 1)) }, set: { playback.seek($0) }), in: 0...max(playback.duration, 1), onEditingChanged: { playback.isSeeking = $0 })
                        .controlSize(.small).disabled(playback.projectID == nil).accessibilityLabel(L10n.tr("Posizione di riproduzione"))
                    Text(timeLabel(playback.duration)).frame(width: 43, alignment: .leading)
                }.font(.caption).monospacedDigit().foregroundStyle(.secondary)
            }.frame(width: 220)
            HStack(spacing: 6) {
                Image(systemName: "speaker.wave.2").foregroundStyle(.secondary)
                Slider(value: $playback.volume, in: 0...1).controlSize(.small).accessibilityLabel(L10n.tr("Volume"))
            }.frame(width: 80)
        }.padding(.vertical, 8).frame(minHeight: 48)
    }
}
