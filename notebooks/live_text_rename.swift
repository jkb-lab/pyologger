#!/usr/bin/swift
//
// Live Text (Apple Vision) video timestamp → rename (full version)
// ---------------------------------------------------------------
// Usage (env + one positional arg):
//
//   export DIR_METADATA="/path/to/00_metadata"
//   export DIR_PROCESSED="/path/to/02_processed-video"
//   export DEPLOY_DATE="YYYY-MM-DD"
//   export ANIMAL_ID="mile-001"
//   export CAMERA_ID="PD-01"
//   # Optional:
//   export FPS="2"            # frames/sec sampled in first DURATION
//   export DURATION="20"      # seconds from start to scan
//   export ROI="x,y,w,h"      # pixels; if unset, auto top-left box
//
//   ./live_text_rename.swift "/path/to/01_raw-video/PD-01"
//
// Outputs:
//   • Renamed copies into DIR_PROCESSED
//   • Per-video CSVs + batch_summary.csv into DIR_METADATA
//

import Foundation
import AVFoundation
import Vision
import CoreGraphics

// MARK: - ENV
let env = ProcessInfo.processInfo.environment
func need(_ k: String) -> String {
    guard let v = env[k], !v.isEmpty else { fputs("Missing env \(k)\n", stderr); exit(1) }
    return v
}
let DIR_METADATA   = need("DIR_METADATA")
let DIR_PROCESSED  = need("DIR_PROCESSED")
let DEPLOY_DATE    = need("DEPLOY_DATE")
let ANIMAL_ID      = need("ANIMAL_ID")
let CAMERA_ID      = need("CAMERA_ID")
let FPS            = Double(env["FPS"] ?? "2") ?? 2.0
let DURATION       = Double(env["DURATION"] ?? "20") ?? 20.0
let ROI_STR        = env["ROI"] // optional

// MARK: - ARGS
guard CommandLine.arguments.count == 2 else {
    fputs("Usage: \(CommandLine.arguments[0]) <INPUT_DIR>\n", stderr)
    exit(1)
}
let INPUT_DIR = CommandLine.arguments[1]

// MARK: - FS helpers
let fm = FileManager.default
func ensureDir(_ path: String) {
    if !fm.fileExists(atPath: path) {
        try? fm.createDirectory(atPath: path, withIntermediateDirectories: true)
    }
}
ensureDir(DIR_METADATA); ensureDir(DIR_PROCESSED)

// MARK: - which + ffmpeg
func which(_ name: String) -> String? {
    let candidates = [
        "/opt/homebrew/bin/\(name)",
        "/usr/local/bin/\(name)",
        "/usr/bin/\(name)"
    ]
    for p in candidates where fm.isExecutableFile(atPath: p) { return p }
    let task = Process()
    task.executableURL = URL(fileURLWithPath: "/usr/bin/which")
    task.arguments = [name]
    let pipe = Pipe(); task.standardOutput = pipe
    try? task.run(); task.waitUntilExit()
    guard task.terminationStatus == 0 else { return nil }
    let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
        .trimmingCharacters(in: .whitespacesAndNewlines)
    return (out?.isEmpty == false) ? out : nil
}
let ffmpegPath = which("ffmpeg")

// MARK: - Conversion policy (AVI → MP4 if MP4 missing)
func listVideosWithConversion(_ dir: String) -> [URL] {
    let u = URL(fileURLWithPath: dir)
    guard let items = try? fm.contentsOfDirectory(at: u, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else { return [] }

    var valid: [URL] = []
    let exts = Set(["mp4","mov","m4v","hevc","avi","mkv","mpg","mpeg","qt"])

    for url in items {
        let ext = url.pathExtension.lowercased()
        guard exts.contains(ext) else { continue }

        if ext == "avi" {
            let mp4Candidate = url.deletingPathExtension().appendingPathExtension("mp4")
            if fm.fileExists(atPath: mp4Candidate.path) {
                // mp4 already exists for this stem → skip AVI
                print("↪️ Skipping \(url.lastPathComponent) (MP4 already exists)")
                continue
            } else if let ff = ffmpegPath {
                // Convert AVI → MP4 (same folder, same stem)
                print("🎬 Converting \(url.lastPathComponent) → \(mp4Candidate.lastPathComponent)")
                let p = Process()
                p.executableURL = URL(fileURLWithPath: ff)
                p.arguments = [
                    "-hide_banner","-loglevel","error","-y",
                    "-i", url.path,
                    "-c:v","libx264","-crf","20","-preset","fast",
                    "-c:a","aac","-b:a","160k",
                    "-movflags","+faststart",
                    mp4Candidate.path
                ]
                let pipe = Pipe(); p.standardError = pipe
                do { try p.run(); p.waitUntilExit() } catch {
                    print("❌ ffmpeg failed for \(url.lastPathComponent): \(error)")
                }
                if fm.fileExists(atPath: mp4Candidate.path) {
                    print("✅ Created \(mp4Candidate.lastPathComponent)")
                    valid.append(mp4Candidate)
                } else {
                    print("⚠️ Conversion failed for \(url.lastPathComponent)")
                }
            } else {
                print("⚠️ ffmpeg not found; cannot convert \(url.lastPathComponent)")
            }
        } else {
            valid.append(url)
        }
    }

    return valid.sorted { $0.lastPathComponent < $1.lastPathComponent }
}

// MARK: - Remux repair (for weird/broken moov)
func remuxIfNeeded(_ url: URL) -> URL {
    guard let ff = ffmpegPath else { return url }
    let tmpDir = URL(fileURLWithPath: NSTemporaryDirectory())
    let out = tmpDir.appendingPathComponent(url.deletingPathExtension().lastPathComponent + "_fixed.mp4")
    if fm.fileExists(atPath: out.path) { return out }
    let p = Process()
    p.executableURL = URL(fileURLWithPath: ff)
    p.arguments = ["-hide_banner","-loglevel","error","-y","-i", url.path,
                   "-c:v","copy","-c:a","aac","-movflags","+faststart", out.path]
    let pipe = Pipe(); p.standardError = pipe
    do { try p.run(); p.waitUntilExit() } catch { return url }
    return fm.fileExists(atPath: out.path) ? out : url
}

// MARK: - Modern duration loader
func loadDurationSeconds(asset: AVURLAsset) -> Double? {
    let sem = DispatchSemaphore(value: 0)
    var statusOut: AVKeyValueStatus = .unknown
    var err: NSError?
    asset.loadValuesAsynchronously(forKeys: ["duration"]) {
        statusOut = asset.statusOfValue(forKey: "duration", error: &err)
        sem.signal()
    }
    sem.wait()
    guard statusOut == .loaded else { return nil }
    let s = CMTimeGetSeconds(asset.duration)
    return (s.isFinite && s > 0) ? s : nil
}

// MARK: - Vision OCR + parsing
let dateFormats = [
    "yyyy/MM/dd HH:mm:ss","yyyy-MM-dd HH:mm:ss","yyyy.MM.dd HH:mm:ss",
    "yyyy/MM/dd HH:mm","yyyy-MM-dd HH:mm","yyyy.MM.dd HH:mm"
]
let parsers: [DateFormatter] = dateFormats.map { f in
    let df = DateFormatter()
    df.locale = .init(identifier: "en_US_POSIX")
    df.timeZone = .current
    df.dateFormat = f
    return df
}
let fullRegex = try! NSRegularExpression(pattern: #"(20\d{2}[-/.]\d{2}[-/.]\d{2}\s+\d{2}:\d{2}:\d{2})"#)

func extractDate(_ text: String) -> (Date, String)? {
    let ns = text as NSString
    if let m = fullRegex.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)) {
        let raw = ns.substring(with: m.range(at: 1))
        let normalized = raw.replacingOccurrences(of: ".", with: "/").replacingOccurrences(of: "-", with: "/")
        for df in parsers { if let d = df.date(from: normalized) { return (d, raw) } }
    }
    for df in parsers { if let d = df.date(from: text) { return (d, text) } }
    return nil
}

func recognizeText(_ cg: CGImage) -> String {
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = true
    req.revision = VNRecognizeTextRequestRevision3
    req.recognitionLanguages = ["en-US"]
    let h = VNImageRequestHandler(cgImage: cg, options: [:])
    do {
        try h.perform([req])
        let observations = req.results as? [VNRecognizedTextObservation] ?? []
        let lines = observations.compactMap { $0.topCandidates(1).first?.string }
        return lines.joined(separator: " ")
    } catch { return "" }
}

func crop(_ cg: CGImage, to r: CGRect) -> CGImage {
    let w = cg.width, h = cg.height
    let x = max(0, min(Int(r.origin.x), w-1))
    let y = max(0, min(Int(r.origin.y), h-1))
    let cw = max(1, min(Int(r.size.width),  w - x))
    let ch = max(1, min(Int(r.size.height), h - y))
    let rect = CGRect(x: x, y: y, width: cw, height: ch)
    return cg.cropping(to: rect) ?? cg
}

func autoROI(width: Int, height: Int) -> CGRect {
    let rw = max(240, Int(Double(width) * 0.35))
    let rh = max(90,  Int(Double(height) * 0.15))
    return CGRect(x: 0, y: 0, width: rw, height: rh)
}

// MARK: - CSV helpers
struct Row { let frame: String; let t: Double; let raw: String; let fixed: String }
func writeCSV(_ rows: [Row], to url: URL) {
    var s = "frame,approx_video_time_s,ocr_raw,ocr_fixed\n"
    for r in rows {
        let raw = r.raw.replacingOccurrences(of: "\"", with: "\"\"")
        let fix = r.fixed.replacingOccurrences(of: "\"", with: "\"\"")
        s += "\(r.frame),\(String(format: "%.3f", r.t)),\"\(raw)\",\"\(fix)\"\n"
    }
    try? s.data(using: .utf8)?.write(to: url)
}

// MARK: - Batch state
struct SummaryItem { let video: String; let renamed: Bool; let reason: String?; let newName: String?; let start: Date?; let end: Date? }
var summary: [SummaryItem] = []

// MARK: - Per-video processing
let dFmt = DateFormatter(); dFmt.locale = .init(identifier: "en_US_POSIX"); dFmt.timeZone = .current; dFmt.dateFormat = "yyyy-MM-dd"
let tFmt = DateFormatter(); tFmt.locale = .init(identifier: "en_US_POSIX"); tFmt.timeZone = .current; tFmt.dateFormat = "HH-mm-ss"

func processOne(_ origURL: URL) {
    print("\n== \(origURL.lastPathComponent) ==")

    // Use the file as-is; if we later detect unreadable duration, try a fast remux
    var workURL = origURL

    // Asset + duration
    var asset = AVURLAsset(url: workURL)
    var totalS = loadDurationSeconds(asset: asset)

    // If duration unreadable, try a quick remux/repair to temp and retry
    if totalS == nil, let _ = ffmpegPath {
        workURL = remuxIfNeeded(origURL)
        asset = AVURLAsset(url: workURL)
        totalS = loadDurationSeconds(asset: asset)
    }

    guard let durationS = totalS else {
        print("❌ No duration (even after repair if attempted)")
        summary.append(.init(video: origURL.lastPathComponent, renamed: false, reason: "no_duration", newName: nil, start: nil, end: nil))
        return
    }

    // First frame → size (sync CGImage; deprecation warning is cosmetic)
    let gen = AVAssetImageGenerator(asset: asset)
    gen.appliesPreferredTrackTransform = true
    var size = CGSize.zero
    if let cg0 = try? gen.copyCGImage(at: CMTime(seconds: 0, preferredTimescale: 600), actualTime: nil) {
        size = CGSize(width: cg0.width, height: cg0.height)
    }

    // ROI
    let roi: CGRect = {
        if let s = ROI_STR {
            let parts = s.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
            if parts.count == 4 { return CGRect(x: parts[0], y: parts[1], width: parts[2], height: parts[3]) }
        }
        return autoROI(width: Int(size.width), height: Int(size.height))
    }()

    // Build times in first N seconds
    var times: [NSValue] = []
    let scan = min(DURATION, durationS)
    let step = CMTime(seconds: 1.0 / FPS, preferredTimescale: 600)
    var t = CMTime(seconds: 0, preferredTimescale: 600)
    while CMTimeGetSeconds(t) <= scan { times.append(NSValue(time: t)); t = t + step }

    var rows: [Row] = []
    var bestDate: Date?
    var bestAt: Double = 0

    for (i, nv) in times.enumerated() {
        let ts = nv.timeValue
        guard let cg = try? gen.copyCGImage(at: ts, actualTime: nil) else { continue }
        let txt = recognizeText(crop(cg, to: roi))
        if let (d, _) = extractDate(txt) {
            if bestDate == nil || d < bestDate! { bestDate = d; bestAt = CMTimeGetSeconds(ts) }
        }
        rows.append(Row(
            frame: String(format: "frame_%06d.png", i+1),
            t: CMTimeGetSeconds(ts),
            raw: txt,
            fixed: bestDate.map { ISO8601DateFormatter.string(from: $0, timeZone: .current, formatOptions: [.withFullDate,.withTime,.withColonSeparatorInTime]) } ?? ""
        ))
    }

    // Per-video OCR CSV
    let csvURL = URL(fileURLWithPath: DIR_METADATA)
        .appendingPathComponent("\(origURL.deletingPathExtension().lastPathComponent)_ocr_first\(Int(DURATION))s.csv")
    writeCSV(rows, to: csvURL)
    print("🧾 CSV → \(csvURL.path)")

    guard let anchor = bestDate else {
        print("⚠️ No full datetime; skip rename")
        summary.append(.init(video: origURL.lastPathComponent, renamed: false, reason: "no_full_datetime", newName: nil, start: nil, end: nil))
        return
    }

    // Compute start/end wall-clock
    let start = anchor.addingTimeInterval(-bestAt)
    let end   = start.addingTimeInterval(durationS)

    // Build new name + copy
    let newName = "\(DEPLOY_DATE)_\(ANIMAL_ID)_\(CAMERA_ID)_\(dFmt.string(from: start))_\(tFmt.string(from: start))_\(tFmt.string(from: end)).mp4"
    let target = URL(fileURLWithPath: DIR_PROCESSED).appendingPathComponent(newName)
    do {
        if fm.fileExists(atPath: target.path) { try fm.removeItem(at: target) }
        try fm.copyItem(at: workURL, to: target)
        print("✅ \(newName)")
        summary.append(.init(video: origURL.lastPathComponent, renamed: true, reason: nil, newName: newName, start: start, end: end))
    } catch {
        print("❌ Copy failed: \(error.localizedDescription)")
        summary.append(.init(video: origURL.lastPathComponent, renamed: false, reason: "copy_failed", newName: nil, start: start, end: end))
    }
}

// MARK: - Run
let vids = listVideosWithConversion(INPUT_DIR)
print("Found \(vids.count) video(s) in \(INPUT_DIR)")
for v in vids { processOne(v) }

// Batch summary
var s = "video,renamed,reason,new_name,start,end\n"
let iso = ISO8601DateFormatter()
for it in summary {
    s += "\(it.video),\(it.renamed),\(it.reason ?? ""),\(it.newName ?? ""),\(it.start.map{iso.string(from:$0)} ?? ""),\(it.end.map{iso.string(from:$0)} ?? "")\n"
}
let sumURL = URL(fileURLWithPath: DIR_METADATA).appendingPathComponent("batch_summary.csv")
try? s.data(using: .utf8)?.write(to: sumURL)
print("\n=== Batch summary → \(sumURL.path) ===")
