import Foundation
import Darwin.Mach
import os

/// Lightweight console metrics for on-device Core ML benchmarks.
public enum CoreMLPerfStats {
    private static let resultLock = NSLock()

    public static func physFootprintBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size
        )
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? info.phys_footprint : 0
    }

    public static func availableMemoryBytes() -> UInt64 {
#if os(iOS) || os(tvOS) || os(watchOS)
        UInt64(os_proc_available_memory())
#else
        0
#endif
    }

    public static func now() -> CFAbsoluteTime {
        CFAbsoluteTimeGetCurrent()
    }

    public static func gb(_ bytes: UInt64) -> String {
        String(format: "%.2f", Double(bytes) / 1024.0 / 1024.0 / 1024.0)
    }

    public static func log(_ label: String) {
        let used = physFootprintBytes()
        let available = availableMemoryBytes()
        print("[PERF] \(label) | used_gb=\(gb(used)) | available_gb=\(gb(available))")
    }

    public static func logInterval(
        _ label: String,
        start: CFAbsoluteTime,
        end: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()
    ) {
        print("[PERF] \(label)=\(String(format: "%.3f", end - start))")
    }

    /// Prints and appends a machine-readable result to Documents.
    public static func recordResult(_ line: String) {
        print(line)
        resultLock.lock()
        defer { resultLock.unlock() }
        guard let url = resultLogURL,
              let data = "\(ISO8601DateFormatter().string(from: Date()))\t\(line)\n"
                .data(using: .utf8)
        else { return }
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: data)
            return
        }
        guard let handle = try? FileHandle(forWritingTo: url) else { return }
        defer { try? handle.close() }
        do {
            try handle.seekToEnd()
            try handle.write(contentsOf: data)
        } catch {
            print("[PERF] benchmark result write failed: \(error.localizedDescription)")
        }
    }

    public static var resultLogURL: URL? {
        try? FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ).appendingPathComponent("benchmark_results.log")
    }

    public static func storedResultLines() -> [String] {
        resultLock.lock()
        defer { resultLock.unlock() }
        guard let url = resultLogURL,
              let text = try? String(contentsOf: url, encoding: .utf8)
        else { return [] }
        return text.split(whereSeparator: \.isNewline).map(String.init)
    }
}
