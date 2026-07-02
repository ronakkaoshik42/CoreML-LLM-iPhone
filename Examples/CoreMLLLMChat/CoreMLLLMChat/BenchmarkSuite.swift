import CoreGraphics
import CoreMLLLM
import Foundation

/// One-shot manual benchmark runner. Launch-argument automation is intentionally
/// kept out until this path has been validated on a physical iPhone.
final class BenchmarkSuite {
    struct LaunchConfiguration {
        let model: Model
        let mode: Mode
        let maxNewTokens: Int

        init?(arguments: [String]) {
            guard arguments.contains("--run-benchmark-suite") else { return nil }

            func value(for prefix: String) -> String? {
                arguments.first(where: { $0.hasPrefix(prefix) })
                    .map { String($0.dropFirst(prefix.count)) }
            }

            model = Model(rawValue: value(for: "--benchmark-model=") ?? "4B")
                ?? .fourB
            mode = Mode(rawValue: value(for: "--benchmark-mode=") ?? "text")
                ?? .text
            maxNewTokens = max(
                1,
                Int(value(for: "--benchmark-max-new-tokens=") ?? "64") ?? 64)
        }
    }

    enum Model: String {
        case fourB = "4B"
        case eightB = "8B"

        var modelInfo: ModelDownloader.ModelInfo {
            switch self {
            case .fourB: return .qwen3vl_4b_stateful
            case .eightB: return .qwen3vl_8b_stateful
            }
        }
    }

    enum Mode: String {
        case text
        case image
    }

    private let runner: LLMRunner

    init(runner: LLMRunner) {
        self.runner = runner
    }

    @discardableResult
    func run(
        model: Model,
        mode: Mode,
        image: CGImage? = nil,
        maxNewTokens: Int = 64
    ) async -> Bool {
        CoreMLPerfStats.recordResult(
            "[BENCH_START] model=\(model.rawValue) mode=\(mode.rawValue)")

        if mode == .image && image == nil {
            CoreMLPerfStats.recordResult(
                "[RESULT_ERROR] model=\(model.rawValue) mode=image reason=no_image_available")
            recordDone(model: model, mode: mode, success: false)
            return false
        }

        do {
            if !runner.isLoaded || !runner.modelName.contains(model.rawValue) {
                guard let modelURL = ModelDownloader.shared.localModelURL(
                    for: model.modelInfo)
                else {
                    throw BenchmarkSuiteError.modelNotAvailable(model.rawValue)
                }
                try await runner.loadModel(from: modelURL)
            }

            let prompt = mode == .text
                ? "Say hello in exactly 20 words."
                : "Describe this image in one sentence."
            let messages = [ChatMessage(role: .user, content: prompt)]
            let stream = try await runner.generate(
                messages: messages,
                image: mode == .image ? image : nil,
                maxNewTokens: max(1, maxNewTokens))
            var streamedError: String?
            for await text in stream {
                if text.hasPrefix("[Error:") { streamedError = text }
            }
            if let streamedError {
                throw BenchmarkSuiteError.generationFailed(streamedError)
            }

            recordDone(model: model, mode: mode, success: true)
            return true
        } catch {
            let reason = error.localizedDescription
                .replacingOccurrences(of: " ", with: "_")
                .replacingOccurrences(of: "\n", with: "_")
            CoreMLPerfStats.recordResult(
                "[RESULT_ERROR] model=\(model.rawValue) mode=\(mode.rawValue) reason=\(reason)")
            recordDone(model: model, mode: mode, success: false)
            return false
        }
    }

    private func recordDone(model: Model, mode: Mode, success: Bool) {
        CoreMLPerfStats.recordResult(
            "[BENCH_DONE] model=\(model.rawValue) mode=\(mode.rawValue) success=\(success)")
    }
}

private enum BenchmarkSuiteError: LocalizedError {
    case modelNotAvailable(String)
    case generationFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotAvailable(let model):
            return "\(model) model files are not available"
        case .generationFailed(let message):
            return message
        }
    }
}
