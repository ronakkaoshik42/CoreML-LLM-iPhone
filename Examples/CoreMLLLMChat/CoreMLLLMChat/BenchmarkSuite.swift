import CoreGraphics
import CoreMLLLM
import Foundation

/// Manual and launch-argument benchmark runner for one or more generations.
final class BenchmarkSuite {
    struct LaunchConfiguration {
        let model: Model
        let mode: Mode
        let maxNewTokens: Int
        let repeatCount: Int
        let runTag: String

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
            repeatCount = max(
                1,
                Int(value(for: "--benchmark-repeat-count=") ?? "1") ?? 1)
            runTag = value(for: "--benchmark-run-tag=") ?? "unspecified"
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
        maxNewTokens: Int = 64,
        repeatCount: Int = 1,
        runTag: String = "unspecified"
    ) async -> Bool {
        let count = max(1, repeatCount)
        let tagBase = sanitizeTag(runTag)
        var allSucceeded = true

        for run in 1...count {
            let tag = "\(tagBase)_model-\(model.rawValue)_mode-\(mode.rawValue)_run-\(run)"
            let needsLoad = !runner.isLoaded || !runner.modelName.contains(model.rawValue)
            let state = needsLoad ? "cold" : "warm"
            let fields = "model=\(model.rawValue) mode=\(mode.rawValue) "
                + "suite_tag=\(tagBase) tag=\(tag) "
                + "run=\(run)/\(count) state=\(state)"
            CoreMLPerfStats.recordResult("[BENCH_START] \(fields)")
            let succeeded = await runOnce(
                model: model, mode: mode, image: image,
                maxNewTokens: maxNewTokens, fields: fields)
            allSucceeded = allSucceeded && succeeded
        }
        return allSucceeded
    }

    private func runOnce(
        model: Model,
        mode: Mode,
        image: CGImage?,
        maxNewTokens: Int,
        fields: String
    ) async -> Bool {
        if mode == .image && image == nil {
            CoreMLPerfStats.recordResult(
                "[RESULT_ERROR] model=\(model.rawValue) mode=image reason=no_image_available")
            recordDone(fields: fields, success: false)
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
            var generatedText = ""
            for await text in stream {
                if text.hasPrefix("[Error:") { streamedError = text }
                generatedText += text
            }
            if let streamedError {
                throw BenchmarkSuiteError.generationFailed(streamedError)
            }
            if mode == .text {
                let wordCount = generatedText.split(whereSeparator: \.isWhitespace).count
                let isValid = wordCount == 20
                let encodedText = Data(generatedText.utf8).base64EncodedString()
                CoreMLPerfStats.recordResult(
                    "[BENCH_TEXT] \(fields) words=\(wordCount) valid=\(isValid) "
                        + "text_b64=\(encodedText)")
                if !isValid {
                    throw BenchmarkSuiteError.generationFailed(
                        "expected 20 words, received \(wordCount)")
                }
            }

            recordDone(fields: fields, success: true)
            return true
        } catch {
            let reason = error.localizedDescription
                .replacingOccurrences(of: " ", with: "_")
                .replacingOccurrences(of: "\n", with: "_")
            CoreMLPerfStats.recordResult(
                "[RESULT_ERROR] model=\(model.rawValue) mode=\(mode.rawValue) reason=\(reason)")
            recordDone(fields: fields, success: false)
            return false
        }
    }

    private func recordDone(fields: String, success: Bool) {
        CoreMLPerfStats.recordResult("[BENCH_DONE] \(fields) success=\(success)")
    }

    private func sanitizeTag(_ tag: String) -> String {
        let sanitized = tag.map { character in
            character.isLetter || character.isNumber || "-_.".contains(character)
                ? character : "_"
        }
        return sanitized.isEmpty ? "unspecified" : String(sanitized)
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
