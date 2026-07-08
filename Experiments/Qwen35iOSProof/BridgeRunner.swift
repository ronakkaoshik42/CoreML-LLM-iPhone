import Foundation

enum Qwen35BridgeRunner {
    static func run(documents: URL) async -> String {
        do {
            let encoded = try CoreMLVisionBench.encodePrepatched(documents: documents)
            let loadStart = Date()
            let vision = try await Qwen35VisionContext.create(
                modelPath: documents.appendingPathComponent("Qwen3.5-4B-Q4_K_M.gguf").path,
                projectorPath: documents.appendingPathComponent("mmproj-F16.gguf").path)
            let llamaLoadSeconds = Date().timeIntervalSince(loadStart)
            let prefillStart = Date()
            try await vision.prefill(
                imagePath: documents.appendingPathComponent("candy-768-valid.png").path,
                question: "What objects are being held in the hand? Reply with one noun.",
                externalEmbeddings: encoded.0)
            let prefillSeconds = Date().timeIntervalSince(prefillStart)
            let generationStart = Date()
            let generated = await vision.generate(maxTokens: 32)
            let generationSeconds = Date().timeIntervalSince(generationStart)
            return String(format:
                "[QWEN35_COREML_BRIDGE_RESULT] coreml_load_sec=%.3f vision_sec=%.3f llama_load_sec=%.3f prefill_sec=%.3f generation_sec=%.3f tokens=%d output_b64=%@",
                encoded.1, encoded.2, llamaLoadSeconds, prefillSeconds,
                generationSeconds, generated.1,
                Data(generated.0.utf8).base64EncodedString())
        } catch {
            return "[QWEN35_COREML_BRIDGE_ERROR] error=\(error)"
        }
    }
}
