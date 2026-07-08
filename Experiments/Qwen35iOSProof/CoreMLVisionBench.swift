import CoreML
import Foundation

enum CoreMLVisionBench {
    static func encodePrepatched(documents: URL) throws -> ([Float], Double, Double) {
        let modelURL = documents.appendingPathComponent("qwen35_vision_768x576.mlmodelc")
        let inputURL = documents.appendingPathComponent("qwen35_vision_768x576_input_f16.bin")
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndGPU
        let loadStart = Date()
        let model = try MLModel(contentsOf: modelURL, configuration: configuration)
        let loadSeconds = Date().timeIntervalSince(loadStart)
        let inputData = try Data(contentsOf: inputURL, options: .mappedIfSafe)
        let expectedBytes = 1728 * 1536 * MemoryLayout<UInt16>.size
        guard inputData.count == expectedBytes else {
            throw NSError(domain: "CoreMLVisionBench", code: 1)
        }
        let pixels = try MLMultiArray(shape: [1728, 1536], dataType: .float16)
        inputData.withUnsafeBytes { source in
            memcpy(pixels.dataPointer, source.baseAddress!, expectedBytes)
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: ["pixel_values": pixels])
        let encodeStart = Date()
        let prediction = try model.prediction(from: provider)
        let encodeSeconds = Date().timeIntervalSince(encodeStart)
        guard let output = prediction.featureValue(for: "image_features")?.multiArrayValue else {
            throw NSError(domain: "CoreMLVisionBench", code: 2)
        }
        let source = output.dataPointer.bindMemory(to: UInt16.self, capacity: output.count)
        let features = (0..<output.count).map { Float(Float16(bitPattern: source[$0])) }
        return (features, loadSeconds, encodeSeconds)
    }

    static func run(documents: URL, runs: Int = 3) throws -> String {
        let modelURL = documents.appendingPathComponent("qwen35_vision_768.mlmodelc")
        let inputURL = documents.appendingPathComponent("qwen35_vision_768_input_f16.bin")
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndGPU

        let loadStart = Date()
        let model = try MLModel(contentsOf: modelURL, configuration: configuration)
        let loadSeconds = Date().timeIntervalSince(loadStart)
        let inputData = try Data(contentsOf: inputURL, options: .mappedIfSafe)
        let expectedBytes = 2304 * 1536 * MemoryLayout<UInt16>.size
        guard inputData.count == expectedBytes else {
            throw NSError(domain: "CoreMLVisionBench", code: 1,
                          userInfo: [NSLocalizedDescriptionKey:
                            "input bytes=\(inputData.count), expected=\(expectedBytes)"])
        }
        let pixels = try MLMultiArray(shape: [2304, 1536], dataType: .float16)
        inputData.withUnsafeBytes { source in
            memcpy(pixels.dataPointer, source.baseAddress!, expectedBytes)
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: ["pixel_values": pixels])

        _ = try model.prediction(from: provider)
        var seconds: [Double] = []
        var checksum = 0.0
        for _ in 0..<runs {
            let start = Date()
            let output = try model.prediction(from: provider)
            seconds.append(Date().timeIntervalSince(start))
            if let features = output.featureValue(for: "image_features")?.multiArrayValue {
                let values = features.dataPointer.bindMemory(to: UInt16.self, capacity: features.count)
                for index in stride(from: 0, to: features.count, by: 4096) {
                    checksum += Double(Float16(bitPattern: values[index]))
                }
            }
        }
        let mean = seconds.reduce(0, +) / Double(seconds.count)
        return String(format:
            "[QWEN35_COREML_VISION_RESULT] backend=cpu_gpu resolution=768 patches=2304 load_sec=%.3f warmup=1 runs=%d mean_sec=%.3f min_sec=%.3f max_sec=%.3f checksum=%.6f",
            loadSeconds, runs, mean, seconds.min()!, seconds.max()!, checksum)
    }
}

