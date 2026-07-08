import Foundation
import llama

enum VisionProofError: Error {
    case modelLoad
    case contextCreate
    case projectorLoad
    case imageLoad
    case tokenize(Int32)
    case evaluate(Int32)
}

actor Qwen35VisionContext {
    private let model: OpaquePointer
    private let context: OpaquePointer
    private let vision: OpaquePointer
    private let vocab: OpaquePointer
    private let sampler: UnsafeMutablePointer<llama_sampler>
    private var batch: llama_batch
    private var nPast: llama_pos = 0
    private var pendingBytes: [CChar] = []

    private init(model: OpaquePointer, context: OpaquePointer, vision: OpaquePointer) {
        self.model = model
        self.context = context
        self.vision = vision
        self.vocab = llama_model_get_vocab(model)
        let samplerParams = llama_sampler_chain_default_params()
        self.sampler = llama_sampler_chain_init(samplerParams)
        llama_sampler_chain_add(self.sampler, llama_sampler_init_greedy())
        self.batch = llama_batch_init(1, 0, 1)
    }

    deinit {
        llama_batch_free(batch)
        llama_sampler_free(sampler)
        mtmd_free(vision)
        llama_free(context)
        llama_model_free(model)
        llama_backend_free()
    }

    static func create(modelPath: String, projectorPath: String) throws -> Qwen35VisionContext {
        llama_backend_init()
        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = 999
        guard let model = llama_model_load_from_file(modelPath, modelParams) else {
            throw VisionProofError.modelLoad
        }
        var contextParams = llama_context_default_params()
        contextParams.n_ctx = 2048
        contextParams.n_batch = 512
        contextParams.n_ubatch = 256
        contextParams.n_threads = 6
        contextParams.n_threads_batch = 6
        guard let context = llama_init_from_model(model, contextParams) else {
            llama_model_free(model)
            throw VisionProofError.contextCreate
        }
        var visionParams = mtmd_context_params_default()
        visionParams.use_gpu = true
        visionParams.n_threads = 6
        visionParams.print_timings = true
        visionParams.warmup = false
        guard let vision = mtmd_init_from_file(projectorPath, model, visionParams) else {
            llama_free(context)
            llama_model_free(model)
            throw VisionProofError.projectorLoad
        }
        return Qwen35VisionContext(model: model, context: context, vision: vision)
    }

    func prefill(imagePath: String, question: String,
                 externalEmbeddings: [Float]? = nil) throws {
        let bitmapWrapper = imagePath.withCString {
            mtmd_helper_bitmap_init_from_file(vision, $0, false)
        }
        guard let bitmap = bitmapWrapper.bitmap else { throw VisionProofError.imageLoad }
        defer { mtmd_bitmap_free(bitmap) }

        let marker = String(cString: mtmd_default_marker())
        let prompt = "<|im_start|>user\n\(marker)\n\(question)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        guard let chunks = mtmd_input_chunks_init() else { throw VisionProofError.tokenize(-1) }
        defer { mtmd_input_chunks_free(chunks) }

        var input = mtmd_input_text(text: nil, add_special: true, parse_special: true)
        var bitmapPointer: OpaquePointer? = bitmap
        let result = prompt.withCString { promptPointer -> Int32 in
            input.text = promptPointer
            return withUnsafeMutablePointer(to: &bitmapPointer) {
                mtmd_tokenize(vision, chunks, &input, $0, 1)
            }
        }
        guard result == 0 else { throw VisionProofError.tokenize(result) }

        var newPast: llama_pos = 0
        if var externalEmbeddings {
            for index in 0..<mtmd_input_chunks_size(chunks) {
                guard let chunk = mtmd_input_chunks_get(chunks, index) else { continue }
                let isLast = index + 1 == mtmd_input_chunks_size(chunks)
                let evalResult: Int32
                if mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE {
                    let expected = mtmd_input_chunk_get_n_tokens(chunk) * 2560
                    guard externalEmbeddings.count == expected else {
                        throw NSError(
                            domain: "Qwen35ExternalEmbeddings", code: 2,
                            userInfo: [NSLocalizedDescriptionKey:
                                "features=\(externalEmbeddings.count) expected=\(expected) " +
                                "chunk_tokens=\(mtmd_input_chunk_get_n_tokens(chunk))"])
                    }
                    evalResult = externalEmbeddings.withUnsafeMutableBufferPointer { features in
                        mtmd_helper_decode_image_chunk(
                            vision, context, chunk, features.baseAddress,
                            newPast, 0, 512, &newPast, nil, nil)
                    }
                } else {
                    evalResult = mtmd_helper_eval_chunk_single(
                        vision, context, chunk, newPast, 0, 512, isLast, &newPast)
                }
                guard evalResult == 0 else { throw VisionProofError.evaluate(evalResult) }
            }
        } else {
            let evalResult = mtmd_helper_eval_chunks(
                vision, context, chunks, 0, 0, 512, true, &newPast)
            guard evalResult == 0 else { throw VisionProofError.evaluate(evalResult) }
        }
        nPast = newPast
    }

    func reset() {
        llama_memory_clear(llama_get_memory(context), true)
        llama_sampler_reset(sampler)
        nPast = 0
        pendingBytes.removeAll(keepingCapacity: true)
    }

    func generate(maxTokens: Int = 32) -> (String, Int, Double?) {
        var output = ""
        var tokenCount = 0
        var firstTokenSeconds: Double?
        let start = Date()
        while tokenCount < maxTokens {
            let token = llama_sampler_sample(sampler, context, -1)
            llama_sampler_accept(sampler, token)
            if llama_vocab_is_eog(vocab, token) { break }
            let piece = tokenToString(token)
            if firstTokenSeconds == nil && !piece.isEmpty {
                firstTokenSeconds = Date().timeIntervalSince(start)
            }
            output += piece
            llama_batch_clear(&batch)
            llama_batch_add(&batch, token, nPast, [0], true)
            guard llama_decode(context, batch) == 0 else { break }
            nPast += 1
            tokenCount += 1
        }
        return (output, tokenCount, firstTokenSeconds)
    }

    private func tokenToString(_ token: llama_token) -> String {
        var buffer = [CChar](repeating: 0, count: 256)
        var count = llama_token_to_piece(vocab, token, &buffer, Int32(buffer.count), 0, true)
        if count < 0 {
            buffer = [CChar](repeating: 0, count: Int(-count) + 1)
            count = llama_token_to_piece(vocab, token, &buffer, Int32(buffer.count), 0, true)
        }
        if count > 0 { pendingBytes.append(contentsOf: buffer.prefix(Int(count))) }
        guard let string = String(validatingUTF8: pendingBytes + [0]) else { return "" }
        pendingBytes.removeAll(keepingCapacity: true)
        return string
    }
}

