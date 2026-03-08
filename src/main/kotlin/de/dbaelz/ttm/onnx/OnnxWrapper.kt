package de.dbaelz.ttm.onnx

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.slf4j.LoggerFactory
import org.springframework.beans.factory.annotation.Value
import org.springframework.stereotype.Component
import java.nio.file.Paths
import kotlin.io.path.absolutePathString

@Component
class OnnxWrapper(
    @Value("\${onnx.use-gpu:true}") private val useGpu: Boolean,
    @Value("\${onnx.gpu-device:0}") private val gpuDevice: Int
) {
    private val logger = LoggerFactory.getLogger(OnnxWrapper::class.java)
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()

    private val sessions: MutableMap<String, OrtSession> = mutableMapOf()

    fun loadModelFiles(modelsPath: String, modelFiles: List<String>) {
        val unloaded = modelFiles.filterNot { sessions.contains(it) }

        val basePath = Paths.get(modelsPath)

        val sessionOptions = try {
            if (useGpu) {
                val tryOpts = OrtSession.SessionOptions()
                tryOpts.addCUDA(gpuDevice)
                logger.info("ONNX: CUDA provider successfully initialized for device={}", gpuDevice)
                tryOpts
            } else {
                OrtSession.SessionOptions()
            }
        } catch (_: Throwable) {
            OrtSession.SessionOptions()
        }

        unloaded.forEach { fileName ->
            try {
                val session = env.createSession(
                    basePath.resolve(fileName).absolutePathString(),
                    sessionOptions
                )
                sessions[fileName] = session
            } catch (e: Exception) {
                logger.error("Failed to load ONNX model file: {}", fileName, e)
            }
        }
    }

    fun getEnvironment() = env

    fun getSession() = sessions.toMap()
}