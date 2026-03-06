package de.dbaelz.ttm.onnx

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Component
import java.nio.file.Paths
import kotlin.io.path.absolutePathString

@Component
class OnnxWrapper {
    private val logger = LoggerFactory.getLogger(OnnxWrapper::class.java)
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()

    private val sessions: MutableMap<String, OrtSession> = mutableMapOf()

    fun loadModelFiles(modelsPath: String, modelFiles: List<String>) {
        val unloaded = modelFiles.filterNot { sessions.contains(it) }

        val sessionOpts = OrtSession.SessionOptions()
        val basePath = Paths.get(modelsPath)

        unloaded.forEach { fileName ->
            try {
                val session = env.createSession(
                    basePath.resolve(fileName).absolutePathString(),
                    sessionOpts
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