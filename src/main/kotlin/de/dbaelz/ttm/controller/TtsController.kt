package de.dbaelz.ttm.controller

import de.dbaelz.ttm.model.TtsJob
import de.dbaelz.ttm.model.TtsEngine
import de.dbaelz.ttm.service.TtsService
import de.dbaelz.ttm.tts.TtsConfig
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.*

@RestController
@RequestMapping("/api/tts")
class TtsController(private val ttsService: TtsService) {

    data class GenerateRequest(val text: String?, val config: TtsConfig? = null, val engine: String? = null)

    @PostMapping
    fun generateAudio(@RequestBody request: GenerateRequest): ResponseEntity<Any> {
        val text = request.text ?: return ResponseEntity.badRequest().build()
        val engineName = request.engine?.trim()?.takeIf { it.isNotEmpty() }
        val engine = if (engineName != null) {
            try {
                TtsEngine.valueOf(engineName.uppercase())
            } catch (_: IllegalArgumentException) {
                val body = mapOf(
                    "error" to "INVALID_ENGINE",
                    "message" to "Engine '$engineName' is not supported",
                    "allowedEngines" to TtsEngine.entries.map { it.name }
                )
                return ResponseEntity.badRequest().body(body)
            }
        } else {
            TtsEngine.POCKET
        }

        val job = ttsService.generate(text, request.config ?: TtsConfig(), engine)
        return ResponseEntity.accepted().body(job)
    }

    @GetMapping("/jobs/{id}")
    fun getJob(@PathVariable id: String): ResponseEntity<TtsJob> {
        val job = ttsService.getJob(id) ?: return ResponseEntity.notFound().build()
        return ResponseEntity.ok(job)
    }

    @GetMapping("/files/{id}")
    fun download(@PathVariable id: String): ResponseEntity<ByteArray> {
        val audio = ttsService.getFile(id) ?: return ResponseEntity.notFound().build()
        return ResponseEntity.ok()
            .header("Content-Type", "audio/wav")
            .body(audio)
    }
}
