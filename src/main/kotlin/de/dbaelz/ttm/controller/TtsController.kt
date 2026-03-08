package de.dbaelz.ttm.controller

import de.dbaelz.ttm.model.TtsJob
import de.dbaelz.ttm.model.TtsProvider
import de.dbaelz.ttm.service.TtsService
import de.dbaelz.ttm.tts.TtsConfig
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.*

@RestController
@RequestMapping("/api/tts")
class TtsController(private val ttsService: TtsService) {

    data class GenerateRequest(val text: String?, val config: TtsConfig? = null, val provider: String? = null)

    @PostMapping
    fun generateAudio(@RequestBody request: GenerateRequest): ResponseEntity<Any> {
        val text = request.text ?: return ResponseEntity.badRequest().build()
        val providerName = request.provider?.trim()?.takeIf { it.isNotEmpty() }
        val provider = if (providerName != null) {
            try {
                TtsProvider.valueOf(providerName.uppercase())
            } catch (_: IllegalArgumentException) {
                val body = mapOf(
                    "error" to "INVALID_PROVIDER",
                    "message" to "Provider '$providerName' is not supported",
                    "allowedProviders" to TtsProvider.entries.map { it.name }
                )
                return ResponseEntity.badRequest().body(body)
            }
        } else {
            TtsProvider.POCKET
        }

        val job = ttsService.generate(text, request.config ?: TtsConfig(), provider)
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
