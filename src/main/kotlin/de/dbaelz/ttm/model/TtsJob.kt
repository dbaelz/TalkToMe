package de.dbaelz.ttm.model

import de.dbaelz.ttm.tts.TtsConfig
import java.time.Instant

data class TtsJob(
    val id: String,
    val text: String,
    val config: TtsConfig = TtsConfig(),
    val createdAt: Instant = Instant.now(),
    var status: JobStatus = JobStatus.PENDING,
    var fileId: String? = null
)

enum class JobStatus { PENDING, RUNNING, DONE, FAILED }
