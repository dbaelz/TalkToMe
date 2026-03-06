package de.dbaelz.ttm.model

import java.time.Instant

data class TtsJob(
    val id: String,
    val text: String,
    val createdAt: Instant = Instant.now(),
    var status: JobStatus = JobStatus.PENDING,
    var fileId: String? = null
)

enum class JobStatus { PENDING, RUNNING, DONE, FAILED }

