package de.dbaelz.ttm.repository

import de.dbaelz.ttm.model.TtsJob
import de.dbaelz.ttm.model.TtsJobEntity

fun TtsJob.toEntity(): TtsJobEntity = TtsJobEntity(
    id = this.id,
    text = this.text,
    engine = this.engine,
    createdAt = this.createdAt,
    status = this.status,
    fileId = this.fileId
)

fun TtsJobEntity.toTtsJob(): TtsJob = TtsJob(
    id = this.id,
    text = this.text,
    config = when (this.engine) {
        // Default config for listing APIs — persistence service will populate real config when needed
        else -> de.dbaelz.ttm.tts.PocketTtsConfig()
    },
    engine = this.engine,
    createdAt = this.createdAt,
    status = this.status,
    fileId = this.fileId
)
