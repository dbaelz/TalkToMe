package de.dbaelz.ttm.repository

import de.dbaelz.ttm.model.TtsJob
import de.dbaelz.ttm.model.TtsJobEntity

fun TtsJob.toEntity(): TtsJobEntity = TtsJobEntity(
    id = this.id,
    text = this.text,
    config = this.config,
    engine = this.engine,
    createdAt = this.createdAt,
    status = this.status,
    fileId = this.fileId
)

fun TtsJobEntity.toTtsJob(): TtsJob = TtsJob(
    id = this.id,
    text = this.text,
    config = this.config,
    engine = this.engine,
    createdAt = this.createdAt,
    status = this.status,
    fileId = this.fileId
)

