package de.dbaelz.ttm.service

import de.dbaelz.ttm.model.TtsJob
import de.dbaelz.ttm.model.TtsEngine
import de.dbaelz.ttm.tts.TtsConfig

interface TtsJobPersistenceService {
    fun createJob(text: String, engine: TtsEngine, config: TtsConfig): TtsJob
    fun findJob(id: String): TtsJob?
}

