package de.dbaelz.ttm.service

import de.dbaelz.ttm.model.TtsJob
import de.dbaelz.ttm.model.TtsEngine
import de.dbaelz.ttm.tts.TtsConfig

interface TtsService {
    fun generate(
        text: String,
        config: TtsConfig,
        engine: TtsEngine
    ): TtsJob

    fun getJob(id: String): TtsJob?
    fun getFile(id: String): ByteArray?
}