package de.dbaelz.ttm.service

import de.dbaelz.ttm.model.TtsJob
import de.dbaelz.ttm.model.TtsProvider
import de.dbaelz.ttm.tts.TtsConfig

interface TtsService {
    fun generate(
        text: String,
        config: TtsConfig,
        provider: TtsProvider
    ): TtsJob

    fun getJob(id: String): TtsJob?
    fun getFile(id: String): ByteArray?
}