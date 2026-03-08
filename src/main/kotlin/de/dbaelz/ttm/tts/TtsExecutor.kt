package de.dbaelz.ttm.tts

import de.dbaelz.ttm.model.TtsJob

interface TtsExecutor {
    operator fun invoke(job: TtsJob): ByteArray
}