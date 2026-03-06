package de.dbaelz.ttm.service

import de.dbaelz.ttm.model.TtsJob

interface TtsService {
    fun generate(text: String): TtsJob
    fun getJob(id: String): TtsJob?
    fun getFile(id: String): ByteArray?
}