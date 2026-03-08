package de.dbaelz.ttm.service

interface StorageService {
    fun save(bytes: ByteArray, extension: String = "wav"): String
    fun load(id: String, extension: String = "wav"): ByteArray?
}
