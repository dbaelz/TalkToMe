package de.dbaelz.ttm.service

import org.springframework.beans.factory.annotation.Value
import org.springframework.stereotype.Service
import java.io.File
import java.nio.file.Files
import java.util.UUID

@Service
class LocalStorageService(
    @Value("\${storage.path:storage}")
    private val basePath: String
) : StorageService {
    init {
        Files.createDirectories(File(basePath).toPath())
    }

    override fun save(bytes: ByteArray, extension: String): String {
        val id = UUID.randomUUID().toString()
        val file = File(basePath, "${id}.$extension")
        file.writeBytes(bytes)
        return id
    }

    override fun load(id: String, extension: String): ByteArray? {
        val file = File(basePath, "${id}.$extension")
        return if (file.exists()) file.readBytes() else null
    }
}