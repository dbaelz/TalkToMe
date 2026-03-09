package de.dbaelz.ttm

import de.dbaelz.ttm.service.StorageService
import org.springframework.boot.test.context.TestConfiguration
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Primary
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

@TestConfiguration
class TestConfig {
    @Bean
    @Primary
    fun storageService(): StorageService = object : StorageService {
        private val map = ConcurrentHashMap<String, ByteArray>()

        override fun save(bytes: ByteArray, extension: String): String {
            val id = UUID.randomUUID().toString()
            map[id] = bytes
            return id
        }

        override fun load(id: String, extension: String): ByteArray? = map[id]
        override fun exists(id: String, extension: String): Boolean {
            return map.containsKey(id)
        }
    }
}