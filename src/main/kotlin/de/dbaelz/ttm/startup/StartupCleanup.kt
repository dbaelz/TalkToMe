package de.dbaelz.ttm.startup

import de.dbaelz.ttm.model.JobStatus
import de.dbaelz.ttm.repository.JobRepository
import de.dbaelz.ttm.service.StorageService
import org.slf4j.LoggerFactory
import org.springframework.boot.context.event.ApplicationReadyEvent
import org.springframework.context.event.EventListener
import org.springframework.stereotype.Component

@Component
class StartupCleanup(
    private val repo: JobRepository,
    private val storage: StorageService
) {
    private val logger = LoggerFactory.getLogger(StartupCleanup::class.java)

    @EventListener(ApplicationReadyEvent::class)
    fun onApplicationReady() {
        var removed = 0
        repo.findAll().forEach { entity ->
            if (entity.status == JobStatus.DONE && entity.fileId != null) {
                val fileExists = storage.load(entity.fileId!!) != null
                if (!fileExists) {
                    repo.deleteById(entity.id)
                    removed++
                    logger.info("Removed job ${entity.id} because file ${entity.fileId} is missing")
                }
            }
        }
        logger.info("Startup cleanup completed. Removed $removed jobs.")
    }
}

