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
        repo.findAll().asSequence()
            .forEach { entity ->
                when (entity.status) {
                    JobStatus.FAILED -> {
                        repo.deleteById(entity.id)
                        removed++
                        logger.info("Removed FAILED job ${entity.id}")
                    }

                    JobStatus.DONE if entity.fileId != null -> {
                        when (fileExists(entity.fileId!!)) {
                            true -> {}

                            false -> {
                                repo.deleteById(entity.id)
                                removed++
                                logger.info("Removed job ${entity.id} because file ${entity.fileId} is missing")
                            }

                            null -> {
                                logger.warn("File exists check failed for file ${entity.fileId} of job ${entity.id}")
                            }
                        }
                    }

                    else -> {}
                }
            }

        logger.info("Startup cleanup completed. Removed $removed jobs.")
    }

    private fun fileExists(fileId: String): Boolean? {
        return try {
            storage.exists(fileId)
        } catch (_: Throwable) {
            null
        }
    }
}
