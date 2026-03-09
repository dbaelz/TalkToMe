package de.dbaelz.ttm.config

import org.springframework.beans.factory.annotation.Value
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor
import java.util.concurrent.Executor

@Configuration
class ExecutorConfig(
    @Value("\${tts.executor.core-pool-size:2}") private val corePoolSize: Int,
    @Value("\${tts.executor.max-pool-size:4}") private val maxPoolSize: Int,
    @Value("\${tts.executor.queue-capacity:500}") private val queueCapacity: Int,
    @Value("\${tts.executor.await-termination-seconds:30}") private val awaitTerminationSeconds: Int,
    @Value("\${tts.executor.thread-name-prefix:tts-exec-}") private val threadNamePrefix: String,
) {

    @Bean(name = ["ttsTaskExecutor"])
    fun ttsTaskExecutor(): Executor {
        val executor = ThreadPoolTaskExecutor()
        executor.setCorePoolSize(corePoolSize)
        executor.setMaxPoolSize(maxPoolSize)
        executor.setQueueCapacity(queueCapacity)
        executor.setThreadNamePrefix(threadNamePrefix)
        executor.setWaitForTasksToCompleteOnShutdown(true)
        executor.setAwaitTerminationSeconds(awaitTerminationSeconds)
        executor.initialize()
        return executor
    }
}
