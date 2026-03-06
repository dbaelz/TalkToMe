package de.dbaelz.ttm

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication

@SpringBootApplication
class TalkToMeApplication

fun main(args: Array<String>) {
    runApplication<TalkToMeApplication>(*args)
}
