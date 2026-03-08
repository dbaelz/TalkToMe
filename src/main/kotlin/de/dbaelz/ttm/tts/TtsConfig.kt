package de.dbaelz.ttm.tts

data class TtsConfig(
    val steps: Int = 10,
    val temperature: Float = 0.5f,
    val seed: Int = 0
)