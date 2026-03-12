package de.dbaelz.ttm.tts

sealed class TtsConfig

data class PocketTtsConfig(
    val steps: Int = 10,
    val temperature: Float = 0.5f,
    val seed: Int = 0
) : TtsConfig()

data class ChatterboxConfig(
    val exaggeration: Float = 0.5f
) : TtsConfig()
