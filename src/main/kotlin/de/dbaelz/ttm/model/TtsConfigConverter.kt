package de.dbaelz.ttm.model

import tools.jackson.module.kotlin.jacksonObjectMapper
import tools.jackson.module.kotlin.readValue
import de.dbaelz.ttm.tts.TtsConfig
import jakarta.persistence.AttributeConverter
import jakarta.persistence.Converter

@Converter(autoApply = false)
class TtsConfigConverter : AttributeConverter<TtsConfig, String> {
    private val mapper = jacksonObjectMapper()

    override fun convertToDatabaseColumn(attribute: TtsConfig?): String {
        return try {
            if (attribute == null) "" else mapper.writeValueAsString(attribute)
        } catch (e: Exception) {
            ""
        }
    }

    override fun convertToEntityAttribute(dbData: String?): TtsConfig {
        return try {
            if (dbData.isNullOrBlank()) TtsConfig() else mapper.readValue(dbData)
        } catch (e: Exception) {
            TtsConfig()
        }
    }
}
