package de.dbaelz.ttm.tts.chatterbox

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import org.springframework.beans.factory.annotation.Value
import org.springframework.stereotype.Component
import java.nio.file.Paths

@Component
class ChatterboxTokenizer(
    @Value("\${tts.models.chatterbox:models/chatterbox}") private val modelsPath: String,
    @Value("\${tts.models.chatterbox.tokenizer:tokenizer.json}") private val tokenizerFile: String
) {
    private val tokenizer: HuggingFaceTokenizer by lazy {
        HuggingFaceTokenizer.newInstance(Paths.get(modelsPath).resolve(tokenizerFile))
    }

    fun tokenize(text: String): LongArray {
        return tokenizer.encode(text).ids
    }
}
