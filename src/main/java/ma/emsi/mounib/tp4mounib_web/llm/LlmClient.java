package ma.emsi.mounib.tp4mounib_web.llm;


import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.data.message.SystemMessage;
import jakarta.enterprise.context.Dependent;
import java.io.Serializable;


/**
 * Classe métier pour communiquer avec le LLM (Gemini) via LangChain4j.
 */
@Dependent
public class LlmClient implements Serializable {


    private String systemRole;
    private transient ChatMemory chatMemory;
    private transient Assistant assistant;

    public LlmClient() {
        // initialisation du modèle et de l'assistant
        init();
    }

    private void init() {
        String apiKey = System.getenv("GEMINI_KEY");
        if (apiKey == null) throw new IllegalStateException("Clé API manquante.");

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .build();

        this.chatMemory = MessageWindowChatMemory.withMaxMessages(10);
        this.assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .build();
    }

    public void setSystemRole(String systemRole) {
        this.systemRole = systemRole;
        if (chatMemory != null) {
            chatMemory.clear();
            chatMemory.add(new SystemMessage(systemRole));
        }
    }

    public String ask(String question) {
        if (assistant == null) init();
        return assistant.chat(question);
    }

    public interface Assistant {
        String chat(String prompt);
    }
}

