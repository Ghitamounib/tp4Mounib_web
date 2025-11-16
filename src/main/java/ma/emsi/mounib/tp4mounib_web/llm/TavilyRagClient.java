package ma.emsi.mounib.tp4mounib_web.llm;


import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import jakarta.enterprise.context.Dependent;

import java.io.InputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Client RAG avec PDF locaux et recherche web via Tavily (variables renommées pour clarté)
 */
@Dependent
public class TavilyRagClient implements Serializable {

    public interface ChatAssistant {
        String chat(String prompt);
    }

    private String roleSysteme;
    private transient ChatMemory memoryChat;
    private transient ChatAssistant llmAssistant;

    public TavilyRagClient() {
        try {
            setupAssistant();
        } catch (Exception e) {
            throw new RuntimeException("Erreur lors de l'initialisation de l'assistant TavilyRagClientRenamed", e);
        }
    }

    private void setupAssistant() throws Exception {
        String geminiApiKey = System.getenv("GEMINI_KEY");
        ChatModel geminiModel = GoogleAiGeminiChatModel.builder()
                .apiKey(geminiApiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.7)
                .logRequestsAndResponses(true)
                .build();

        // --- Préparer les PDF ---
        ApacheTikaDocumentParser tikaParser = new ApacheTikaDocumentParser();
        EmbeddingModel embeddingGem = new AllMiniLmL6V2EmbeddingModel();

        List<ContentRetriever> retrieversPdf = new ArrayList<>();
        retrieversPdf.add(createPdfRetriever("rag.pdf", tikaParser, embeddingGem));
        retrieversPdf.add(createPdfRetriever("La_psychologie.pdf", tikaParser, embeddingGem));

        // --- Préparer Tavily WebSearch ---
        String tavilyApiKey = System.getenv("TAVILY_KEY");
        WebSearchEngine tavilyEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyApiKey)
                .build();

        ContentRetriever retrieverWeb = WebSearchContentRetriever.builder()
                .webSearchEngine(tavilyEngine)
                .build();

        // --- Routage ---
        List<ContentRetriever> allRetrievers = new ArrayList<>(retrieversPdf);
        allRetrievers.add(retrieverWeb);

        QueryRouter queryRouter = new DefaultQueryRouter(allRetrievers);

        // --- Mémoire de chat ---
        this.memoryChat = MessageWindowChatMemory.withMaxMessages(10);

        // --- Retrieval augmentor ---
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // --- Construction de l'assistant ---
        this.llmAssistant = AiServices.builder(ChatAssistant.class)
                .chatModel(geminiModel)
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(memoryChat)
                .build();
    }

    private ContentRetriever createPdfRetriever(String nomFichier,
                                                DocumentParser parserPdf,
                                                EmbeddingModel embModel) throws Exception {
        InputStream inputPdf = getClass().getClassLoader().getResourceAsStream(nomFichier);
        if (inputPdf == null) {
            throw new RuntimeException("Fichier introuvable dans resources : " + nomFichier);
        }

        Document doc = parserPdf.parse(inputPdf);
        List<TextSegment> segmentsTexte = DocumentSplitters.recursive(500, 50).split(doc);
        List<Embedding> listeEmbeddings = embModel.embedAll(segmentsTexte).content();

        EmbeddingStore<TextSegment> storeEmb = new InMemoryEmbeddingStore<>();
        storeEmb.addAll(listeEmbeddings, segmentsTexte);

        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeEmb)
                .embeddingModel(embModel)
                .maxResults(3)
                .minScore(0.5)
                .build();
    }

    public void setSystemRole(String role) {
        this.roleSysteme = role;
        this.memoryChat.clear();
        memoryChat.add(dev.langchain4j.data.message.SystemMessage.from(role));
    }

    public String ask(String prompt) {
        return llmAssistant.chat(prompt);
    }
}

