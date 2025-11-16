package ma.emsi.mounib.tp4mounib_web.llm;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import jakarta.enterprise.context.Dependent;

import java.io.Serializable;
import java.net.URL;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Dependent
public class RoutageRagClient implements Serializable {

    private transient Assistant assistant;
    private String systemRole;

    public RoutageRagClient() {
        try {
            initializeAssistant();
        } catch (Exception e) {
            throw new RuntimeException("Erreur lors de l'initialisation de l'assistant", e);
        }
    }

    private void initializeAssistant() throws Exception {
        ApacheTikaDocumentParser parser = new ApacheTikaDocumentParser();
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // Création des récupérateurs de contenu
        ContentRetriever ragRetriever = createRetriever("rag.pdf", parser, embeddingModel);
        ContentRetriever psychologyRetriever = createRetriever("La_psychologie.pdf", parser, embeddingModel);

        // Modèle Gemini
        String geminiApiKey = System.getenv("GEMINI_KEY");
        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(geminiApiKey)
                .modelName("gemini-2.5-flash")
                .logRequestsAndResponses(true)
                .build();

        // Associer chaque ContentRetriever à une description
        Map<ContentRetriever, String> retrieverDescriptions = new HashMap<>();
        retrieverDescriptions.put(ragRetriever, "Documents techniques sur le RAG, AI et embeddings");
        retrieverDescriptions.put(psychologyRetriever, "Documents sur la psychologie");

        // Routeur pour choisir dynamiquement quel retriever utiliser
        QueryRouter queryRouter = new LanguageModelQueryRouter(chatModel, retrieverDescriptions);

        // RAG augmentor
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // Construire l'assistant LangChain
        this.assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .retrievalAugmentor(retrievalAugmentor)
                .build();
    }

    private ContentRetriever createRetriever(String resourceName, ApacheTikaDocumentParser parser, EmbeddingModel embeddingModel) throws Exception {
        URL resourceUrl = getClass().getClassLoader().getResource("/" +resourceName);
        var path = Paths.get(resourceUrl.toURI());

        Document document = FileSystemDocumentLoader.loadDocument(path, parser);
        List<TextSegment> segments = DocumentSplitters.recursive(300, 30).split(document);

        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.5)
                .build();
    }
    public void setSystemRole(String systemRole) {
        this.systemRole = systemRole;

    }

    public interface Assistant {
        String chat(String prompt);
    }

    public String ask(String question) {
        return assistant.chat(question);
    }
}
