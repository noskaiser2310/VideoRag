```mermaid
flowchart TD
    %% ==================== PHASE 1 ====================
    subgraph Phase1["<b>Phase 1: Research & System Design</b><br/><i>(Week 1-2)</i>"]
        direction LR
        
        subgraph P1_Tasks["📋 Tasks"]
            direction TB
            P1_1["Problem Definition<br/>- Visual → Movie Retrieval<br/>- Text → Knowledge RAG<br/>- Evaluation KPIs"]
            P1_2["Literature Review<br/>- Video Retrieval<br/>- GraphRAG<br/>- VideoRAG"]
            P1_3["Technology Analysis<br/>- CLIP, FAISS, Neo4j<br/>- VLM Models"]
            P1_4["System Architecture<br/>- Retrieval Pipelines<br/>- Agent Workflow<br/>- Data Layers"]
            P1_1 --> P1_2 --> P1_3 --> P1_4
        end
        
        subgraph P1_Team["👥 Phân công"]
            direction TB
            P1_Son["🔵 <b>Sơn (Lead)</b><br/>- Code Project Skeleton<br/>- Design Workflow<br/>- Setup Core Architecture"]
            P1_Hieu["🟠 <b>Hiếu </b><br/>- Research Video Retrieval<br/>- Test CLIP/VLM models<br/>- Design Visual Pipeline"]
            P1_Thang["🟢 <b>Thắng </b><br/>- Search papers<br/>- Summarize documents<br/>- Note taking"]
            P1_Vinh["🟣 <b>Vinh </b><br/>- Search Dataset sources<br/>- Setup Repo <br/>- Write docs"]
        end
        
        P1_Tasks --- P1_Team
    end

    %% ==================== PHASE 2 ====================
    subgraph Phase2["<b>Phase 2: Data Pipeline & Database</b><br/><i>(Week 3-5)</i>"]
        direction LR
        
        subgraph P2_Tasks["📋 Tasks"]
            direction TB
            P2_1["Dataset Collection<br/>- MovieNet<br/>- MovieGraphs<br/>- IMDB Metadata"]
            P2_2["Video Preprocessing<br/>- Shot Detection<br/>- Scene Segmentation<br/>- Keyframe Extraction"]
            P2_3["Text Processing<br/>- Subtitle Parsing<br/>- Speech-to-Text<br/>- Chunk Builder"]
            P2_4["Embedding & Storage<br/>- CLIP/Text Embedding<br/>- FAISS Index<br/>- Graph DB"]
            P2_1 --> P2_2 --> P2_3 --> P2_4
        end
        
        subgraph P2_Team["👥 Phân công"]
            direction TB
            P2_Son["🔵 <b>Sơn (Lead)</b><br/>- FAISS index optimization<br/>- Text embedding pipeline<br/>- Integrate all modules"]
            P2_Hieu["🟠 <b>Hiếu </b><br/>- Shot detection module<br/>- Keyframe extraction<br/>- CLIP embedding pipeline<br/>- Video preprocessing"]
            P2_Thang["🟢 <b>Thắng </b><br/>- Whisper integration<br/>- Subtitle parsing<br/>- Data cleaning "]
            P2_Vinh["🟣 <b>Vinh </b><br/>- IMDB API crawler<br/>- Download datasets<br/>- Data validation"]
        end
        
        P2_Tasks --- P2_Team
    end

    %% ==================== PHASE 3 ====================
    subgraph Phase3["<b>Phase 3: Retrieval Engine & Agents</b><br/><i>(Week 6-9)</i>"]
        direction LR
        
        subgraph P3_Tasks["📋 Tasks"]
            direction TB
            P3_1["Visual Retrieval Engine<br/>- CLIP Search<br/>- Frame Matching<br/>- Scene Ranking"]
            P3_2["Knowledge RAG Engine<br/>- Text Retrieval<br/>- Context Builder<br/>- Answer Generator"]
            P3_3["Agentic Workflow<br/>- Query Router<br/>- VLM Integration<br/>- Verification Loop"]
            P3_4["Graph Context Engine<br/>- Neo4j Query<br/>- Entity Linking<br/>- Scene Knowledge"]
            P3_1 --> P3_2 --> P3_3 --> P3_4
        end
        
        subgraph P3_Team["👥 Phân công"]
            direction TB
            P3_Son["🔵 <b>Sơn (Lead)</b><br/>- Multi-hop reasoning<br/>- Query Router (6 intents)<br/>- Core RAG logic"]
            P3_Hieu["🟠 <b>Hiếu </b><br/>- Visual retrieval pipeline<br/>- CLIP search + reranking<br/>- Frame matching logic"]
            P3_Thang["🟢 <b>Thắng </b><br/>- Context Builder<br/>- Prompt templates<br/>- Test RAG outputs"]
            P3_Vinh["🟣 <b>Vinh </b><br/>- Neo4j queries<br/>- API endpoints (FastAPI)<br/>- Module integration support"]
        end
        
        P3_Tasks --- P3_Team
    end

    %% ==================== PHASE 4 ====================
    subgraph Phase4["<b>Phase 4: Deployment & Evaluation</b><br/><i>(Week 10)</i>"]
        direction LR
        
        subgraph P4_Tasks["📋 Tasks"]
            direction TB
            P4_1["User Interface<br/>- Gradio UI<br/>- Visual Gallery<br/>- Query Trace"]
            P4_2["System Optimization<br/>- LLM Client<br/>- API Optimization<br/>- Fallback Mechanism"]
            P4_3["Testing<br/>- Run test queries<br/>- Bug fixing<br/>- Performance check"]
            P4_4["Documentation<br/>- Readme<br/>- Final Report<br/>- Demo"]
            P4_1 --> P4_2 --> P4_3 --> P4_4
        end
        
        subgraph P4_Team["👥 Phân công"]
            direction TB
            P4_Son["🔵 <b>Sơn (Lead)</b><br/>- Gradio UI<br/>- Write Readme<br/>- Prepare Demo slides"]
            P4_Hieu["🟠 <b>Hiếu </b><br/>- Visual Gallery component<br/>- Visual pipeline optimization<br/>- VLM integration final"]
            P4_Thang["🟢 <b>Thắng </b><br/>- Query Trace UI<br/>- Run test queries<br/>- Write test reports"]
            P4_Vinh["🟣 <b>Vinh </b><br/>- System Optimization<br/>- LLM Client fallback<br/>- Performance tuning"]
        end
        
        P4_Tasks --- P4_Team
    end

    %% ==================== CONNECTIONS ====================
    Phase1 --> Phase2 --> Phase3 --> Phase4

    %% ==================== STYLING ====================
    classDef phaseStyle fill:#FFFFFF,stroke:#333,stroke-width:2px,rx:10px
    classDef taskBoxStyle fill:#F5F5F5,stroke:#666,stroke-width:1px,rx:5px
    classDef sonStyle fill:#E3F2FD,stroke:#1976D2,stroke-width:2px
    classDef thangStyle fill:#E8F5E9,stroke:#388E3C,stroke-width:2px
    classDef hieuStyle fill:#FFF3E0,stroke:#F57C00,stroke-width:2px
    classDef vinhStyle fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px

    class Phase1,Phase2,Phase3,Phase4 phaseStyle
    class P1_1,P1_2,P1_3,P1_4,P2_1,P2_2,P2_3,P2_4,P3_1,P3_2,P3_3,P3_4,P4_1,P4_2,P4_3,P4_4 taskBoxStyle
    class P1_Son,P2_Son,P3_Son,P4_Son sonStyle
    class P1_Thang,P2_Thang,P3_Thang,P4_Thang thangStyle
    class P1_Hieu,P2_Hieu,P3_Hieu,P4_Hieu hieuStyle
    class P1_Vinh,P2_Vinh,P3_Vinh,P4_Vinh vinhStyle