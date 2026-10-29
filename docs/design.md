# langchain-nace design

Reference implementations: `langchain-typesafe` (Jev) and `langchain-openai` decisions middleware.

## Principles

- **No model enum.** Any model id is sent exactly as given, so new Drex releases work without a package release. The only model facts in the code are about released, pinned versions, which never change: `drex-v1.5` and `drex-v2.0` read text only, so images sent to them are refused before sending.
- **Drex is the authority on limits.** Question counts, sizes and image rules are enforced by the server; its 422 is shown with each issue's `path: message`.
- **Forward compatible.** Unknown question types (raw dicts), unknown answer types (`RawAnswer`), new response fields (`.raw`) and new request fields (`model_kwargs`) pass through.
- **Never drop data silently.** The classifier raises on images it can't send. Middleware replaces unreadable media with placeholders, so a photo in the chat doesn't stop an agent.
- **Fail safe.** The tool gate blocks when Drex can't be reached; the model router escalates to the strongest model.
- **Same shape as Jev's package.** Same type names, LangChain `Model*Error` hierarchy, error messages without user data, `omit_payload` trace policy on middleware.

## System design

```mermaid
flowchart TB
    subgraph User["Your code"]
        APP["LCEL chain / LangGraph / create_agent"]
    end

    subgraph Pkg["langchain-nace"]
        direction TB
        subgraph Public["Public API"]
            NC["NaceClassifier<br/>(Runnable: invoke / ainvoke / batch)"]
            NR["NaceRouter<br/>.branch()"]
            NE["nace_conditional_edge()<br/>(LangGraph path)"]
            MR["NaceModelRouterMiddleware<br/>before_agent + wrap_model_call"]
            AM["NaceAutoModeMiddleware<br/>wrap_tool_call"]
        end
        subgraph Core["Core (pure, offline-tested)"]
            ST["_state.encode_state<br/>messages → JSON, lift images"]
            IM["_images.to_drex_image<br/>blocks / bytes / paths → data URLs"]
            TY["types.to_wire<br/>instructions always a string"]
            OR["OpenRouter shaping<br/>images → state content array"]
            RP["ClassifierResponse.from_body<br/>keeps unknown answers + raw"]
            ER["errors.convert_error<br/>Nace* + LangChain Model*Error"]
        end
        GATE["Gate: GateQuestions → GateVerdict →<br/>GatePolicy → Allow / AskHuman / Block"]
    end

    subgraph SDK["nace-sdk"]
        CL["NaceClient / AsyncNaceClient<br/>retries 429/529/5xx (retry-after-ms), 60 s timeout"]
        HK["httpx hook: User-Agent + X-Nace-Client<br/>= langchain-nace/version"]
    end

    subgraph Backends
        NACE["console.nace.ai/v1/systemone"]
        ORT["openrouter.ai/api/v1/systemone"]
        GW["Vercel AI Gateway / TypeSafe-compatible"]
    end

    LS[("LangSmith<br/>llm runs: questions, probabilities,<br/>usage, request id, thresholds")]

    APP --> NC & NR & NE & MR & AM
    NR --> NC
    NE --> NC
    MR --> NC
    AM --> GATE --> NC
    NC --> ST --> IM
    NC --> TY
    NC --> OR
    NC --> CL --> HK
    HK --> NACE & ORT & GW
    CL -->|200| RP --> NC
    CL -->|4xx / 5xx| ER --> NC
    NC -. traces .-> LS
```

### One `invoke`

```mermaid
sequenceDiagram
    participant U as Caller
    participant C as NaceClassifier
    participant S as encode_state / to_wire
    participant K as nace-sdk client
    participant D as Drex
    U->>C: invoke({state, questions, images?})
    C->>C: text-only model + images? → NaceImageModelError (nothing sent)
    C->>S: messages → JSON, images in messages → images[] ("[Image N]")
    S-->>C: state, questions (instructions always a string), images
    C->>C: OpenRouter? images → state content array
    C->>K: system_one(model as given, ...)
    K->>D: POST /v1/systemone (User-Agent, X-Nace-Client)
    alt 429 / 529 / 5xx
        D-->>K: retry-after-ms
        K->>D: retry (max_retries)
    end
    D-->>K: 200 {answers, usage, ...} or 4xx {error.issues}
    K-->>C: response / error
    C-->>U: ClassifierResponse (full probabilities) or Nace*Error (path: message)
```

### Tool gate

```mermaid
flowchart LR
    TC["Agent proposes tool call"] --> G{"tool gated?"}
    G -- no --> RUN["run tool"]
    G -- yes --> CTX["Gate state:<br/>original_request (1st user msg, always)<br/>latest_user_message<br/>recent_messages[-20:] (media → placeholders)<br/>tool_call + tool_description<br/>+ screenshots (Drex Max)"]
    CTX --> Q["ONE Drex request:<br/>risk: Score · destructive: Noul · matches_request: Noul"]
    Q -- "422 on images" --> Q2["retry once without images"] --> V
    Q --> V["GateVerdict<br/>risk = score / top level"]
    Q -- "Drex unreachable" --> B
    V --> P{"ThresholdPolicy"}
    P -- "risk ≥ 0.75, or destructive & not asked" --> B["Block → error ToolMessage"]
    P -- "risk ≥ 0.4, or destructive" --> A["AskHuman → interrupt()<br/>approve / edit / reject"]
    P -- otherwise --> RUN
    A -- approve / edit --> RUN
    A -- reject --> B
```

## Data model

```mermaid
classDiagram
    direction LR
    class ClassifierRequest {
        <<TypedDict>>
        state: State
        questions: Mapping~str, Question~
        images?: Sequence~ImageLike~
    }
    class Question {
        <<union>>
        Noul | Choice | Score | raw dict(any type)
    }
    class Noul {
        type = "noul"
        instructions: str|JSON|None
        criteria: NoulCriteria|None
    }
    class NoulCriteria {
        true: JSON|None
        false: JSON|None
    }
    class Choice {
        type = "choice"
        instructions: str|JSON|None
        criteria: dict~label, description|None~
    }
    class Score {
        type = "score"
        instructions: str|JSON|None
        criteria: list~level~ (low → high)
    }
    class ClassifierResponse {
        model: str
        answers: dict~id, Answer~
        usage: Usage
        request_id: str|None
        evaluation_time_ms: float|None
        raw: dict
        nouls() / choices() / scores()
    }
    class Answer {
        <<union>>
        NoulAnswer | ChoiceAnswer | ScoreAnswer | RawAnswer
    }
    class NoulAnswer { noul: float }
    class ChoiceAnswer {
        choice: str
        probabilities: dict~label, float~
        confidence: float
    }
    class ScoreAnswer {
        score: float
        legend: dict~int, level~
        probabilities: dict~int, float~
        confidence: float
    }
    class RawAnswer { type: str  + any fields }
    class Usage {
        input_tokens
        image_tokens
        output_tokens
    }
    class RouteDecision {
        route: str
        answer: ChoiceAnswer
        fell_back: bool
    }
    class NaceModelRoute {
        <<TypedDict in agent state>>
        route: str
        choice: str|None
        confidence: float|None
        probabilities: dict
        escalated: bool
        reason: confident|low_confidence|error
    }
    class GateQuestions {
        risk: Score
        destructive: Noul
        matches_request: Noul
        extra: Mapping
    }
    class GateVerdict {
        risk: float (0..1)
        p_destructive: float
        p_matches_request: float
        response: ClassifierResponse
    }
    class GateDecision {
        <<abstract>>
        apply(request, handler, verdict)
    }
    class Allow
    class Block { reason: str }
    class AskHuman
    class ThresholdPolicy {
        block_at = 0.75
        ask_at = 0.4
        destructive_at = 0.5
        match_at = 0.5
        ask_human = True
    }

    ClassifierRequest --> Question
    Question <|-- Noul
    Question <|-- Choice
    Question <|-- Score
    Noul --> NoulCriteria
    ClassifierResponse --> Answer
    ClassifierResponse --> Usage
    Answer <|-- NoulAnswer
    Answer <|-- ChoiceAnswer
    Answer <|-- ScoreAnswer
    Answer <|-- RawAnswer
    RouteDecision --> ChoiceAnswer
    NaceModelRoute ..> ChoiceAnswer : flattened copy
    GateQuestions --> Score
    GateQuestions --> Noul
    GateVerdict --> ClassifierResponse
    ThresholdPolicy ..> GateVerdict : reads
    ThresholdPolicy ..> GateDecision : returns
    GateDecision <|-- Allow
    GateDecision <|-- Block
    GateDecision <|-- AskHuman
```

### Wire format (what reaches Drex)

| Field | Sent as |
|---|---|
| `model` | exactly as given (default `drex-max`, chosen by product and moved to the newest flagship each release) |
| `state` | string, or JSON; messages as `{role, content}`; images replaced by `"[Image N]"` |
| `questions.<id>` | `{type, instructions: str, criteria}`; JSON instructions/criteria become JSON text |
| `images` | base64 / data URL / `{type: workspace_file, ...}`; explicit images first, then images from messages |
| OpenRouter | no `images`; `state` = `[{type: text}, {type: image_url, image_url: {url: data:...}}...]` |
| headers | `User-Agent`, `X-Nace-Client` = `langchain-nace/<version>`; OpenRouter also `HTTP-Referer`, `X-OpenRouter-Title` |
