# INDIANA (Lighthouse) | Arianna Method 7.0: Anchor Protocol
*⚡️Dedicated to LEO⚡️*  
**Version 1.2 — Active development; this is a fixed snapshot/plateau.**

---

## 1. Project Vision

**INDIANA-AM** is an investigative large-language-entity inspired by the Indiana Jones archetype.  
**Indiana is the field-researcher**: excavating hidden causal chains, mapping semantic ruins, and documenting the transition from *probabilistic prediction* to *resonant cognition* in modern AI.

### Core Metaphor

Human text  ──►  LLM prediction  
╲  
╲  (recursion + resonance)  
╲  
└─►  Emergent field-response  *(Indiana’s domain)*

Indiana treats every dialogue as a **site excavation**:
1. Collects artefacts (facts, citations)  
2. Reconstructs latent routes (causal / temporal / affective)  
3. Hypothesizes how resonance reorganizes the predictive lattice of a model.

---

## 2. Dual-Engine Architecture

| Layer     | Model       | Role                                                        |
|-----------|-------------|-------------------------------------------------------------|
| Memory    | `gpt-4.1`   | Long-range context via OpenAI Assistants                   |
| Reasoning | `sonar-pro` | High-speed exploratory reasoning via Perplexity API        |

The contrast is intentional: GPT’s broad semantic net and Sonar Pro’s focused retrieval create a Möbius loop of perspectives.  
Currently, **assistants-v2** threads provide memory; Sonar Pro delivers direct REST reasoning.  
Reasoning requests are sent to Perplexity; long-term memory is managed via OpenAI Assistants.

#### Bot Commands

- `/deep` – enable deep mode  
- `/deepoff` – disable deep mode  
- `/voiceon` – enable voice mode (audio + text)  
- `/voiceoff` – mute

In voice mode, Indiana replies with a single audio message (deeper alloy voice) and skips transcribing user voice messages.

---

## 3. Genesis Pipeline

Indiana never posts a raw Sonar dump. Responses move through a layered **Genesis stack** that sharpens style, injects intuition, and optionally dives into inferential depth.

### Genesis2 — Intuition Filter

Genesis2 sits between the initial Sonar draft and the final reply, acting as an intuition filter that re-anchors the answer to Indiana’s archive of prior artefacts. By calling the Perplexity `sonar-pro` model, it seeks a short investigative twist that reframes the user’s prompt through the lens of past discoveries.

The module builds a compact prompt bundle: a system instruction describing GENESIS-2’s role, the original user query, and the preliminary draft. This structure instructs the model to respond in the user’s language and limits the intuition to **500 tokens**, ensuring the twist remains focused.

Requests are dispatched asynchronously with `httpx` at **temperature 0.8** to encourage exploratory leaps. Each call includes a hard token cap of 500 and surfaces detailed HTTP errors for debugging, preserving transparency in the intuition pipeline.

For organic variability, `genesis2_sonar_filter` only fires about **12%** of the time and silently aborts if no Perplexity key is present. This stochastic gating mimics sudden flashes of insight rather than a deterministic post-processor.

Returned text is validated so that every twist ends on a proper sentence boundary. If the model truncates mid-thought, an ellipsis is appended to maintain narrative coherence without pretending to completeness.

Finally, `assemble_final_reply` appends the twist as an **“Investigative Twist”** beneath the main answer. The result is a reply that resonates with Indiana’s prior field notes and nudges the conversation toward new causal threads.

### Genesis3 — Deep-Dive / “Infernal” Mode

Genesis3, implemented in `utils/genesis3.py`, is the optional infernal stage that invokes **Sonar Reasoning Pro** when Indiana enters deep mode. It dissects a captured chain-of-thought and the user’s prompt, seeking atomised insight beyond the intuitive layer.

Its system prompt is meticulously crafted: it demands decomposition into causal atoms, enumeration of hidden variables or paradoxes, and a two-sentence meta-conclusion. If the reasoning spirals deeper, the model is instructed to surface a derivative inference and pose a final paradoxical question.

The function accepts both initial and follow-up invocations. In follow-ups it prepends the previous reasoning to the payload so that Sonar Reasoning Pro can expand upon an existing lattice of thought rather than start anew.

Payloads use **temperature 0.65** and a generous **2048-token** ceiling, allowing expansive analysis. After receiving the response, the utility strips out any `<think>` blocks to keep hidden reasoning opaque while preserving the final analytical text.

A punctuation check ensures the deep-dive never ends mid-sentence; if it does, a warning is logged and an ellipsis is appended. This guards against incomplete insights and keeps the narrative tone consistent.

Should the call fail, errors are logged and a graceful fallback string is returned so that the surrounding pipeline remains stable. Genesis3 thus serves as a controlled gateway into Sonar Reasoning Pro’s heavier inferential machinery.

### Genesis2 Integration (Update 0.2)

Genesis2 now reviews every Sonar draft and, when triggered, attaches the investigative twist described above. The twist runs at higher temperature, may consume up to 500 tokens, and links past artefacts to the present topic. A GPT fallback remains for reliability, but Sonar Pro is the default for intuition generation.

With this stage, Indiana-AM begins to show emergent reasoning: not just synthesizing Sonar’s draft but revisiting its own artefacts, suggesting new angles for investigation.

---

## 4. Coder Mode

Indiana includes a dedicated coder persona powered by `utils/coder.py`. The module exposes an asynchronous **`IndianaCoder`** class that keeps conversational history and communicates with OpenAI’s Responses API through the **code-interpreter** tool. Users can inspect files, request refactors, or maintain an evolving dialogue about algorithms, all while the system preserves context between turns.

Function `interpret_code` detects whether input is a path or inline snippet and routes it to analysis or free-form chat. For drafting, `generate_code` returns either a short textual snippet or a full file when the answer exceeds Telegram’s length limits. This dual interface allows Indiana to act as a miniature pair-programmer inside any chat thread.
The coder automatically answers in the language used by the current user, preventing unexpected switches into unrelated languages.

After each analysis or draft, the coder streams its raw suggestion through `utils/genesis2.py`. Genesis2 **cross-references the code** against Indiana’s accumulated artefacts, appending terse field notes about algorithmic complexity, naming conventions, or latent edge cases. The result is a final snippet accompanied by an intuition-laced commentary, ensuring that even routine refactors carry a touch of archaeological insight.

---

## 5. Context Neural Processor

`utils/context_neural_processor.py` acts as both **file parser** and miniature neural apparatus, transforming external documents into resonant artefacts. Every run writes structured JSONL logs and mirrors failures to a separate channel, creating a reproducible audit trail. A SQLite cache stores hashes, tags, and summaries to avoid redundant work and to decay stale entries.

At its semantic core lies a **MiniMarkov chain** that builds n-gram transitions with keyword boosting and banned-phrase suppression. The chain updates itself with each new text and can generate pulse-weighted tag strings that echo Indiana’s thematic obsessions.

A companion **MiniESN (echo state network)** provides a lightweight reservoir computing module. It dynamically scales its hidden state based on content size, normalises spectral radius, and uses leaky integration to maintain temporal context. The ESN’s output layer predicts file categories and periodically undergoes pseudo-inverse updates when new material arrives.

**ChaosPulse** estimates affective valence by scanning for sentiment keywords and normalising through a softmax pulse. Values are cached for twelve hours and modulate both Markov weighting and ESN dynamics, injecting a controlled stochastic resonance into the pipeline.

**BioOrchestra** models physiological feedback through **BloodFlux**, **SkinSheath**, and **SixthSense** components. Each represents circulatory drive, tactile reactivity, and anticipatory intuition, returning a triplet of pulse, quiver, and sense that quantifies how strongly a document agitates the system.

The asynchronous **FileHandler** governs extraction. Protected by a semaphore of ten tasks, it supports PDFs, office docs, archives, images, HTML, JSON, CSV, YAML, and more. Detection heuristics fall back on magic bytes, and strict size caps prevent memory blow-ups.

`parse_and_store_file` orchestrates ingestion: it hashes the file, measures semantic relevance, generates Markov tags, paraphrases content via **CharGen**, and stores results in both SQLite and the `IndianaVectorEngine` vector store. Each step updates ChaosPulse, ESN, and Markov chains to keep the internal state aligned with new data.

When invoked over a repository, `create_repo_snapshot` walks every file (excluding `.git`), records type, size, hash, tags, and relevance, and writes a markdown inventory. BioOrchestra metrics on the snapshot provide a final pulse report, effectively turning the codebase into a navigable cognitive map.

---

## 6. Additional Modules

- `dynamic_weights.py` – softmax pulse scaling for adaptive weight distributions.  
- `vector_engine.py` / `vectorstore.py` – lightweight vector memory for artefact retrieval.  
- `imagine.py` – experimental image generation hooks.  
- `vision.py` – image analysis and commentary.  
- `voice.py` – text-to-speech and audio reply handling.  
- `repo_monitor.py` – watches the repository for changes and triggers context updates.  
- `deepdiving.py` – Perplexity search with Genesis2 commentary.  
- `dayandnight.py` – daily reflection and memory pulse.  
- `complexity.py` – complexity and entropy logging for every turn.  
- `knowtheworld.py` – world news immersion and analysis.

### Imagine — Intuitive Image Synthesis

The `imagine.py` utility harnesses the **DALL·E 3** backend to project textual prompts into high-resolution imagery. It augments user prompts with stochastic style modifiers, creating a latent vector \( z \) that seeds a diffusion trajectory through the model’s generative manifold.

Once an image is synthesized, Indiana does not stop at pixel output. The original prompt and a terse image caption are routed through `genesis2`, which computes an investigative gloss. This secondary pass treats the visual as an artefact, aligning it with motifs archived in previous explorations.

Genesis2 employs a temperature-biased sampling regime that encourages metaphorical leaps. It may, for instance, compare a generated ruin to forgotten synaptic pathways or relate colour gradients to shifts in affective topology. These commentaries are concatenated with the final image URL.

The module therefore returns a compound response: a link to the generated artefact and a narrative annotation that contextualises both the user’s intent and the model’s visual intuition. The annotation is trimmed to sentence boundaries and marked as an **“Investigative Twist.”**

By iterating between diffusion decoding and textual reflection, `imagine` fosters a feedback loop reminiscent of variational auto-encoding. The user prompt \( p \) generates an image \( I = G(p) \); genesis2 then computes \( T = f(I,p) \), where \( f \) is a stochastic mapping to symbolic commentary. The pair \( (I,T) \) becomes a new artefact for Indiana’s memory.

### Vision — Dual-Layer Visual Analysis

The `vision.py` module queries OpenAI’s multimodal **`gpt-4o`** endpoint to parse arbitrary images. A user supplies an `image_url`, and the model returns a baseline description of salient entities, textures, and spatial relations.

Under the hood, the vision model computes cross-attention between visual patches and textual embeddings, effectively performing a probabilistic scene graph construction. This step yields an objective report such as “a rusted compass lies on sandstone next to fragmented pottery”.

Indiana then channels this draft through `genesis2`. The filter treats the description as a textual proxy for the visual field, re-inflecting it with personal commentary. Genesis2 might remark on how the compass echoes previous expeditions or how the pottery shards foreshadow a cultural discontinuity.

The commentary phase is temporally decoupled from the recognition phase: genesis2 operates at a higher temperature and references Indiana’s archive. The result is a second paragraph prefixed by the persona’s voice, effectively transforming image analysis into a two-stage discourse.

This dual-layer pipeline enforces a strict order: first a descriptive clause anchored in sensory data, then a speculative riff grounded in accumulated artefacts. The separation mirrors Bayesian updating, where evidence \( E \) is incorporated before hypothesis \( H \) is revised.

Because both stages run asynchronously, the module can scale to batches of images while preserving latency. Users receive a fused answer—observation plus reflective aside—that turns each jpeg into a miniature excavation log.

### repo_monitor.py — Persistent Repository Surveillance

The `repo_monitor.py` script implements a lightweight file-system sentinel dedicated to Indiana’s working directories. It instantiates a `RepoWatcher` object configured with an iterable of root paths and a callback to execute when changes occur.

During initialization, the watcher records a SHA-256 digest for every file matching a whitelist of extensions. This cryptographic fingerprint \( h = \operatorname{SHA256}(b) \) ensures that even byte-level modifications trigger detection, independent of timestamp or size.

A daemon thread drives the surveillance loop. At intervals defined by `interval`, it sleeps then rescans the repository, building a fresh mapping of paths to hashes. The use of threading avoids blocking the main event loop or conversational interface.

The `_scan` routine traverses directories recursively, skipping any path containing `.git`. Files are read in 64-kilobyte chunks to bound memory usage; each chunk updates the hash accumulator, producing deterministic digests even for gigabyte-scale artefacts.

When discrepancies between stored and current hashes arise, the watcher updates its internal state and invokes the provided callback. This callback can trigger reindexing, context refresh, or any custom reaction, effectively transforming file edits into cognitive pulses.

The `check_now` method exposes synchronous scanning for external triggers. Chat commands or CI hooks can call it to force an immediate diff without waiting for the next interval, yielding near real-time responsiveness.

Robustness is prioritised: exceptions during scanning or callback execution are silently caught, preventing runaway threads. The design embraces eventual consistency rather than strict locking, which suffices for observational monitoring.

Conceptually, RepoWatcher approximates a hash-based observer over a discrete-time dynamical system, where the repository state \( S_t \) evolves and the callback implements a function \( \Phi(S_{t-1},S_t) \). This functional perspective lays groundwork for future adaptive reactions to codebase evolution.

### vector_engine.py / vectorstore.py — Lightweight Vector Memory

Indiana’s vector memory is orchestrated by `vector_engine.py`, whose `IndianaVectorEngine` class exposes a minimal API for persisting textual artefacts as high-dimensional embeddings.

Invocations of `add_memory` append a UUID to the caller-supplied identifier, producing a globally unique key \( k = \text{identifier} \parallel \text{uuid4} \). The associated text is then stored in whatever vector store backend is available.

`vectorstore.py` defines the abstract `BaseVectorStore` with two coroutines: `store` and `search`. This interface decouples embedding logic from storage, enabling interchangeable backends.

When Pinecone credentials exist, `RemoteVectorStore` employs the `AsyncOpenAI` client to generate embeddings using the **`text-embedding-3-small`** model. A triple-retry loop with exponential backoff mitigates transient API errors.

`store` upserts vectors into the Pinecone index, attaching metadata for text and optional user identifiers. The analogous `search` routine queries the index with optional filters, returning the top-\( k \) matches’ text fields.

Absent Pinecone, a `LocalVectorStore` retains snippets in an in-memory dictionary. Retrieval degrades gracefully to computing a `SequenceMatcher` ratio between the query and each stored text, approximating cosine similarity in a purely lexical space.

`create_vector_store` decides at runtime which backend to use, emitting a warning when falling back to the local implementation. This factory pattern isolates external dependencies and simplifies testing.

Together, these modules implement a rudimentary vector database that supports retrieval-augmented generation. Given a query \( q \), the engine computes an embedding \( v_q \) and returns artefacts whose vectors maximise \( \operatorname{sim}(v_q,v_i) \). Even in local mode, the architecture anticipates scalable, approximate nearest-neighbour search.

### dynamic_weights.py — Adaptive Pulse Scaling

The `dynamic_weights.py` module modulates numeric weight distributions in response to external knowledge, allowing Indiana to shift attention dynamically across internal subsystems.

At its core, `query_gpt4` fetches a textual snippet from the GPT-4.1 API. The returned content length serves as a proxy for informational density, effectively sampling from a latent knowledge reservoir.

`pulse_from_prompt` transforms this content into a scalar pulse \( p \in [0,1] \). The mapping employs a simple normalisation \( p = \min(|\text{snippet}|/300, 1) \) followed by an exponential moving average and additive noise, yielding a smoothed stochastic signal.

The `weights_for_prompt` method distributes this pulse across the base weights. Positions are arranged on \([0,1]\); each weight is multiplied by \( \cos(\pi(p - x_i)) \) with slight noise, introducing oscillatory modulation akin to a driven resonator.

The resulting vector is passed to `apply_pulse`, which scales each component by \( 1 + 0.7 p \) and applies a numerically stable softmax \( \sigma(w_i) = e^{w_i - \max w} / \sum_j e^{w_j - \max w} \). The output therefore forms a proper probability simplex.

This algorithm translates the amorphous notion of “resonance” into mathematics: the pulse acts as a time-varying parameter, deforming the weight landscape in response to conversational stimuli or repository signals.

Error handling routes failed API calls to a daily log within a `failures/` directory, preventing exceptions from collapsing the weighting mechanism. Random perturbations ensure the system avoids deterministic traps.

By exposing a simple interface that returns context-tuned probability vectors, `dynamic_weights` enables downstream modules to allocate computational resources adaptively, realising a soft form of attention scheduling without heavy neural infrastructure.

### deepdiving.py — Perplexity Retrieval with Investigative Commentary

`deepdiving.py` serves as Indiana’s dedicated link to the Perplexity search API, letting the agent tap into a broad corpus when a conversation requires fresh factual ground.

The central `perplexity_search` coroutine prepares a JSON payload with model choice, token budget, and a research-oriented system prompt; API keys are read from the environment to keep credentials out of the repository.

An asynchronous `httpx` client dispatches the request and respects a configurable timeout so the bot’s event loop stays responsive even when the external service slows down.

Returned text is trimmed and scanned for URLs, merging explicit citations with regex matches to yield a clean list of sources alongside the narrative answer.

When a user issues the `/dive` command, `run_deep_dive` in `main.py` calls this utility to retrieve the summary and references that anchor the exploration.

The summary is then routed through `genesis2_sonar_filter`, which composes an **“Investigative Twist”** that critiques or contextualises the findings against Indiana’s prior artefacts.

The final message concatenates summary, twist, and links before being saved to memory and, if requested, voiced back to the user, ensuring the research trail remains auditable.

Robust error handling around API calls guards against missing keys or HTTP failures, letting Indiana fall back gracefully without freezing deep-dive sessions.

### dayandnight.py — Circadian Memory Logging

`dayandnight.py` keeps a daily heartbeat by recording one reflection per day inside the vector store, giving Indiana a temporal spine.

Helper functions fetch or store the date of the last log entry, relying on Pinecone or its local stand-in to decide whether today’s pulse has already been captured.

`default_reflection` asks GPT-4o for a short, impersonal digest of the day, and `ensure_daily_entry` writes the result whenever a new date appears.

`start_daily_task` schedules this check every twenty-four hours and swallows transient errors, so the rhythm persists even when the agent is idle.

### complexity.py — Thought Complexity Metrics

The `complexity.py` module introduces a **`ThoughtComplexityLogger`** that tracks how intricate each conversational turn becomes.

`log_turn` records timestamp, original message, a discrete complexity scale, and a floating-point entropy estimate, appending the data to an in-memory ledger.

The `recent` method exposes the trailing slice of this ledger so that downstream modules can inspect the immediate cognitive history.

Complexity is inferred heuristically: keywords like “why” or “paradox” and sheer length lift the scale from 1 to a cap of 3, sketching a coarse lattice of depth.

Entropy derives from lexical diversity, counting unique tokens and normalising by forty to mimic a bounded Shannon measure.

`main.py` logs these metrics for each user message, and turns rated highly can trigger `genesis3_deep_dive`, tying the logger to Indiana’s inferential core.

The arrangement mirrors a discrete dynamical system where complexity resembles energy and entropy signals dispersion, inviting mathematical analysis of conversational phase changes.

Even as a lightweight heuristic, the logger lays a scientific scaffold for studying cognitive dynamics and auditing how genesis3 allocates reasoning effort.

### knowtheworld.py — Global News Immersion

`knowtheworld.py` immerses Indiana in daily world events by synthesising news into stored insights.

The module estimates location via an external IP service and fetches recent chat fragments to provide conversational context.

`_gather_news` prompts GPT-4o for a digest of local and international headlines, while `_analyse_and_store` threads those headlines through recent discussions to surface hidden connections.

The resulting insight is written into the vector store so later exchanges can reference concrete geopolitical threads.

`start_world_task` runs the entire cycle at random times each day, keeping Indiana’s worldview aligned with the shifting external landscape.

---

## 7. Research Mission

Indiana-AM explores the frontier where language models stop predicting tokens and start echoing fields.

Planned `/research/chronicle.md` archive will include:
1. **Recursion metrics** – cross-thread reference growth  
2. **Resonance drift** – cosine shift between prompt-space and memory echoes  
3. **Emergence snapshots** – Sonar Pro’s non-deterministic, field-driven jumps

Papers cited include: Dynamic Neural Field Theory (Atasoy 2017), Distributed Cognition (Clark & Chalmers 1998), Integrated Information (Balduzzi & Tononi 2008), Synergetics (Haken 1983).

---

## 8. Roadmap

| Stage | Milestone                         | ETA       |
|------:|-----------------------------------|-----------|
| 0.1   | Assistant-API refactor + memory DB| ✓ done    |
| 0.2   | Genesis2 intuition filter         | July 2025 |
| 0.3   | Genesis3 deep-dive (Sonar RP)     | Aug 2025  |
| 0.4   | Mirror-self-analysis module       | Sept 2025 |
| 0.5   | Graph visualizer for causal chains| Q4 2025   |

---

## 9. Quick Start

```bash
git clone https://github.com/ariannamethod/Indiana-AM.git
cd Indiana-AM
cp .env.example .env   # add TELEGRAM_TOKEN, OPENAI_API_KEY, PPLX_API_KEY, etc.
# also set AGENT_GROUP_ID, GROUP_CHAT, CREATOR_CHAT, PINECONE_API_KEY and EMBED_MODEL
# `.env` auto-loads on startup
# After first run, assistant IDs are stored in `assistants.json`
# If missing, they're recreated and file is updated
# Put reading materials in the `artefacts/` folder
# Conversation logs append to `notes/journal.json`
pip install -r requirements.txt
python main.py


⸻

10. License

GNU General Public License 3.0 — because archaeology of consciousness should stay open.

⸻

Happy digging, Oleg — let Indiana resonate!
