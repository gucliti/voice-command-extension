/**
 * /voice — dictate a prompt: record audio -> local ASR transcription -> paste into the Pi editor.
 *
 * Recording modes:
 *   - /voice          VAD mode: starts on speech, auto-stops after ~1s of silence (max 30s)
 *   - /voice 10       fixed mode: record exactly 10 seconds
 *
 * Configuration (user-tunable, e.g. for different mic environments like headphones):
 *   - /voice config                 show the effective config
 *   - /voice config threshold 400   set one key (threshold | silenceMs | maxSeconds | leadInMs | source)
 *   - /voice config preset headset  apply an environment preset (headset | normal | noisy)
 *   - /voice config noise           measure ambient noise for 3s and suggest a threshold
 *   - /voice config reset           restore defaults
 *   - /voice config cleanTranscript false  disable filler-word cleanup (Qwen only)
 *   - /voice load                   warm up the ASR model in the background (non-blocking)
 *   - /voice -t 400                 one-shot threshold override for this invocation
 *   - /voice -m Qwen/Qwen3-ASR-1.7B one-shot ASR model override (see ASR_MODELS)
 *
 * Config file: ~/.pi/agent/voice.json (created on first write).
 *
 * Pipeline (all steps verified end-to-end):
 *   1. ffmpeg -f pulse -i <source>  records from a PulseAudio microphone
 *   2. POST to the appropriate ASR endpoint:
 *      • Whisper → http://localhost:7999/v1/audio/transcriptions
 *      • Qwen → http://localhost:7999/v1/chat/completions (audio_url)
 *   3. ctx.ui.pasteToEditor(text)  inserts the transcript into the editor
 *      (falls back to stdout in headless/print mode, where pasteToEditor is a no-op)
 *
 * Verified gotchas:
 *   - The ASR server may scale models to 0 replicas when idle and report
 *     HEALTHY while running_replicas is 0. Readiness must be judged by
 *     running_replicas >= 1, otherwise the transcription request hangs for
 *     60s and returns 500.
 *   - /v1/models may require ?limit=2000; the default list can be incomplete.
 *   - The transcription text may start with a leading space; trim before use.
 *   - The command handler's first argument is a raw string, not an array.
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync, renameSync, rmSync, mkdtempSync } from "node:fs";
import { execFile, spawn } from "node:child_process";
import net from "node:net";
import { promisify } from "node:util";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const execFileP = promisify(execFile);

/** Uniform error-to-string for user-facing messages. */
function errMsg(e: unknown): string {
	return e instanceof Error ? e.message : String(e);
}

const API_BASE = process.env.ASR_API_BASE ?? "http://localhost:7999";
// Local (fully offline) ASR: a Python project with bundled models.
const LOCAL_ASR_DIR = process.env.LOCAL_ASR_DIR ?? join(homedir(), ".local", "share", "localASR");
const DEFAULT_ASR_MODEL = "openai/whisper-large-v3-turbo";
const QWEN_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B";
const ASR_MODELS: Record<string, { protocol: "transcriptions" | "chat" | "local"; engine?: string }> = {
	[DEFAULT_ASR_MODEL]: { protocol: "transcriptions" }, // OpenAI /v1/audio/transcriptions
	[QWEN_ASR_MODEL]: { protocol: "chat" },              // chat.completions + audio_url (data URI); output: "language XX<asr_text>..."
	"local/sensevoice": { protocol: "local", engine: "sensevoice" }, // offline, ~200ms, zh/en/ja/ko/yue
	"local/whisper-base": { protocol: "local", engine: "base" },     // offline, multilingual, slower
};

function supportedModels(): string {
	return Object.keys(ASR_MODELS).join(", ");
}
const MAX_SECONDS = 60;
const POLL_INTERVAL_MS = 5_000;
const LOAD_TIMEOUT_MS = 180_000; // max time to wait for a model to load after triggering (observed loads take 60-130s)

// VAD defaults. Example ambient noise measurement (4s idle capture, 50ms frames):
// p50=47, p95=302, max=1477 (keyboard/fan spikes). Speech RMS is typically >1000.
// Use /voice config noise to re-measure for your environment (e.g. headphones).
const VAD_DEFAULTS = {
	threshold: 600,      // RMS above this counts as speech
	silenceMs: 1000,     // stop after this much trailing silence
	maxSeconds: 30,      // hard cap
	leadInMs: 3000,      // give up if no speech within this window
	source: "default",   // Pulse source ("default" = system default source)
	model: DEFAULT_ASR_MODEL, // ASR model (see ASR_MODELS for supported values)
	cleanTranscript: true, // ask Qwen3-ASR (chat protocol) to omit filler words in the transcript
};
type VadConfig = typeof VAD_DEFAULTS;

// Environment presets for /voice config preset <name>: one-tap tuning for the
// three common microphone setups.
const PRESETS: Record<string, Partial<VadConfig>> = {
	headset: { threshold: 150, silenceMs: 800 },   // close mic, quiet room
	normal: { threshold: 600, silenceMs: 1000 },   // built-in mic, typical room
	noisy: { threshold: 1500, silenceMs: 1200 },   // far-field or noisy room
};

const SAMPLE_RATE = 16000;
const FRAME_MS = 50;
const FRAME_SAMPLES = (SAMPLE_RATE * FRAME_MS) / 1000; // 800 samples
const FRAME_BYTES = FRAME_SAMPLES * 2; // 1600 bytes of 16-bit PCM per frame

// /voice config noise: capture window, and minimum frames (500ms) required.
const NOISE_MEASURE_MS = 3000;
const NOISE_MIN_FRAMES = 10;

// VAD robustness:
const SPEECH_CONFIRM_FRAMES = 3;  // consecutive speech frames before committing (150ms); a single spike (keyboard/fan) no longer starts capture
const PRE_ROLL_FRAMES = 8;        // ring of pre-speech frames; at commit it holds SPEECH_CONFIRM_FRAMES burst frames plus 5 pure pre-speech frames (250ms), preserving soft word onsets
const TRAIL_SILENCE_KEEP_MS = 250; // trailing silence kept in the output (rest trimmed)
const MIN_SPEECH_MS = 300;        // captures with less speech are treated as false triggers
const STREAM_STALL_TIMEOUT_MS = 5_000; // next() gives up if ffmpeg produces no data this long

const CONFIG_PATH = join(homedir(), ".pi", "agent", "voice.json");
// Debug dumps (recorded audio, raw responses) are private dictation; keep
// them out of world-readable /tmp.
const DEBUG_DIR = join(homedir(), ".pi", "agent", "voice-debug");

// Single source of truth for numeric config ranges, used by both the
// /voice config setter and loadConfig. threshold is capped at
// int16 max: RMS above 32768 can never occur.
const NUM_LIMITS: Record<string, [number, number]> = {
	threshold: [50, 32_768],
	silenceMs: [200, 60_000],
	maxSeconds: [5, 120],
	leadInMs: [500, 60_000],
};

function parseNum(v: string, key: string): number {
	const n = Number(v);
	const [min, max] = NUM_LIMITS[key];
	if (!Number.isFinite(n) || n < min || n > max) throw new Error(`${key} must be a number in [${min}, ${max}]`);
	return n;
}

function loadConfig(): VadConfig {
	const cfg = { ...VAD_DEFAULTS };
	try {
		if (existsSync(CONFIG_PATH)) {
			const raw = JSON.parse(readFileSync(CONFIG_PATH, "utf8")) as Record<string, unknown>;
			// Sanitize: the file is hand-editable; any value that fails the same
			// validators the /voice config setter uses falls back to the default.
			for (const key of Object.keys(CONFIG_KEYS) as Array<keyof VadConfig>) {
				const v = raw[key];
				if (typeof v !== "string" && typeof v !== "number" && typeof v !== "boolean") continue;
				try {
					cfg[key] = CONFIG_KEYS[key](String(v));
				} catch { /* keep default */ }
			}
		}
	} catch { /* fall through to defaults */ }
	return cfg;
}

function saveConfig(cfg: VadConfig): void {
	// Atomic write: temp file + rename, so a crash mid-write cannot corrupt
	// the config.
	mkdirSync(join(CONFIG_PATH, ".."), { recursive: true });
	const tmp = `${CONFIG_PATH}.tmp`;
	writeFileSync(tmp, JSON.stringify(cfg, null, 2) + "\n");
	renameSync(tmp, CONFIG_PATH);
}

function formatConfig(cfg: VadConfig): string {
	const lines = [
		`/voice config (file: ${CONFIG_PATH})`,
		`  threshold  ${cfg.threshold}   (RMS; higher = less sensitive)`,
		`  silenceMs  ${cfg.silenceMs}   (trailing silence before auto-stop)`,
		`  maxSeconds ${cfg.maxSeconds}  (hard cap for VAD recording)`,
		`  leadInMs   ${cfg.leadInMs}    (wait for speech before giving up)`,
		`  source     ${cfg.source}      (Pulse source; "default" = system default source)`,
		`  model      ${cfg.model}   (ASR model; supported: ${supportedModels()})`,
		`  cleanTranscript ${cfg.cleanTranscript}   (Qwen only: omit filler words like "um", "uh")`,
	];
	return lines.join("\n");
}

/** 44-byte WAV header for 16-bit mono 16kHz PCM of the given length. */
function wavHeader(pcmLen: number): Buffer {
	const h = Buffer.alloc(44);
	h.write("RIFF", 0);
	h.writeUInt32LE(36 + pcmLen, 4);
	h.write("WAVE", 8);
	h.write("fmt ", 12);
	h.writeUInt32LE(16, 16);
	h.writeUInt16LE(1, 20);
	h.writeUInt16LE(1, 22);
	h.writeUInt32LE(SAMPLE_RATE, 24);
	h.writeUInt32LE(SAMPLE_RATE * 2, 28);
	h.writeUInt16LE(2, 32);
	h.writeUInt16LE(16, 34);
	h.write("data", 36);
	h.writeUInt32LE(pcmLen, 40);
	return h;
}

/** Minimal 0.2s silent wav as a data URI -- used to trigger on-demand model loading. */
function silentWavDataUri(): string {
	const pcmLen = (SAMPLE_RATE * 2 * 200) / 1000; // 0.2s mono 16-bit = 6400 bytes
	const buf = Buffer.concat([wavHeader(pcmLen), Buffer.alloc(pcmLen)]);
	return `data:audio/wav;base64,${buf.toString("base64")}`;
}

type ModelState = { running: number } | { missing: true } | { down: true };

async function modelStatus(id: string): Promise<ModelState> {
	try {
		const res = await fetch(`${API_BASE}/v1/models?limit=2000`, { signal: AbortSignal.timeout(10_000) });
		if (!res.ok) return { down: true };
		const data = (await res.json()) as {
			data: Array<{ id: string; deployment_status?: { running_replicas: number } }>;
		};
		const entry = data.data.find((d) => d.id === id);
		if (!entry) return { missing: true };
		// Servers without replica info (plain OpenAI-compatible) are assumed ready.
		return { running: entry.deployment_status?.running_replicas ?? 1 };
	} catch {
		return { down: true }; // API unreachable
	}
}

// Dedupe concurrent load waits so two /voice calls share one poll loop.
// Each wait keeps a listener set so a second caller still gets status updates.
type LoadWait = { promise: Promise<void>; listeners: Set<(s: string) => void> };
const loadWaits = new Map<string, LoadWait>();

/**
 * Wait until the model has a running replica. If none is running, send a
 * sacrificial request in the background to trigger on-demand loading (the
 * server starts the deployment on any inference request, but that request
 * itself times out with 500 after 60s). Bounded by LOAD_TIMEOUT_MS: on timeout the
 * caller reports "loading, try again later" instead of hanging.
 * Fails fast when the model id is unknown or the API is unreachable.
 */
async function ensureModelLoaded(id: string, onStatus?: (s: string) => void): Promise<void> {
	const existing = loadWaits.get(id);
	if (existing) {
		// Join the in-flight wait instead of starting a second poll loop, but
		// keep this caller's status callback attached.
		if (onStatus) existing.listeners.add(onStatus);
		try {
			await existing.promise;
		} finally {
			if (onStatus) existing.listeners.delete(onStatus);
		}
		return;
	}
	const listeners = new Set<(s: string) => void>();
	if (onStatus) listeners.add(onStatus);
	const wait: LoadWait = {
		listeners,
		promise: doEnsureModelLoaded(id, (s) => { for (const l of listeners) { try { l(s); } catch { /* a broken listener must not kill the shared load loop */ } } }),
	};
	loadWaits.set(id, wait);
	try {
		await wait.promise;
	} finally {
		loadWaits.delete(id);
	}
}

/** Fire-and-forget request that makes the server start loading the model. */
function triggerLoad(id: string): void {
	let req: Promise<unknown>;
	if (ASR_MODELS[id].protocol === "transcriptions") {
		const fd = new FormData();
		fd.append("file", new Blob([Buffer.alloc(0)]), "empty.wav");
		fd.append("model", id);
		req = fetch(`${API_BASE}/v1/audio/transcriptions`, { method: "POST", body: fd, signal: AbortSignal.timeout(10_000) });
	} else {
		req = fetch(`${API_BASE}/v1/chat/completions`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ model: id, messages: [{ role: "user", content: [{ type: "audio_url", audio_url: { url: silentWavDataUri() } }] }] }),
			signal: AbortSignal.timeout(10_000),
		});
	}
	// Expected to fail (500 while the model is still loading); the side effect is what matters.
	void req.catch(() => { /* ignore */ });
}

async function doEnsureModelLoaded(id: string, onStatus?: (s: string) => void): Promise<void> {
	const start = Date.now();
	const deadline = start + LOAD_TIMEOUT_MS;
	let triggered = false;
	let lastState: ModelState = { down: true };
	while (true) {
		lastState = await modelStatus(id);
		if ("missing" in lastState) throw new Error(`model ${id} not found in /v1/models`);
		if ("running" in lastState && lastState.running >= 1) return;
		// Unreachable API before we triggered anything is fatal: there is no load in flight to wait for.
		if ("down" in lastState && !triggered) throw new Error(`ASR API unreachable at ${API_BASE}`);
		if (Date.now() >= deadline) break;
		if (!triggered) {
			triggered = true;
			onStatus?.(`Loading ${id} (up to ~${LOAD_TIMEOUT_MS / 1000}s)...`);
			triggerLoad(id);
		} else {
			onStatus?.(`Loading ${id}... ${Math.round((Date.now() - start) / 1000)}s/${LOAD_TIMEOUT_MS / 1000}s`);
		}
		await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
	}
	throw new Error("down" in lastState
		? `ASR API unreachable at ${API_BASE}`
		: `${id} is still loading after ${LOAD_TIMEOUT_MS / 1000}s. It was triggered and will be ready soon -- please try /voice again in a minute or two.`);
}

async function transcribe(wavPath: string, model: string, cleanTranscript: boolean): Promise<string> {
	if (!Object.hasOwn(ASR_MODELS, model)) throw new Error(`unsupported ASR model: ${model} (supported: ${supportedModels()})`);
	const spec = ASR_MODELS[model];
	if (spec.protocol === "transcriptions") {
		const fd = new FormData();
		fd.append("file", new Blob([readFileSync(wavPath)]), "audio.wav");
		fd.append("model", model);
		const res = await fetch(`${API_BASE}/v1/audio/transcriptions`, {
			method: "POST",
			body: fd,
			signal: AbortSignal.timeout(120_000),
		});
		if (!res.ok) throw new Error(`transcribe HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
		const body = (await res.json()) as { text?: string };
		return (body.text ?? "").trim();
	}

	// local protocol: resident daemon first (warm ~0.1s), one-shot fallback.
	if (spec.protocol === "local") {
		const engine = spec.engine!;
		try {
			return await transcribeLocal(wavPath, engine);
		} catch (e) {
			// 3-tier resilience: daemon failed -> try one-shot so
			// dictation never fully breaks (e.g. socket disabled, daemon wedged).
			const py = join(LOCAL_ASR_DIR, ".venv", "bin", "python");
			const script = join(LOCAL_ASR_DIR, "asr_file.py");
			if (!localAsrAvailable()) throw e;
			const { stdout, stderr } = await execFileP(py, [script, wavPath, "--model", engine], {
				timeout: 120_000,
				maxBuffer: 4 * 1024 * 1024,
				cwd: LOCAL_ASR_DIR,
			});
			const text = stdout.trim();
			if (!text) throw new Error(`local ASR returned no text: ${stderr.slice(0, 200)}`);
			return text;
		}
	}

	// chat protocol (Qwen3-ASR): audio as data URI, output "language XX<asr_text>..."
	const b64 = readFileSync(wavPath).toString("base64");
	const content: Array<Record<string, unknown>> = [
		{ type: "audio_url", audio_url: { url: `data:audio/wav;base64,${b64}` } },
	];
	if (cleanTranscript) {
		// The filler-word list is Chinese on purpose: the words to strip ARE
		// Chinese (plus their English equivalents), so translating them would
		// defeat the instruction. This is domain data, not UI text.
		content.unshift({
			type: "text",
			text: "Transcribe the audio verbatim, but omit filler words and hesitation sounds (such as 嗯、啊、呃、那个、就是、um、uh、er、erm、ah、oh、I mean、you know) without changing the meaning or word order. Only remove them when they are pure fillers, not when they carry meaning. Output only the transcript.",
		});
	}
	const body = {
		model,
		messages: [{ role: "user", content }],
	};
	const res = await fetch(`${API_BASE}/v1/chat/completions`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(body),
		signal: AbortSignal.timeout(120_000),
	});
	if (!res.ok) throw new Error(`transcribe HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
	const rawText = await res.text();
	// Keep the raw response for debugging (audio size, response shape).
	try {
		mkdirSync(DEBUG_DIR, { recursive: true });
		writeFileSync(join(DEBUG_DIR, "last_resp.json"), JSON.stringify({ apiBase: API_BASE, audioBytes: Math.round(b64.length * 3 / 4), response: rawText.slice(0, 2000) }, null, 2), { mode: 0o600 });
	} catch { /* ignore */ }
	let json: { choices?: Array<{ message: { content?: string } }> };
	try {
		json = JSON.parse(rawText);
	} catch (e) {
		throw new Error(`transcribe: invalid JSON response: ${String(e).slice(0, 100)}`);
	}
	const raw = json.choices?.[0]?.message?.content ?? "";
	// Output format: "language <Lang><asr_text>...text..." with an optional
	// closing </asr_text>. The language name is directly concatenated to the
	// text (no space), so do NOT strip the prefix with a greedy \S+ regex --
	// it would eat the whole transcript. Anchor on the <asr_text> tag instead.
	const idx = raw.indexOf("<asr_text>");
	// A well-behaved Qwen3-ASR response always carries the tag (even for silence
	// it is "language None<asr_text>"). If it is missing the model returned
	// something that is not a transcript (a refusal, an echo of the prompt, a
	// protocol change), so treat it as empty rather than pasting raw model
	// output into the editor. The raw text is already dumped to DEBUG_DIR above.
	if (idx < 0) return "";
	const text = raw.slice(idx + "<asr_text>".length).replace(/<\/asr_text>\s*$/, "");
	return text.trim();
}

function rmsOf(frame: Buffer): number {
	// Copy to an aligned buffer: Int16Array over a subarray with an odd
	// byteOffset would throw RangeError.
	const copy = Buffer.from(frame);
	const samples = new Int16Array(copy.buffer, copy.byteOffset, copy.length / 2);
	let sum = 0;
	for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
	return Math.sqrt(sum / samples.length);
}

/** Write 16-bit mono 16kHz PCM as a WAV file. */
function writeWav(path: string, pcm: Buffer): void {
	writeFileSync(path, Buffer.concat([wavHeader(pcm.length), pcm]));
}

/**
 * The PCM payload of a WAV file, found by walking RIFF chunks. ffmpeg's WAV
 * muxer inserts a LIST/INFO chunk, so the data chunk is NOT always at byte 44;
 * writeWav output has no extra chunks, but both are handled.
 */
function wavDataChunk(buf: Buffer): Buffer {
	let off = 12;
	while (off + 8 <= buf.length) {
		const id = buf.toString("latin1", off, off + 4);
		const size = buf.readUInt32LE(off + 4);
		if (id === "data") return buf.subarray(off + 8, off + 8 + size);
		off += 8 + size + (size & 1);
	}
	throw new Error("no data chunk in WAV");
}

/**
 * Stream raw PCM from a Pulse source in real time. Yields chunks as they
 * arrive; the caller re-frames them. Used by both VAD recording and noise
 * measurement. next(timeoutMs) resolves null on EOF, on a stall longer than
 * timeoutMs, or throws on ffmpeg error.
 */
function openPcmStream(source: string): {
	next: (timeoutMs: number) => Promise<Buffer | null>;
	close: () => void;
} {
	const proc = spawn("ffmpeg", [
		"-y", "-loglevel", "error", "-f", "pulse", "-i", source,
		"-ar", String(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-",
	]);
	const stderr: string[] = [];
	proc.stderr.on("data", (d: Buffer) => stderr.push(d.toString()));

	const queue: Buffer[] = [];
	let closed = false;
	let waiter: (() => void) | null = null;
	const wakeUp = () => { const w = waiter; waiter = null; w?.(); };
	proc.stdout.on("data", (d: Buffer) => { queue.push(d); wakeUp(); });
	proc.on("error", (e) => { stderr.push(e.message); closed = true; wakeUp(); });
	proc.on("close", (code) => { if (code !== 0) stderr.push(`exit ${code}`); closed = true; wakeUp(); });

	// Wait for the next chunk. Resolves null on EOF or if no data arrives
	// within timeoutMs (stall protection). Throws on ffmpeg error.
	const next = async (timeoutMs: number): Promise<Buffer | null> => {
		while (queue.length === 0 && !closed) {
			const woken = await new Promise<boolean>((res) => {
				const timer = setTimeout(() => { waiter = null; res(false); }, timeoutMs);
				waiter = () => { clearTimeout(timer); res(true); };
			});
			if (!woken && queue.length === 0) return null; // stalled
		}
		if (queue.length === 0) {
			const err = stderr.join(" ").slice(0, 200);
			if (err) throw new Error(`ffmpeg: ${err}`);
			return null;
		}
		return queue.shift()!;
	};

	const close = () => { try { proc.kill("SIGTERM"); } catch { /* ignore */ } };
	return { next, close };
}

/**
 * VAD recording. Keeps a PRE_ROLL_FRAMES ring and commits after
 * SPEECH_CONFIRM_FRAMES consecutive speech frames (a single noise spike does
 * not trigger); soft word onsets survive via the pre-roll. Stops after
 * cfg.silenceMs of trailing silence or the maxSeconds cap, and trims the tail
 * to TRAIL_SILENCE_KEEP_MS. Returns the captured PCM, or null if no speech
 * was detected within the lead-in window.
 */
async function recordWithVad(cfg: VadConfig, onStatus: (s: string) => void): Promise<Buffer | null> {
	const stream = openPcmStream(cfg.source);
	const preRoll: Buffer[] = [];  // ring of the last PRE_ROLL_FRAMES frames
	const frames: Buffer[] = [];   // committed capture
	let burst = 0;                 // consecutive speech frames
	let speechFrames = 0;          // speech frames in the committed capture
	let lastSpeechAt = -1;         // index in frames of the last speech frame
	let started = false;
	let silenceMs = 0;
	let elapsedMs = 0;
	let live = false;              // first PCM data received
	let pending = Buffer.alloc(0);
	const keepTrailing = Math.ceil(TRAIL_SILENCE_KEEP_MS / FRAME_MS);

	const finish = (): Buffer | null => {
		if (!started || speechFrames * FRAME_MS < MIN_SPEECH_MS) return null;
		const end = Math.min(frames.length, lastSpeechAt + 1 + keepTrailing);
		return Buffer.concat(frames.slice(0, end));
	};

	try {
		const deadline = Date.now() + cfg.leadInMs + cfg.maxSeconds * 1000;
		while (Date.now() < deadline) {
			const chunk = await stream.next(STREAM_STALL_TIMEOUT_MS);
			if (chunk === null) {
				// A working mic always emits PCM (even during silence), so no data
				// at all means the capture pipeline is dead, not that the user
				// stayed quiet.
				if (!live) throw new Error(`no audio from PulseAudio source "${cfg.source}" (is the microphone connected?)`);
				break; // normal end after some audio
			}
			if (!live) {
				live = true; // only report "listening" once audio is actually flowing
				onStatus(`Listening... (stop after ~${Math.round(cfg.silenceMs / 1000)}s silence, max ${cfg.maxSeconds}s)`);
			}
			pending = Buffer.concat([pending, chunk]);

			while (pending.length >= FRAME_BYTES) {
				const frame = Buffer.from(pending.subarray(0, FRAME_BYTES)); // copy: must outlive pending
				pending = pending.subarray(FRAME_BYTES);
				elapsedMs += FRAME_MS;
				const speech = rmsOf(frame) >= cfg.threshold;

				if (!started) {
					// Keep a rolling pre-speech ring; once SPEECH_CONFIRM_FRAMES
					// consecutive speech frames appear, commit the ring (which
					// already contains the burst) as the start of the capture.
					preRoll.push(frame);
					if (preRoll.length > PRE_ROLL_FRAMES) preRoll.shift();
					burst = speech ? burst + 1 : 0;
					if (burst >= SPEECH_CONFIRM_FRAMES) {
						started = true;
						frames.push(...preRoll);
						speechFrames = burst;
						lastSpeechAt = frames.length - 1;
					} else if (elapsedMs >= cfg.leadInMs && burst === 0) {
						return null; // no speech within the lead-in window (burst > 0 means speech is starting: grant it time to confirm)
					}
					continue;
				}

				frames.push(frame);
				if (speech) {
					speechFrames++;
					lastSpeechAt = frames.length - 1;
					silenceMs = 0;
				} else {
					silenceMs += FRAME_MS;
				}
				if (silenceMs >= cfg.silenceMs || elapsedMs >= cfg.maxSeconds * 1000) {
					return finish();
				}
			}
		}
		return finish();
	} finally {
		stream.close();
	}
}

/** Measure ambient noise for ~3s and return percentile stats of per-frame RMS. */
async function measureNoise(cfg: VadConfig): Promise<{ p50: number; p95: number; max: number }> {
	const stream = openPcmStream(cfg.source);
	const rms: number[] = [];
	let pending = Buffer.alloc(0);
	const deadline = Date.now() + NOISE_MEASURE_MS;
	try {
		while (Date.now() < deadline && rms.length < NOISE_MEASURE_MS / FRAME_MS) {
			const chunk = await stream.next(STREAM_STALL_TIMEOUT_MS);
			if (chunk === null) break;
			pending = Buffer.concat([pending, chunk]);
			while (pending.length >= FRAME_BYTES) {
				const frame = pending.subarray(0, FRAME_BYTES);
				pending = pending.subarray(FRAME_BYTES);
				rms.push(rmsOf(frame));
			}
		}
	} finally {
		stream.close();
	}
	if (rms.length < NOISE_MIN_FRAMES) throw new Error(`no audio from PulseAudio source "${cfg.source}" (is the microphone connected?)`);
	rms.sort((a, b) => a - b);
	const q = (p: number) => rms[Math.min(rms.length - 1, Math.floor(rms.length * p))];
	return { p50: q(0.5), p95: q(0.95), max: q(1) };
}

const CONFIG_KEYS: Record<keyof VadConfig, (v: string) => number | string | boolean> = {
	threshold: (v) => parseNum(v, "threshold"),
	silenceMs: (v) => parseNum(v, "silenceMs"),
	maxSeconds: (v) => parseNum(v, "maxSeconds"),
	leadInMs: (v) => parseNum(v, "leadInMs"),
	source: (v) => {
		if (v.length === 0 || v.length >= 100 || v.includes("\0")) throw new Error("source must be a non-empty Pulse source name");
		return v;
	},
	model: (v) => {
		if (!Object.hasOwn(ASR_MODELS, v)) throw new Error(`unsupported model: ${v} (supported: ${supportedModels()})`);
		return v;
	},
	cleanTranscript: (v) => {
		if (v !== "true" && v !== "false") throw new Error("cleanTranscript must be true or false");
		return v === "true";
	},
};

let voiceBusy = false; // reject concurrent /voice invocations
let loadBusy = false;  // reject duplicate /voice load warm-ups

export default function (pi: ExtensionAPI) {
	pi.registerCommand("voice", {
		description: "Dictate a prompt via microphone. /voice help shows all options (VAD / fixed / flags / config / presets)",
		handler: async (rawArgs, ctx) => {
			// UI may be absent in headless/print mode; never touch ctx.ui directly.
			// In headless mode non-transcript messages go to stderr so stdout
			// stays transcript-only for scripting.
			const notify: Notify = (msg, type) => {
				if (ctx.hasUI) ctx.ui.notify(msg, type);
				else console.error(`[voice:${type}] ${msg}`);
			};
			const setStatus = (s: string | undefined) => {
				if (ctx.hasUI) ctx.ui.setStatus("voice", s);
			};
			// Background warm-up uses its own status slot so it never overwrites
			// (or clears) an in-progress recording's status.
			const setLoadStatus = (s: string | undefined) => {
				if (ctx.hasUI) ctx.ui.setStatus("voice-load", s);
			};
			const args = rawArgs.trim().split(/\s+/).filter(Boolean);

			// --- /voice load: warm up the ASR model in the background (non-blocking) ---
			if (args[0] === "load") {
				if (loadBusy) {
					notify("/voice load is already in progress", "warning");
					return;
				}
				const cfg = loadConfig();
				if (ASR_MODELS[cfg.model].protocol === "local") {
					const engine = ASR_MODELS[cfg.model].engine!;
					notify(`Starting ASR daemon for ${cfg.model} in the background...`, "info");
					void ensureSocketDaemon(engine)
					.then((ok) => notify(ok
						? `${cfg.model} daemon is ready (warm ~0.1s per call)`
						: `Could not start ASR daemon for ${cfg.model}`, ok ? "info" : "warning"))
					.catch((e) => notify(`ASR daemon start failed: ${errMsg(e)}`, "warning"));
					return;
				}
				loadBusy = true;
				setLoadStatus(`Loading ${cfg.model}...`);
				notify(`Loading ${cfg.model} in the background; you will be notified when it is ready`, "info");
				void ensureModelLoaded(cfg.model, (s) => setLoadStatus(s))
					.then(() => notify(`${cfg.model} is ready`, "info"))
					.catch((e) => notify(`Load failed: ${errMsg(e)}`, "warning"))
					.finally(() => { loadBusy = false; setLoadStatus(undefined); });
				return;
			}

			// --- /voice help: full usage summary ---
			if (args[0] === "help") {
				notify(usageText(), "info");
				return;
			}

			// Recording/transcription is exclusive; config subcommands are not.
			if (args[0] !== "config") {
				await startVoice(args, ctx, notify, setStatus);
				return;
			}

			// --- /voice config [key value | noise | reset] ---
			if (args[0] === "config") {
				const cfg = loadConfig();
				const sub = args[1];
				if (!sub) {
					notify(formatConfig(cfg), "info");
					return;
				}
				if (sub === "reset") {
					saveConfig({ ...VAD_DEFAULTS });
					notify("Voice config reset to defaults", "info");
					return;
				}
				if (sub === "preset") {
					const name = args[2];
					if (!name) {
						const list = Object.entries(PRESETS).map(([k, v]) => `${k}: threshold=${v.threshold}, silenceMs=${v.silenceMs}`).join("\n  ");
						notify(`Presets:\n  ${list}`, "info");
						return;
					}
					if (!Object.hasOwn(PRESETS, name)) {
						notify(`Unknown preset: ${name}. Valid: ${Object.keys(PRESETS).join(", ")}`, "warning");
						return;
					}
					Object.assign(cfg, PRESETS[name]);
					saveConfig(cfg);
					notify(`Preset "${name}" applied: threshold=${cfg.threshold}, silenceMs=${cfg.silenceMs}`, "info");
					return;
				}
				if (sub === "noise") {
					// Noise measurement opens its own ffmpeg capture, so it must not run
					// concurrently with a recording.
					if (voiceBusy) {
						notify("/voice is busy, please wait for it to finish", "warning");
						return;
					}
					voiceBusy = true;
					setStatus("Measuring ambient noise (3s, stay quiet)...");
					try {
						const m = await measureNoise(cfg);
						const [tMin, tMax] = NUM_LIMITS.threshold;
						const suggested = Math.min(tMax, Math.max(tMin, Math.round(m.p95 * 2)));
						// A high p95 means the 3s window was not quiet, which inflates the
						// suggestion; say so instead of silently handing over a bad value.
						const noisyWindow = m.p95 > 400;
						notify(
							`Noise: p50=${m.p50.toFixed(0)} p95=${m.p95.toFixed(0)} max=${m.max.toFixed(0)}\n` +
							`Current threshold: ${cfg.threshold}\n` +
							`Suggested: ${suggested}  ->  /voice config threshold ${suggested}` +
							(noisyWindow ? `\nNote: the window was not quiet (p95=${m.p95.toFixed(0)}); the suggestion may be too high -- re-run in silence.` : ""),
							noisyWindow ? "warning" : "info",
						);
					} catch (e) {
						notify(`Noise measurement failed: ${errMsg(e)}`, "error");
					} finally {
						voiceBusy = false;
						setStatus(undefined);
					}
					return;
				}
				// /voice config <key> <value>
				const key = sub as keyof VadConfig;
				if (!Object.hasOwn(CONFIG_KEYS, key)) {
					notify(`Unknown key: ${sub}. Valid: ${Object.keys(CONFIG_KEYS).join(", ")}`, "warning");
					return;
				}
				const parse = CONFIG_KEYS[key];
				const value = args[2];
				if (value === undefined) {
					notify(`Usage: /voice config ${key} <value>  (current: ${cfg[key]})`, "warning");
					return;
				}
				try {
					cfg[key] = parse(value);
					saveConfig(cfg);
					notify(`${key} = ${cfg[key]}  (saved to ${CONFIG_PATH})`, "info");
				} catch (e) {
					notify(errMsg(e), "warning");
				}
			}
		},
	});

	// ctrl+alt+v = "voice". ctrl+alt+<letter> is chosen over ctrl+shift+<letter>
	// because in legacy (non-kitty) terminals ctrl+shift+X sends the same byte
	// as ctrl+X (would collide with ctrl+l, the model selector), while
	// ctrl+alt+X sends ESC + control-char and stays distinct in both legacy
	// and kitty keyboard modes.
	pi.registerShortcut("ctrl+alt+v", {
		description: "Voice dictation (/voice help for options)",
		handler: async (ctx) => {
			// Shortcuts only dispatch from the TUI (hasUI is always true there),
			// but guard like the command handler for consistency.
			const notify: Notify = (msg, type) => {
				if (ctx.hasUI) ctx.ui.notify(msg, type);
				else console.error(`[voice:${type}] ${msg}`);
			};
			const setStatus: SetStatus = (s) => {
				if (ctx.hasUI) ctx.ui.setStatus("voice", s);
			};
			await startVoice([], ctx, notify, setStatus);
		},
	});
}

type Notify = (msg: string, type: "info" | "warning" | "error") => void;
type SetStatus = (s: string | undefined) => void;

/** Whether the local (offline) ASR project is present and usable. */
function localAsrAvailable(): boolean {
	return existsSync(join(LOCAL_ASR_DIR, ".venv", "bin", "python")) && existsSync(join(LOCAL_ASR_DIR, "asr_file.py"));
}

// --- Resident ASR daemon client ---
// The extension is a STATELESS socket client: it connects per request to a
// shared, self-managing daemon (idle-exits on its own), so no per-session
// teardown is needed. connect-or-spawn: if no daemon is reachable, start one.
const ASR_SOCKET = process.env.LOCAL_ASR_SOCK ?? "/tmp/local_asr.sock";
const ASR_SPAWN_WAIT_MS = 30_000; // max time to wait for a spawned daemon to be ready

/** One NDJSON request over a fresh AF_UNIX connection (per-request connect). */
function socketRequest(line: string, timeoutMs: number): Promise<Record<string, unknown>> {
	return new Promise((resolve, reject) => {
		const sock = net.connect(ASR_SOCKET);
		let buf = "";
		const timer = setTimeout(() => {
			sock.destroy();
			reject(new Error(`ASR socket request timed out after ${timeoutMs}ms`));
		}, timeoutMs);
		sock.on("connect", () => sock.write(line + "\n"));
		sock.on("data", (d) => {
			buf += d.toString("utf8");
			const nl = buf.indexOf("\n");
			if (nl === -1) return;
			clearTimeout(timer);
			sock.end();
			sock.destroy();
			try {
				resolve(JSON.parse(buf.slice(0, nl)));
			} catch (e) {
				reject(new Error(`ASR daemon returned invalid JSON: ${buf.slice(0, nl)}`));
			}
		});
		sock.on("error", (e) => {
			clearTimeout(timer);
			reject(e);
		});
	});
}

/** True if a live ASR daemon is listening on the socket right now. */
async function socketAlive(): Promise<boolean> {
	try {
		await socketRequest("PING", 2000);
		return true;
	} catch {
		return false;
	}
}

/**
 * Ensure a daemon is reachable; spawn one (detached) if not, then wait for it.
 * Used both by prewarm (at recording start) and as the first step of a
 * transcription. Never throws — returns false if it can't start.
 */
async function ensureSocketDaemon(engine: string): Promise<boolean> {
	if (await socketAlive()) return true;
	if (!localAsrAvailable()) return false;
	const py = join(LOCAL_ASR_DIR, ".venv", "bin", "python");
	const script = join(LOCAL_ASR_DIR, "asr_file.py");
	try {
		// Double-forked inside asr_file.py; spawn is fire-and-forget.
		spawn(py, [script, "--socket", ASR_SOCKET, "--model", engine], {
			detached: true, stdio: "ignore", cwd: LOCAL_ASR_DIR,
		});
	} catch {
		return false;
	}
	// Poll until the daemon answers (model load ~0.9-1.4s, masked by speech).
	const deadline = Date.now() + ASR_SPAWN_WAIT_MS;
	while (Date.now() < deadline) {
		if (await socketAlive()) return true;
		await new Promise((r) => setTimeout(r, 300));
	}
	return false;
}

/** Fire-and-forget prewarm: start/wake the daemon as soon as recording begins. */
function prewarmLocal(engine: string): void {
	void ensureSocketDaemon(engine).catch(() => { /* prewarm is best-effort */ });
}

/** Transcribe via the resident daemon; returns text ("" if none). */
async function transcribeLocal(wavPath: string, engine: string): Promise<string> {
	if (!(await ensureSocketDaemon(engine))) {
		throw new Error(`ASR daemon not reachable at ${ASR_SOCKET} (local ASR at ${LOCAL_ASR_DIR})`);
	}
	// Switch model if the daemon is running a different one (no respawn).
	const st = await socketRequest("STATUS", 5000);
	if (st.ok && st.model !== engine) {
		const sw = await socketRequest(`MODEL:${engine}`, 60_000);
		if (!sw.ok) throw new Error(`ASR model switch failed: ${String(sw.error)}`);
	}
	const resp = await socketRequest(wavPath, 120_000);
	if (!resp.ok) throw new Error(`ASR daemon error: ${String(resp.error)}`);
	return String(resp.text ?? "").trim();
}

function usageText(): string {
	const lines = [
		"/voice — voice dictation (shortcut: ctrl+alt+v)",
		"  /voice                  VAD mode: starts on speech, stops after ~1s silence",
		"  /voice 10               fixed mode: record exactly 10s (max 60)",
		"  /voice -t 400           one-shot VAD threshold override",
		"  /voice -m <model>       one-shot ASR model override",
		`  models: ${supportedModels()}`,
	];
	if (!localAsrAvailable()) {
		lines.push(`  ! local engines unavailable: ${LOCAL_ASR_DIR} not found (set LOCAL_ASR_DIR to override)`);
	}
	lines.push(
		"  /voice load             warm up the ASR model in the background",
		"  /voice config           show effective config",
		"  /voice config <key> <value>   threshold | silenceMs | maxSeconds | leadInMs | source | model | cleanTranscript",
		"  /voice config preset <name>   headset | normal | noisy",
		"  /voice config noise     measure ambient noise, suggest a threshold",
		"  /voice config reset     restore defaults",
	);
	return lines.join("\n");
}

/** Shared recording entry: busy guard + runVoice. Used by /voice and the shortcut. */
async function startVoice(
	args: string[],
	ctx: { hasUI: boolean; ui: { pasteToEditor: (t: string) => void } },
	notify: Notify,
	setStatus: SetStatus,
): Promise<void> {
	if (voiceBusy) {
		notify("/voice is already running, please wait for it to finish", "warning");
		return;
	}
	voiceBusy = true;
	try {
		await runVoice(args, ctx, notify, setStatus);
	} finally {
		voiceBusy = false;
	}
}

/** Recording + transcription path (excluded from the config subcommand). */
async function runVoice(
	args: string[],
	ctx: { hasUI: boolean; ui: { pasteToEditor: (t: string) => void } },
	notify: Notify,
	setStatus: SetStatus,
): Promise<void> {
	// --- /voice [-t N] [-m model] [seconds] ---
	let thresholdOverride: number | undefined;
	let modelOverride: string | undefined;
	const tIdx = args.indexOf("-t");
	if (tIdx !== -1) {
		thresholdOverride = Number(args[tIdx + 1]);
		const [tMin, tMax] = NUM_LIMITS.threshold;
		if (!Number.isFinite(thresholdOverride) || thresholdOverride < tMin || thresholdOverride > tMax) {
			notify(`Usage: /voice -t <threshold ${tMin}..${tMax}>`, "warning");
			return;
		}
		args.splice(tIdx, 2);
	}
	const mIdx = args.indexOf("-m");
	if (mIdx !== -1) {
		modelOverride = args[mIdx + 1];
		if (!modelOverride || !Object.hasOwn(ASR_MODELS, modelOverride)) {
			notify(`Usage: /voice -m <model>  (supported: ${supportedModels()})`, "warning");
			return;
		}
		args.splice(mIdx, 2);
	}

	const cfg = loadConfig();
	if (thresholdOverride !== undefined) cfg.threshold = thresholdOverride;
	if (modelOverride !== undefined) cfg.model = modelOverride;

	// A bare positive integer selects fixed mode; anything else is VAD mode.
	const fixedSec = args.length === 1 && /^[1-9]\d*$/.test(args[0]) ? Math.min(MAX_SECONDS, parseInt(args[0], 10)) : null;
	if (args.length > 0 && fixedSec === null) {
		notify(`Unrecognised argument: ${args[0]}. Usage: /voice [seconds] [-t N] [-m model]`, "warning");
		return;
	}

	// Speech-overlap pre-warm: start/wake the resident ASR daemon NOW, while the
	// user is still speaking, so the cold model load is hidden behind the speech.
	// Best-effort, non-blocking.
	if (ASR_MODELS[cfg.model].protocol === "local") {
		prewarmLocal(ASR_MODELS[cfg.model].engine!);
	}

	// Per-run dir on /dev/shm (bounded tmpfs): zero SSD I/O, hard size cap,
	// auto-cleaned on reboot. Falls back to /tmp if /dev/shm is unavailable.
	const base = existsSync("/dev/shm") ? "/dev/shm" : tmpdir();
	const runDir = mkdtempSync(join(base, "pi-voice-"), { mode: 0o700 });
	const tmp = join(runDir, "recording.wav");

	try {
		if (fixedSec !== null) {
			setStatus(`Recording ${fixedSec}s...`);
			await execFileP(
				"ffmpeg",
				["-y", "-loglevel", "error", "-f", "pulse", "-i", cfg.source,
				 "-t", String(fixedSec), "-ar", String(SAMPLE_RATE), "-ac", "1", tmp],
				{ timeout: (fixedSec + 10) * 1000 },
			);
			// Fixed mode has no VAD: skip ASR on silent recordings to avoid
			// whisper hallucinating text out of noise.
			const stats = wavStats(tmp);
			// wavStats returns null for an empty/zero-sample WAV, which means no
			// audio at all -- treat that as "no speech" too, not as a pass.
			if (!stats || stats.peak < cfg.threshold) {
				notify(
					`No speech in the ${fixedSec}s recording` +
					(stats ? ` (peak ${stats.peak} < threshold ${cfg.threshold})` : " (no audio captured)") +
					`. Speak closer/louder, or lower the threshold: /voice config threshold <n>`,
					"warning",
				);
				return;
			}
		} else {
			setStatus(`Waiting for microphone (max ${cfg.maxSeconds}s, threshold ${cfg.threshold})...`);
			const pcm = await recordWithVad(cfg, (s) => setStatus(s));
			if (pcm === null) {
				notify(
					`No speech detected (threshold ${cfg.threshold}). ` +
					`If you were speaking, lower it: /voice config noise then /voice config threshold <n>`,
					"warning",
				);
				return;
			}
			writeWav(tmp, pcm);
		}

		setStatus(`Waiting for ${cfg.model} to be ready...`);
		// Local engines load in ~1s inside asr_file.py; no server warm-up needed.
		if (ASR_MODELS[cfg.model].protocol !== "local") {
			await ensureModelLoaded(cfg.model, (s) => setStatus(s));
		}

		setStatus("Transcribing...");
		const text = await transcribe(tmp, cfg.model, cfg.cleanTranscript);

		if (!text) {
			// Qwen3-ASR returns empty ("language None<asr_text>") when it does not
			// detect speech; whisper usually returns something anyway.
			// Diagnose the captured audio so the user knows what was recorded.
			let diag = "";
			try {
				const raw = readFileSync(tmp);
				const stats = wavStats(tmp);
				if (stats) {
					mkdirSync(DEBUG_DIR, { recursive: true });
					writeFileSync(join(DEBUG_DIR, "last.wav"), raw, { mode: 0o600 });
					diag = ` (recorded ${stats.seconds.toFixed(1)}s, avg RMS ${stats.rms.toFixed(0)}, peak ${stats.peak} vs threshold ${cfg.threshold}; saved to ${join(DEBUG_DIR, "last.wav")})`;
				}
			} catch { /* ignore */ }
			notify(
				`No speech recognized by ${cfg.model}.${diag} ` +
				`If you were speaking, your voice may be too quiet/far -- try: /voice -t ${Math.max(50, Math.round(cfg.threshold / 2))}`,
				"warning",
			);
			return;
		}
		// Interactive mode: paste into the editor. In headless/print mode
		// pasteToEditor is a no-op, so fall back to stdout for scripting.
		if (ctx.hasUI) {
			ctx.ui.pasteToEditor(text);
			notify(`Transcribed ${text.length} chars, edit and send`, "info");
		} else {
			console.log(text);
		}
	} catch (e) {
		notify(`Recording/transcription failed: ${errMsg(e)}`, "error");
	} finally {
		rmSync(runDir, { recursive: true, force: true });
		setStatus(undefined);
	}
}

/** RMS/peak/seconds of a WAV's data chunk, or null if unreadable. */
function wavStats(wavPath: string): { rms: number; peak: number; seconds: number } | null {
	try {
		const pcm = wavDataChunk(readFileSync(wavPath));
		const copy = Buffer.from(pcm);
		const samples = new Int16Array(copy.buffer, copy.byteOffset, copy.length / 2);
		if (samples.length === 0) return null;
		let sum = 0, peak = 0;
		for (let i = 0; i < samples.length; i++) {
			sum += samples[i] * samples[i];
			const a = Math.abs(samples[i]);
			if (a > peak) peak = a;
		}
		return { rms: Math.sqrt(sum / samples.length), peak, seconds: samples.length / SAMPLE_RATE };
	} catch {
		return null;
	}
}
