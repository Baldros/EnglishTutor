import argparse
from datetime import datetime
from time import perf_counter

import numpy as np
import sounddevice as sd
from rich.console import Console

from LLM import TutorLLM
from STT import ContinuousSTT
from TTS import RealtimeTTS

console = Console()


def log_stage(stage: str, message: str, style: str = "cyan") -> None:
    now = datetime.now().strftime("%H:%M:%S")
    console.print(f"[{now}] [{stage}] {message}", style=style)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="English Tutor voice loop")
    parser.add_argument("--stt-model", default="distil-whisper/distil-large-v3")
    parser.add_argument("--llm-model", default="gemma:7b")
    parser.add_argument("--tts-model", default="facebook/mms-tts-eng")
    parser.add_argument("--language", default="en")
    parser.add_argument("--speech-threshold", type=float, default=0.015)
    parser.add_argument("--silence-duration", type=float, default=0.8)
    parser.add_argument(
        "--input-device",
        type=int,
        default=3,
        help="Indice do dispositivo de microfone.",
    )
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="Lista os dispositivos de audio e encerra.",
    )
    return parser.parse_args()


def list_input_devices() -> list[tuple[int, dict]]:
    devices = sd.query_devices()
    result: list[tuple[int, dict]] = []
    for idx, dev in enumerate(devices):
        if int(dev.get("max_input_channels", 0)) > 0:
            result.append((idx, dev))
    return result


def pick_input_device(explicit_index: int | None) -> int | None:
    if explicit_index is not None:
        dev = sd.query_devices(explicit_index)
        if int(dev.get("max_input_channels", 0)) <= 0:
            raise ValueError(
                f"Dispositivo idx={explicit_index} nao possui canal de entrada."
            )
        return explicit_index

    # Prefer a webcam/camera-like microphone name when available.
    webcam_tokens = ("webcam", "camera", "usb", "pnp")
    for idx, dev in list_input_devices():
        name = str(dev.get("name", "")).lower()
        if any(token in name for token in webcam_tokens):
            return idx
    return None


def print_input_devices() -> None:
    devices = list_input_devices()
    if not devices:
        log_stage("AUDIO", "Nenhum dispositivo de entrada encontrado.", "red")
        return
    log_stage("AUDIO", "Dispositivos de entrada detectados:", "bold cyan")
    for idx, dev in devices:
        name = dev.get("name", "Unknown")
        channels = dev.get("max_input_channels", 0)
        hostapi = dev.get("hostapi", "?")
        default_sr = dev.get("default_samplerate", "?")
        console.print(
            f"  - idx={idx} | ch={channels} | sr={default_sr} | hostapi={hostapi} | {name}"
        )


def main() -> None:
    args = parse_args()
    if args.list_audio_devices:
        print_input_devices()
        return

    try:
        selected_input_device = pick_input_device(args.input_device)
    except Exception as exc:
        log_stage("AUDIO", f"Erro ao selecionar microfone: {exc}", "red")
        log_stage("AUDIO", "Use --list-audio-devices para ver indices validos.", "red")
        return
    if selected_input_device is None:
        log_stage(
            "AUDIO",
            "Usando microfone de entrada padrao do sistema.",
            "yellow",
        )
    else:
        dev = sd.query_devices(selected_input_device)
        log_stage(
            "AUDIO",
            f"Usando microfone idx={selected_input_device}: {dev.get('name', 'Unknown')}",
            "yellow",
        )

    log_stage("INIT", "Inicializando STT...", "yellow")
    start = perf_counter()
    stt = ContinuousSTT(
        model_name=args.stt_model,
        speech_threshold=args.speech_threshold,
        silence_duration=args.silence_duration,
        input_device=selected_input_device,
    )
    log_stage("INIT", f"STT pronto em {perf_counter() - start:.2f}s", "green")

    log_stage("INIT", "Inicializando LLM...", "yellow")
    start = perf_counter()
    llm = TutorLLM(model_name=args.llm_model)
    log_stage("INIT", f"LLM pronta em {perf_counter() - start:.2f}s", "green")

    log_stage("INIT", "Inicializando TTS...", "yellow")
    start = perf_counter()
    tts = RealtimeTTS(model_name=args.tts_model)
    log_stage("INIT", f"TTS pronto em {perf_counter() - start:.2f}s", "green")

    log_stage("SYSTEM", "English Tutor em execucao.", "bold green")
    log_stage(
        "SYSTEM",
        "Fale naturalmente. A resposta vem apos detectar pausa de fala.",
        "bold green",
    )
    log_stage("SYSTEM", "Diga 'exit', 'quit' ou 'stop' para encerrar.", "bold green")

    stt.start()
    log_stage("LISTEN", "Microfone ativo e aguardando fala...", "bold cyan")
    try:
        while True:
            audio = stt.get_next_audio(timeout=0.1)
            if audio is None:
                continue

            duration = len(audio) / float(stt.sample_rate)
            log_stage(
                "CAPTURE",
                f"Audio capturado ({duration:.2f}s). Iniciando transcricao STT...",
                "magenta",
            )

            start = perf_counter()
            with console.status("[bold yellow]Processando audio no STT...[/bold yellow]"):
                user_text = stt.transcribe(audio, language=args.language)
            stt_time = perf_counter() - start

            if not user_text:
                log_stage(
                    "STT",
                    f"Transcricao vazia em {stt_time:.2f}s. Voltando para escuta.",
                    "red",
                )
                continue

            log_stage("STT", f"Texto reconhecido em {stt_time:.2f}s", "green")
            console.print(f"You: [bold white]{user_text}[/bold white]")

            if user_text.lower().strip() in {"exit", "quit", "stop"}:
                log_stage("SYSTEM", "Comando de encerramento detectado.", "yellow")
                stt.pause()
                log_stage("TTS", "Gerando audio final...", "yellow")
                audio_out, sr = tts.synthesize("Goodbye.")
                log_stage("AUDIO", "Reproduzindo audio final...", "yellow")
                tts.play_audio(audio_out, sr)
                break

            log_stage("LLM", "Enviando texto para LLM...", "blue")
            start = perf_counter()
            with console.status("[bold blue]Gerando resposta da LLM...[/bold blue]"):
                reply = llm.respond(user_text)
            llm_time = perf_counter() - start

            log_stage("LLM", f"Resposta pronta em {llm_time:.2f}s", "green")
            console.print(f"Tutor: [bold green]{reply}[/bold green]")

            log_stage("TTS", "Pausando escuta para evitar eco.", "yellow")
            stt.pause()
            try:
                start = perf_counter()
                with console.status("[bold yellow]Gerando audio TTS...[/bold yellow]"):
                    audio_out, sr = tts.synthesize(reply)
                tts_time = perf_counter() - start
                samples = int(np.asarray(audio_out).shape[0])
                approx_secs = samples / float(sr)
                log_stage(
                    "TTS",
                    f"Audio sintetizado em {tts_time:.2f}s ({approx_secs:.2f}s de fala).",
                    "green",
                )

                start = perf_counter()
                with console.status("[bold yellow]Reproduzindo audio no alto-falante...[/bold yellow]"):
                    tts.play_audio(audio_out, sr)
                play_time = perf_counter() - start
                log_stage("AUDIO", f"Reproducao concluida em {play_time:.2f}s", "green")
            finally:
                stt.resume()
                log_stage(
                    "LISTEN",
                    "Microfone reativado. Aguardando proxima fala...",
                    "bold cyan",
                )

    except KeyboardInterrupt:
        log_stage("SYSTEM", "Interrompido por teclado (Ctrl+C).", "yellow")
    finally:
        stt.close()
        log_stage("SYSTEM", "Microfone encerrado. Sistema finalizado.", "bold yellow")


if __name__ == "__main__":
    main()
