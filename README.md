# EnglishTutor

EnglishTutor e um projeto de tutor de linguas conversacional. O prototipo atual e focado em ensino de ingles por voz, com um loop local de:

- `STT` para transcrever a fala do usuario
- `LLM` para gerar a resposta do tutor
- `TTS` para sintetizar a resposta em audio

## Estado atual

Hoje o projeto usa:

- `distil-whisper/distil-large-v3` para STT
- `gemma:7b` via Ollama para LLM
- `facebook/mms-tts-eng` para TTS

O ponto de entrada principal e [main.py](E:\EnglishTutor\main.py:1). A implementacao atual de TTS fica em [TTS.py](E:\EnglishTutor\TTS.py:1).

## OmniVoice

Existe um guia dedicado para avaliar a migracao do TTS atual para OmniVoice:

- [docs/omnivoice.md](E:\EnglishTutor\docs\omnivoice.md)

Esse guia cobre:

- o que e o OmniVoice
- de onde vem o codigo e os pesos
- como baixar tudo localmente
- como evitar dependencia online em runtime
- a melhor forma de integrar no EnglishTutor

## Direcao recomendada

Para este projeto, a melhor estrategia nao e trocar o TTS atual diretamente por chamadas espalhadas ao `OmniVoice.from_pretrained(...)` em varios pontos do codigo. O caminho recomendado e:

1. introduzir uma interface de TTS estavel no projeto
2. manter o backend atual como fallback simples
3. adicionar um backend `OmniVoiceTTS` separado
4. carregar o modelo uma unica vez por processo
5. apontar o modelo para um caminho local no disco

Isso evita duplicacao de VRAM, reduz acoplamento com Hugging Face e facilita reaproveitar a mesma implementacao em outros projetos.
