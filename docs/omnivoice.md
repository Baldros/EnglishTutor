# OmniVoice no EnglishTutor

## Objetivo

Este documento explica:

- o que e o `OmniVoice`
- de onde ele vem
- onde baixar codigo e pesos
- como rodar tudo localmente
- qual e a melhor forma de integrar isso no `EnglishTutor`

## O que e o OmniVoice

`OmniVoice` e um modelo de TTS zero-shot multilingue mantido por `k2-fsa`. Ele foi publicado como:

- repositorio de codigo no GitHub
- repositorio de pesos no Hugging Face

Links principais:

- GitHub: `https://github.com/k2-fsa/OmniVoice`
- Modelo: `https://huggingface.co/k2-fsa/OmniVoice`
- Space de demonstracao: `https://huggingface.co/spaces/k2-fsa/OmniVoice`
- Paper: `https://arxiv.org/abs/2604.00688`

## De onde vem cada parte

E importante separar duas coisas:

- **GitHub**: contem o codigo Python, CLI, exemplos e documentacao do projeto.
- **Hugging Face**: contem os pesos do modelo e arquivos necessarios para carregar o checkpoint.

Na pratica:

- voce baixa o **codigo** do GitHub
- voce baixa os **pesos** do Hugging Face
- depois executa tudo **localmente**, apontando para pastas locais

Isso significa que o Hugging Face pode ser usado so como origem do download inicial. Ele nao precisa participar da inferencia depois.

## O Hugging Face e obrigatorio em runtime?

Nao, desde que os pesos ja tenham sido baixados.

O `OmniVoice` aceita carregar o modelo por:

- `repo id`, por exemplo `k2-fsa/OmniVoice`
- ou por **caminho local**, por exemplo `E:\AI\Models\OmniVoice`

Para o `EnglishTutor`, o ideal e usar **somente caminho local em producao**.

## Estrutura local recomendada

Uma estrutura simples e reutilizavel no Windows:

```text
E:\AI\Models\
  OmniVoice\
  whisper-large-v3-turbo\
```

Sugestao:

- `E:\AI\Models\OmniVoice` para o TTS principal
- `E:\AI\Models\whisper-large-v3-turbo` apenas se a clonagem de voz com transcricao automatica for realmente necessaria

## Como baixar

Voce tem duas opcoes boas.

### Opcao 1: baixar pelo repositrio do Hub

Baixe os pesos do modelo do Hugging Face para uma pasta local dedicada. O importante nao e o metodo exato; o importante e que o resultado final seja uma pasta local completa contendo arquivos como:

- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `audio_tokenizer\...`

Depois disso, o carregamento deve ser feito por caminho local.

### Opcao 2: clonar o repositorio do modelo do Hugging Face

O Hub funciona como repositorio versionado. Em vez de depender do identificador remoto em runtime, voce pode materializar o modelo no disco e passar a usar apenas a pasta local.

## Modo offline recomendado

Se a intencao e evitar qualquer dependencia de rede durante execucao, use estas variaveis de ambiente no processo que sobe o TTS:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
```

Isso nao baixa nada por voce. Essas variaveis servem para garantir que, depois de tudo baixado, a aplicacao continue local e nao tente buscar arquivos na internet por tras.

## Como o EnglishTutor funciona hoje

O projeto atual tem um backend simples de TTS em [TTS.py](E:\EnglishTutor\TTS.py:1), baseado em:

```python
pipeline("text-to-speech", model="facebook/mms-tts-eng")
```

Isso e bom para prototipo porque:

- e simples
- exige pouco codigo
- integra facil com o loop atual

Mas ele tem algumas limitacoes frente ao `OmniVoice`:

- menor flexibilidade de voz
- menos controle de estilo
- menos capacidade de clonagem
- menos alinhado com um caso de tutor multilingue mais ambicioso

## Melhor forma de implementar no EnglishTutor

A melhor forma **nao** e alterar o `main.py` para conhecer detalhes de `OmniVoice`.

A melhor forma e introduzir uma camada de abstracao de TTS e manter o `main.py` consumindo apenas uma interface estavel, por exemplo:

```python
class BaseTTS:
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        ...
```

Com isso, voce pode ter dois backends:

- `RealtimeTTS` atual, baseado em `facebook/mms-tts-eng`
- `OmniVoiceTTS`, baseado em checkpoint local

### Arquitetura recomendada

Para o `EnglishTutor`, eu recomendo esta evolucao:

1. manter [TTS.py](E:\EnglishTutor\TTS.py:1) como implementacao simples ou fallback
2. criar um modulo novo, por exemplo `tts_backends/omnivoice_tts.py`
3. carregar o `OmniVoice` uma unica vez na inicializacao
4. configurar o caminho local do modelo por argumento ou variavel de ambiente
5. deixar `main.py` escolher o backend via configuracao

### Por que isso e melhor

- evita acoplamento do app com uma biblioteca especifica
- facilita testes A/B entre TTS atual e OmniVoice
- permite fallback rapido se o modelo estiver indisponivel
- facilita reaproveitar a mesma implementacao em outros projetos
- reduz risco de duplicar logica de carga de modelo

## O que nao fazer

Algumas escolhas vao piorar a manutencao:

- carregar `OmniVoice.from_pretrained("k2-fsa/OmniVoice")` diretamente dentro do loop principal
- depender de `repo id` remoto em vez de caminho local
- carregar o modelo a cada chamada de sintese
- misturar logica de download, inicializacao e inferencia na mesma classe

## Melhor estrategia para este projeto

Para o `EnglishTutor`, a estrategia mais pragmatica e:

### Fase 1

- manter o TTS atual funcionando
- documentar e preparar a integracao local do OmniVoice

### Fase 2

- adicionar `OmniVoiceTTS` como backend opcional
- usar somente modelo local no disco
- comecar sem ASR auxiliar do proprio OmniVoice

### Fase 3

- avaliar clonagem de voz e controle de estilo
- decidir se vale usar `voice design` para perfis diferentes de tutor

## Recomendacao especifica para o EnglishTutor

Como este projeto e um tutor por voz, o `OmniVoice` faz mais sentido em tres cenarios:

- dar ao tutor uma voz mais natural e consistente
- criar variacoes de perfil de tutor
- evoluir para um produto mais polido sem depender de servicos externos

Para o uso atual, eu recomendaria:

- usar `OmniVoice` no modo **TTS local padrao**
- evitar `voice cloning` na primeira integracao
- evitar a carga do Whisper auxiliar no inicio

Ou seja: primeiro resolva **texto para fala local e estavel**. Depois, se fizer sentido, voce adiciona recursos extras.

## Exemplo de desenho de configuracao

Uma configuracao simples poderia ficar assim:

```text
TTS_BACKEND=omnivoice
OMNIVOICE_MODEL_DIR=E:\AI\Models\OmniVoice
OMNIVOICE_DEVICE=cuda
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

E o `main.py` continuaria escolhendo apenas o backend, sem saber detalhes do checkpoint.

## Conclusao

Para este projeto, o `OmniVoice` deve ser tratado como um backend local de TTS de maior qualidade, e nao como uma chamada remota ao Hugging Face.

Resumo:

- o **codigo** vem do GitHub
- os **pesos** vem do Hugging Face
- a **inferencia** pode e deve rodar localmente
- o melhor desenho para o `EnglishTutor` e integrar via uma camada de TTS reutilizavel
- o melhor primeiro passo e adicionar `OmniVoice` como backend opcional, com modelo em pasta local e modo offline habilitado
