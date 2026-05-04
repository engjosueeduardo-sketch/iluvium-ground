# 📚 Base de Conhecimento da IA Iluvium

Esta pasta alimenta o assistente **IA Iluvium** com informações técnicas
sobre aterramento e SPDA. Tudo que está aqui é indexado e fica pesquisável
no chat do app.

## Formatos aceitos

| Extensão | Descrição                                    | Recomendado? |
|----------|----------------------------------------------|--------------|
| `.md`    | Markdown — ideal para conteúdo estruturado   | ⭐ Sim, melhor |
| `.txt`   | Texto puro — simples e leve                  | ✅ Sim        |
| `.pdf`   | PDFs com texto extraível (não escaneados)    | ✅ Sim        |

## Como adicionar novos conteúdos

### Para PDFs de normas (NBR, IEEE, IEC):

1. **Coloque o PDF nesta pasta** (`conhecimento/`)
2. **Reinicie o app** Streamlit (Ctrl+C → `streamlit run main.py`)
3. Pronto — a IA já indexou e busca dentro do PDF

A IA mostra o **número da página** onde encontrou cada trecho, facilitando
a consulta da norma original.

⚠️ **PDFs escaneados não funcionam.** Se você tem um PDF que é uma imagem
escaneada de uma norma impressa, o pypdf não consegue extrair texto. Faça
OCR antes (Adobe Acrobat → "Reconhecer Texto", ou converta com ferramentas
online). O app vai te avisar na sidebar quando detectar um PDF escaneado.

⚠️ **Direitos autorais:** Você precisa ter os PDFs legalmente (ABNT, IEEE, etc).
A pasta `conhecimento/` fica **apenas no PC do usuário** — nada é enviado
para servidores. Se for distribuir o software, **não inclua os PDFs** —
deixe o usuário copiar a própria cópia legal.

### Para conteúdo próprio (notas, FAQ, manuais internos):

Crie um arquivo `.md` (Markdown) com a estrutura:

```markdown
# Título Principal do Documento

## Seção 1
Conteúdo da seção 1...

## Seção 2
Conteúdo da seção 2...

### Subseção 2.1
Texto...
```

Cada `## Seção` ou `### Subseção` vira um chunk independente na busca,
o que melhora a precisão. Quanto melhor estruturado, melhores os resultados.

## O que já vem nesta pasta (conteúdo inicial)

| Arquivo                         | Conteúdo                                  |
|---------------------------------|-------------------------------------------|
| `NBR_7117_estratificacao.md`    | Método Wenner, estratificação, Palmer    |
| `NBR_5419_SPDA.md`              | SPDA, níveis de proteção, captores       |
| `IEEE_80_malha.md`              | Malha de subestação, tensões de toque    |
| `resistividades_tipicas.md`     | Tabelas de solos brasileiros             |
| `formulas.md`                   | 16 fórmulas essenciais                   |
| `faq.md`                        | Perguntas comuns de campo                |

Você pode editar esses arquivos para adicionar/corrigir conteúdo conforme
a experiência da Iluvium for crescendo. É texto Markdown puro — qualquer
editor de texto serve (VS Code, Notepad++, até Bloco de Notas).

## Dependência adicional para PDFs

Se você for usar PDFs, instale a biblioteca:

```bash
pip install pypdf
```

Sem ela, os PDFs serão ignorados e a sidebar vai avisar.
