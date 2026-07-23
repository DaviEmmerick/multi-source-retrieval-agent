# Guia de Uso - RAG Engine

## Pré-requisitos

Antes de iniciar, você precisa de:

1. **Python 3.9+** instalado
2. **Docker e Docker Compose** (para Neo4j)
3. **Ollama** instalado e rodando com o modelo Qwen 2.5 Coder 3B

### Verificar Ollama
```bash
# Ollama deve estar rodando na porta 11434
curl http://localhost:11434/api/tags

# Se não tiver o modelo, baixe:
ollama pull qwen2.5-coder:3b
```

---

## Passo 1: Iniciar Neo4j com Docker Compose

Sim, você precisa rodar o Docker Compose para iniciar o Neo4j:

```bash
# Na pasta raiz do projeto
docker-compose up -d
```

Isso vai:
- ✅ Iniciar o Neo4j na porta 7474 (interface web) e 7687 (driver)
- ✅ Usar credenciais padrão: `neo4j` / `password123`
- ✅ Criar volume para persistir dados em `./neo4j_data`

**Verificar se está rodando:**
```bash
# Acesse a interface web
http://localhost:7474
# Login: neo4j / password123
```

---

## 🔧 Passo 2: Instalar Dependências Python

```bash
# Ativar ambiente virtual (se tiver)
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

---

## ▶️ Passo 3: Executar a Aplicação

```bash
# A partir da pasta raiz
streamlit run app/interface.py
```

Isso vai:
- ✅ Iniciar o servidor Streamlit na porta 8501
- ✅ Abrir automaticamente em `http://localhost:8501`
- ✅ Carregar o dataset de exemplo no Neo4j

**Ou com configurações customizadas via variáveis de ambiente:**

```bash
# Customizar modelo LLM
export LLM_MODEL="qwen2.5-coder:3b"
export LLM_TEMPERATURE=0.1
export LLM_BASE_URL="http://localhost:11434"

# Customizar Neo4j
export NEO4J_URL="neo4j://127.0.0.1:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password123"

# Customizar FAISS
export FAISS_MODEL="all-MiniLM-L6-v2"
export FAISS_SIMILARITY_THRESHOLD=0.3

# Ativar logs de debug
export ENABLE_DEBUG_LOGGING="true"

# Então rodar
streamlit run app/interface.py
```

---

## Como Usar a Interface

### Fluxo Completo:

#### 1️⃣ **Upload de Documentos**
- Clique em "Arraste ou selecione um PDF técnico"
- Escolha um arquivo PDF
- Clique em "Finalizar e Indexar Documento"
- ⏳ Aguarde: o sistema vai:
  - Dividir o PDF em chunks (pedaços de texto)
  - Indexar no FAISS (busca semântica)
  - Extrair entidades e relacionamentos
  - Injetar o grafo no Neo4j

#### 2️⃣ **Ver Chunks Gerados**
- Expanda a seção "Visualizar Chunks Gerados"
- Veja todos os trechos de texto criados

#### 3️⃣ **Fazer Perguntas**
- Na seção "Faça uma Pergunta"
- Digite sua pergunta sobre o documento
- Exemplo: "Quais são as principais entidades?"
- Clique em "Processar Pergunta"

#### 4️⃣ **Ver Resultados**
- **Documentos Recuperados**: Contexto encontrado no grafo + FAISS
- **Resposta Gerada**: A resposta do LLM baseada no contexto
- ℹ️ **Iterações**: Quantas vezes a query foi reescrita

---

## 🔄 Como Funciona o Sistema

```
┌─────────────────────────────────────────────────────┐
│ USUÁRIO FAZ PERGUNTA                                │
└────────────────┬────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────┐
│ BUSCA HÍBRIDA                                       │
│ • Neo4j (grafo estruturado)                         │
│ • FAISS (busca semântica)                           │
└────────────────┬────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────┐
│ AVALIAR RELEVÂNCIA DOS DOCUMENTOS                   │
│ (Similaridade semântica)                            │
└────────────┬───────────────────┬────────────────────┘
             │                   │
        SIM (≥0.3)          NÃO (<0.3)
             │                   │
             ↓                   ↓
     ┌──────────────┐  ┌─────────────────┐
     │ GERAR        │  │ REESCREVER      │
     │ RESPOSTA     │  │ QUERY com LLM   │
     │ com LLM      │  └────────┬────────┘
     └──────┬───────┘           │
            ↓                   ↓ (até 3x)
     ┌──────────────┐   (retry)
     │ RESPOSTA     │
     │ FINAL        │
     └──────────────┘
```

---

## Troubleshooting

### "Connection refused" para Neo4j
```bash
# Verificar se o container está rodando
docker ps | grep neo4j

# Se não estiver, reiniciar
docker-compose restart neo4j

# Ver logs
docker-compose logs neo4j
```

### "Could not connect to Ollama"
```bash
# Verificar se Ollama está rodando
curl http://localhost:11434/api/tags

# Se não estiver, inicie (depende do SO)
# macOS: brew services start ollama
# Linux: sudo systemctl start ollama
# Windows: Execute o programa Ollama
```

### "Erro ao processar PDF"
- Verifique se o PDF é válido
- Tente com um PDF menor primeiro
- Veja os logs da aplicação para mais detalhes

---

## Estrutura do Projeto

```
agentic-knowledge-engine/
├── app/
│   ├── interface.py          # UI Streamlit
│   ├── graph.py              # Workflow LangGraph
│   ├── setup_neo4j.py        # Configuração Neo4j
│   ├── config.py             # Configurações centralizadas 
│   ├── logger.py             # Sistema de logging 
├── docker-compose.yml        # Neo4j setup
├── requirements.txt          # Dependências Python
├── Dockerfile                # Para containerizar
└── fluxo.png                 # Diagrama da arquitetura
```

---

## Variáveis de Configuração

Todas em `app/config.py`:

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `LLM_MODEL` | qwen2.5-coder:3b | Modelo Ollama a usar |
| `LLM_TEMPERATURE` | 0.1 | Criatividade do LLM (0-1) |
| `LLM_BASE_URL` | http://localhost:11434 | URL do Ollama |
| `NEO4J_URL` | neo4j://127.0.0.1:7687 | URL do Neo4j |
| `FAISS_MODEL` | all-MiniLM-L6-v2 | Modelo embedding |
| `FAISS_SIMILARITY_THRESHOLD` | 0.3 | Threshold de relevância |
| `MAX_REWRITE_ITERATIONS` | 3 | Max tentativas de reescrita |
| `ENABLE_DEBUG_LOGGING` | false | Logs detalhados |

---

## Melhorias Implementadas

Este guia reflete a seguinte stack:

- ✅ **Query Interface**: Seção de perguntas na UI
- ✅ **LLM-based Rewriting**: Query é reescrita por IA se não trovados resultados bons
- ✅ **Semantic Grading**: Documentos avaliados por similaridade do cosseno

---

## Próximos Passos (Opcional)

1. **Testar com seu próprio PDF** (não finanças.pdf)
2. **Ajustar thresholds** conforme resultados
3. **Adicionar mais documentos** para melhor grafo
4. **Monitoring** com logs em `ENABLE_DEBUG_LOGGING=true`

---