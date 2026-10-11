"""Publica o trainify-ai num Space do Hugging Face (Docker).

Antes, faça login uma vez no terminal (o token fica guardado só na sua máquina):
    huggingface-cli login

Uso (na raiz do projeto):
    python -m scripts.publicar_no_hf --space SEU_USUARIO/trainify-ai

O que o script faz:
  1. cria o Space (Docker) se ainda não existir;
  2. envia os arquivos versionados no git, trocando o README pelo de
     deploy/huggingface/README.md (o Space exige um cabeçalho de configuração);
  3. grava como "secrets" do Space a GROQ_API_KEY do seu .env e a
     INTERNAL_API_KEY (a mesma que vai no AI_INTERNAL_KEY do Trainify).

Os secrets nunca são impressos nem gravados em arquivo.
"""

import argparse
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi

RAIZ = Path(__file__).resolve().parent.parent


def arquivos_versionados() -> list[str]:
    saida = subprocess.run(["git", "ls-files"], cwd=RAIZ, capture_output=True, text=True, check=True)
    return [linha for linha in saida.stdout.splitlines() if linha]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", required=True, help="usuario/nome do Space, ex.: luanbandeira/trainify-ai")
    parser.add_argument("--privado", action="store_true", help="cria o Space como privado")
    parser.add_argument("--chave-interna", default=os.getenv("INTERNAL_API_KEY"),
                        help="INTERNAL_API_KEY a usar (padrão: gera uma nova e grava em .chave-interna-hf)")
    args = parser.parse_args()

    api = HfApi()
    try:
        usuario = api.whoami()["name"]
    except Exception:
        print("Faça login antes: huggingface-cli login")
        return 1
    print(f"Logado no Hugging Face como {usuario}")

    env = dotenv_values(RAIZ / ".env")
    groq = env.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not groq:
        print("GROQ_API_KEY não encontrada no .env")
        return 1

    # A chave interna precisa ser a mesma nos dois lados. Se não foi informada,
    # geramos uma e guardamos num arquivo ignorado pelo git, para configurar no Render.
    arquivo_chave = RAIZ / ".chave-interna-hf"
    chave = args.chave_interna or (arquivo_chave.read_text().strip() if arquivo_chave.exists() else None)
    if not chave:
        chave = secrets.token_hex(24)
        arquivo_chave.write_text(chave + "\n", encoding="utf-8")
        print(f"Chave interna nova gravada em {arquivo_chave.name} (use em AI_INTERNAL_KEY no Render)")

    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.privado, exist_ok=True)

    with tempfile.TemporaryDirectory() as pasta:
        destino = Path(pasta)
        for arquivo in arquivos_versionados():
            origem = RAIZ / arquivo
            if origem.is_file():
                (destino / arquivo).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(origem, destino / arquivo)
        shutil.copy2(RAIZ / "deploy" / "huggingface" / "README.md", destino / "README.md")

        print("Enviando os arquivos...")
        api.upload_folder(folder_path=str(destino), repo_id=args.space, repo_type="space",
                          commit_message="Atualiza o trainify-ai")

    api.add_space_secret(args.space, "GROQ_API_KEY", groq)
    api.add_space_secret(args.space, "INTERNAL_API_KEY", chave)

    dono, nome = args.space.split("/")
    url = f"https://{dono}-{nome}.hf.space".lower().replace("_", "-")
    print(f"\nPronto. O build leva alguns minutos: https://huggingface.co/spaces/{args.space}")
    print(f"Endereço da API (AI_SERVICE_URL no Render): {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
