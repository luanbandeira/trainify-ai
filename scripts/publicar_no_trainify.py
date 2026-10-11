"""Cria o curso de NRs dentro do Trainify, pela API REST, como um instrutor faria.

Cada NR vira uma aula de texto (ARTICLE) e cada módulo ganha uma aula de
avaliação (QUIZ) vazia, para testar o "Gerar com IA". Ao salvar as aulas, o
próprio Trainify manda o conteúdo para o trainify-ai indexar: o curso passa
pelo mesmo caminho de qualquer material enviado por um instrutor.

Uso (na raiz do projeto, com o Trainify e o trainify-ai rodando):
    python -m scripts.publicar_no_trainify --url http://localhost:8080 --email instrutor@empresa.com

A senha é pedida no terminal (ou lida de TRAINIFY_PASSWORD) e não é gravada.
"""

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

import httpx

from app.config import DATA_DIR, PROCESSED_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.getenv("TRAINIFY_URL", "http://localhost:8080"),
                        help="endereço do backend do Trainify (sem /api/v1)")
    parser.add_argument("--email", default=os.getenv("TRAINIFY_EMAIL"), required=os.getenv("TRAINIFY_EMAIL") is None,
                        help="e-mail de um instrutor ou administrador")
    parser.add_argument("--publicar", action="store_true", help="publica o curso no fim (fica visível aos alunos)")
    parser.add_argument("--senha-arquivo", type=Path,
                        help="lê a senha deste arquivo e o apaga em seguida (para terminais que não "
                             "aceitam digitação em campo de senha)")
    args = parser.parse_args()

    if args.senha_arquivo:
        if not args.senha_arquivo.exists():
            print(f"Arquivo {args.senha_arquivo} não encontrado.")
            return 1
        try:
            senha = args.senha_arquivo.read_text(encoding="utf-8-sig").strip()
        finally:
            args.senha_arquivo.unlink()  # a senha não fica no disco
    else:
        senha = os.getenv("TRAINIFY_PASSWORD") or getpass.getpass(f"Senha de {args.email}: ")
    curso = json.loads((DATA_DIR / "curso_nrs.json").read_text(encoding="utf-8"))

    with httpx.Client(base_url=args.url.rstrip("/") + "/api/v1", timeout=120) as api:
        login = api.post("/auth/login", json={"email": args.email, "password": senha})
        if login.status_code != 200:
            print(f"Falha no login ({login.status_code}): {login.text[:300]}")
            return 1
        api.headers["Authorization"] = f"Bearer {login.json()['accessToken']}"

        criado = api.post("/courses", json={"title": curso["title"], "description": curso["description"]})
        criado.raise_for_status()
        course_id = criado.json()["id"]
        print(f"Curso criado: {curso['title']} ({course_id})")

        for modulo in curso["modules"]:
            m = api.post(f"/courses/{course_id}/modules", json={"title": modulo["title"], "description": ""})
            m.raise_for_status()
            module_id = m.json()["id"]
            print(f"  Módulo: {modulo['title']}")

            for licao in modulo["lessons"]:
                texto = (PROCESSED_DIR / f"{licao['lesson_id']}.txt").read_text(encoding="utf-8")
                r = api.post(f"/modules/{module_id}/lessons",
                             json={"title": licao["title"], "content": texto, "lessonType": "ARTICLE"})
                r.raise_for_status()
                print(f"    Aula: {licao['title']} ({len(texto.split())} palavras)")

            r = api.post(f"/modules/{module_id}/lessons",
                         json={"title": f"Avaliação: {modulo['title']}", "lessonType": "QUIZ"})
            r.raise_for_status()
            print("    Aula de avaliação (vazia, para gerar o quiz com IA)")

        if args.publicar:
            api.patch(f"/courses/{course_id}/publish").raise_for_status()
            print("Curso publicado.")

    print("\nPronto. O Trainify está mandando as aulas para o trainify-ai indexar em segundo plano;")
    print("acompanhe pelo log do trainify-ai (cada aula aparece como 'indexada').")
    return 0


if __name__ == "__main__":
    sys.exit(main())
