"""O antigo preenchimento sintético não é mais um caminho de treinamento.

Dados de demonstração não podem preencher classes sem gravações reais. Use a aba
Coleta para registrar pessoas e sessões independentes, incluindo exemplos negativos.
Os arquivos antigos estão preservados em legacy/ e não são importados pela v2.
"""


def main():
    print("O preenchimento sintético foi desativado.")
    print("Execute py app.py, entre em Administração e colete gravações reais.")
    print("Cada classe exige pessoas independentes; veja o protocolo no README.")


if __name__ == "__main__":
    main()
