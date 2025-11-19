PARA CORRER A INTERFACE!!!!!!

1. Cria/ativa o teu ambiente virtual (`python -m venv .venv && source .venv/bin/activate`) e instala as dependências (spade, python-dotenv, pandas, matplotlib, etc.). Verifica o `.env` na raiz para garantir que os JIDs/passwords estão corretos.
2. Vai para a pasta `src` e corre `python -m interface.run_dashboard`. Isto levanta os agentes, o dashboard writer e abre a UI no browser. O `run_all` corre automaticamente através deste comando.
3. Deixa o cenário correr o tempo necessário e termina com `Ctrl+C`. Ao sair, os ficheiros `src/interface/reports/dashboard_history.json` e `src/interface/reports/dashboard_metrics.csv` são atualizados com os reports mais recentes.
4. (Opcional) Para gerar o gráfico dos KPIs, ainda em `src`, corre `python -m reports.reports.plot_dashboard_metrics --csv interface/reports/dashboard_metrics.csv --output interface/reports/dashboard_metrics.png`. O PNG fica na mesma pasta `interface/reports`.

## Docstrings & documentation

- O projeto segue docstrings em estilo Google/PEP 257 para módulos, classes e funções. Garante um resumo numa linha, seguido das secções `Args`, `Returns`, `Raises`, etc., quando fizer sentido.
- A consistência é importante: usa sempre `"""Triple quotes"""`, descreve apenas o comportamento não óbvio e documenta atributos em docstrings de classe.
- Podes pré-visualizar a documentação gerada automaticamente com [`pdoc`](https://pdoc.dev). Exemplo:
  ```bash
  # a partir da raiz do repo
  PYTHONPATH=src pdoc agents.ranger_agent agents.sensor_agent core.messages -o docs
  ```
  O comando cria HTML em `docs/` que podes abrir no browser. Ajusta os módulos conforme precisares (por exemplo `PYTHONPATH=src pdoc agents core -o docs` para todo o pacote).
- `help(obj)` no REPL ou `python -m pydoc module` continuam a mostrar as mesmas descrições, portanto mantém os docstrings atualizados sempre que alterares o comportamento.
