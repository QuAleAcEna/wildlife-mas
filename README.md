PARA CORRER A INTERFACE!!!!!!

1. Cria/ativa o teu ambiente virtual (`python -m venv .venv && source .venv/bin/activate`) e instala as dependências (spade, python-dotenv, pandas, matplotlib, etc.). Verifica o `.env` na raiz para garantir que os JIDs/passwords estão corretos.
2. Vai para a pasta `src` e corre `python -m interface.run_dashboard`. Isto levanta os agentes, o dashboard writer e abre a UI no browser. O `run_all` corre automaticamente através deste comando.
3. Deixa o cenário correr o tempo necessário e termina com `Ctrl+C`. Ao sair, os ficheiros `src/interface/reports/dashboard_history.json` e `src/interface/reports/dashboard_metrics.csv` são atualizados com os reports mais recentes.
4. (Opcional) Para gerar o gráfico dos KPIs, ainda em `src`, corre `python -m reports.reports.plot_dashboard_metrics --csv interface/reports/dashboard_metrics.csv --output interface/reports/dashboard_metrics.png`. O PNG fica na mesma pasta `interface/reports`.
