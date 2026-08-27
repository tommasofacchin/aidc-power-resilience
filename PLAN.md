# Piano operativo — Huawei Tech Arena 2026, Topic 2 (Phase 1)

**Aggiornato:** 19 agosto 2026
**Deadline:** 31 agosto 2026 (prorogata dal 24; annuncio admin del 21 agosto sul forum)
**Obiettivo interno:** pacchetto completo pronto la sera del **23 agosto** (1 giorno di buffer).

---

## 0. Cosa è cambiato rispetto al piano precedente

Il documento `docs/submission_guidelines_phase1-AIDC.docx` sovrascrive diversi punti di `docs/rules.md`. Dove i due divergono, **vincono le submission guidelines**.

| Punto | Vecchia assunzione (rules.md) | Regola effettiva Phase 1 |
|---|---|---|
| Ground truth | NaFIRS, UK | **EAGLE-I, USA** (contee + FIPS) |
| Target | risk score sito-specifico | **solo `x = customers_out / total_customers`** |
| Architettura AIDC | 4 topologie, sigmoide (k, x0) | **fuori scope** — rinviata a Phase 2 |
| Risoluzione Task B | 5 minuti | **15 minuti** (granularità nativa EAGLE-I) |
| Siti | forniti dagli organizzatori | **5 contee scelte da noi**, da giustificare |
| Meteo | Open-Meteo generico | **solo ECMWF IFS HRES via Single Runs API** |

**Conseguenza pratica:** tutto il lavoro su UPS/HVDC/2N/backup duration è rinviato. Phase 1 è un problema puro di *forecasting meteo → outage ratio a livello contea*. Non spendere tempo sull'architettura elettrica.

---

## 1. Il deliverable esatto

Tre componenti, tutti obbligatori (manca uno → non viene valutato):

1. **`report.pdf`** — 3–8 pagine, esclusi appendici
2. **`predictions.csv`** — Task A + Task B insieme, distinti da `task_id`
3. **`code/` + `README.md`** — pipeline completa e riproducibile

Struttura pacchetto suggerita dagli organizzatori:

```
submission/
├── report.pdf
├── predictions.csv
├── code/
│   ├── data_acquisition/
│   ├── preprocessing/
│   ├── features/
│   ├── train.py
│   ├── predict.py
│   └── requirements.txt
└── README.md
```

### Formato CSV

`task_id, fips_code, county, state, issue_time, target_time, predicted_x`

- `task_id`: `"A"` o `"B"`
- `fips_code`: stringa **5 cifre, zero-padded** (⚠️ mai aprire il CSV in Excel: distrugge gli zeri iniziali)
- `issue_time` / `target_time`: ISO 8601 UTC con `Z` — es. `2025-09-01T00:00:00Z`
- `predicted_x`: float in `[0, 1]`, **rapporto**, non conteggio grezzo

---

## 2. Decisioni chiave (già prese, da confermare)

### 2.1 Schedule di emissione

Horizon e risoluzione sono fissi: ogni batch deve contenere **l'insieme completo ed esatto** dei lead time. Batch parziali = non conformi.

| Task | Issue times | Lead times | Righe/batch (5 contee) | N. batch |
|---|---|---|---|---|
| A | ogni giorno 00:00Z, dal **2025-08-30** al **2025-11-30** | +1h … +48h (48 step) | 240 | 93 |
| B | ogni 6h (00/06/12/18Z), dal **2025-08-31T18:00Z** al **2025-11-30T18:00Z** | +15m … +6h (24 step) | 120 | 365 |

Le prime emissioni partono **prima** del 1° settembre perché servono a coprire i `target_time` del primo giorno della finestra (le righe fuori finestra semplicemente non vengono valutate).

⚠️ **Rettifica del 24 agosto (sera).** La versione precedente di questa tabella faceva
finire il Task A il **29 novembre**, cioè all'ultimo giorno il cui orizzonte +48h resta
dentro la finestra. Ma la frequenza minima richiesta è *un batch per giorno di calendario*
e vale **su tutta la finestra annunciata**, che finisce il 30: il 30 novembre restava senza
nessuna emissione Task A. Nessun gap fra due `issue_time` consecutivi era però più largo di
24h, quindi il controllo sui gap — l'unico che il validator faceva — non poteva vederlo.
`validate_submission.py` ora verifica la **copertura degli slot**, non i gap, e
`code/predict.py` emette anche il 30. Il batch non costa quota (usa il run 29 nov 12Z che il
Task B di quel giorno già consuma) e 23 dei suoi 48 lead cadono dentro la finestra, a 1–23h
invece delle 25–48h con cui il batch del 29 le copriva.

**Totale righe: 66.120** (22.320 per A + 43.800 per B). Volume trascurabile.

⚠️ Target sovrapposti tra batch diversi sono **voluti** — non deduplicare. Lo scoring guarda l'accuratezza in funzione del lead time.

### 2.2 Mapping issue_time → run meteo (il punto più delicato)

ECMWF IFS HRES gira 4×/giorno (00/06/12/18Z) ma i dati non sono disponibili *all'istante* del run: la disseminazione richiede ~5–7 ore. Usare il run `T` all'issue time `T` sarebbe leakage mascherato.

**Regola adottata (conservativa e difendibile):**

- **Task A**, issue `D 00:00Z` → run **`D-1 12:00Z`** (lead +12h … +60h)
- **Task B**, issue `T` → run **`T − 6h`** (lead +6h15m … +12h)

Questa scelta va scritta esplicitamente nel report: è il tipo di dettaglio che distingue una pipeline realistica da una che bara.

Run distinti necessari: ~4/giorno × 94 giorni ≈ **376 chiamate API**. Costo trascurabile.

### 2.3 API meteo — parametri verificati

```
https://single-runs-api.open-meteo.com/v1/forecast
  ?latitude=<lat1,lat2,...>&longitude=<lon1,lon2,...>
  &run=2025-09-01T00:00
  &models=ecmwf_ifs
  &hourly=<variabili>
```

- `models=ecmwf_ifs` → IFS HRES, **9 km**, orizzonte 10 giorni, archivio **dal 14 marzo 2024**
- Coordinate multiple comma-separated supportate → 5 contee in **una** chiamata (da smoke-testare sull'endpoint single-runs)
- ⚠️ **IFS non ha `minutely_15` nativo** (solo HRRR / ICON-D2 / AROME, che però hanno archivio Single Runs solo dal 2 aprile 2026 → inutilizzabili). Quindi per il Task B il meteo a 15 minuti è **interpolato dall'orario**.

**Implicazione modellistica centrale:** nel Task B il segnale genuino a 15 minuti **non arriva dal meteo** — arriva dallo **stato autoregressivo dell'outage** osservato fino a `issue_time`. Il meteo dà la busta di rischio, l'autoregressivo dà il timing. Progettare le feature di conseguenza.

### 2.4 Causalità: cosa è lecito usare

Le guidelines vincolano **solo l'input meteo** (§3.3). Lo storico degli outage osservati **fino a `issue_time` incluso** è legittimo e realistico (EAGLE-I/ODIN è near-real-time) — ed è la principale fonte di skill del Task B.

Regola ferrea nel codice: ogni feature deve essere funzione esclusiva di dati con timestamp `≤ issue_time`. Un solo assert centralizzato che lo verifica, non controlli sparsi.

### 2.5 Dati di training — aggiornato dopo la ricognizione reale (19 agosto)

⚠️ **Rettifica del 20 agosto — la versione precedente di questo paragrafo era sbagliata.** Diceva che gli anni 2015–2024 non fossero raggiungibili da nessuna fonte scriptabile ("figshare: solo 2014"). Falso: la ricognizione originale aveva enumerato solo i file id visibili sulla pagina dello share token, senza mai risolvere il token all'articolo sottostante. **L'intero archivio 2014–2025 sta su figshare, articolo 24237376 v4, CC BY 4.0 — 17 file, ~11,6 GB, ogni annata scaricabile con un GET anonimo su `https://ndownloader.figshare.com/files/<id>`, senza login né Globus.** La tabella completa degli id è in `code/data_acquisition/eagle_i.py` (`FIGSHARE_YEAR_FILES`), rigenerabile con `curl -s https://api.figshare.com/v2/articles/24237376`. Il link Globus indicato dagli organizzatori (DOI 10.13139/ORNLNCCS/1975202) copre solo 2014–2022: sottoinsieme stretto, mai conveniente.

**Ma il vincolo che stringe davvero non era EAGLE-I.** Come spiegato subito sotto, l'archivio IFS Single Runs parte dal 14 marzo 2024, e il budget giornaliero Open-Meteo è la risorsa scarsa. Quindi la disponibilità ritrovata **non riapre** il training multi-anno: gli anni ≤ 2023 restano privi di feature IFS. Quello che sblocca concretamente è (a) il **2024 dal 14 marzo in poi**, unico anno aggiuntivo compatibile con IFS, se e solo se avanza quota Open-Meteo, e (b) la **climatologia per contea su tutti e 12 gli anni**, che non consuma nemmeno una chiamata meteo. Entrambi restano opzionali rispetto alla deadline.

⚠️ **Trappola di ingest verificata sui dodici header reali:** i file annuali **non hanno schema stabile**, e la deriva è silenziosa, non fatale. 2014–2022 e 2025 usano `fips_code,county,state,customers_out,run_start_time`; il **2023 chiama `sum` la quarta colonna** invece di `customers_out`; il **2024 porta una sesta colonna `total_customers`**. Un `pd.concat` ingenuo produce quindi un `customers_out` interamente NaN per tutto il 2023, senza sollevare nulla. `load_outage_events()` ora normalizza gli header prima di concatenare e **solleva** su un rinomino sconosciuto invece di degradare in NaN.

**Vincolo aggiuntivo che restringe ulteriormente la finestra utile:** l'archivio ECMWF IFS Single Runs parte dal **14 marzo 2024**. Il campione EAGLE-I 2014 è quindi **inutilizzabile per il training basato su IFS** (non c'è sovrapposizione: EAGLE-I 2014 esiste, IFS 2014 no). L'unica finestra con **sia** ground truth **sia** feature IFS disponibili è:

**Training set: 2025-01-01 → 2025-08-31 (8 mesi), su tutte le ~3.048 contee attive in EAGLE-I nella finestra di test** (verificate scaricando e ispezionando il file reale — non una stima).

Compromesso onesto da scrivere nel report: 8 mesi di un solo anno significa **nessuna diversità stagionale multi-anno** — niente Helene/Milton 2024, niente varianza interannuale. Copre però primavera (stagione tornado, aprile–giugno) e l'inizio della stagione uragani (giugno–agosto), quindi non è banale. Il volume compensa in parte: ~3.000 contee × 8 mesi × 96 intervalli/giorno è comunque un dataset ampio, e il problema è più di generalizzazione spaziale che temporale.

**Ulteriore riduzione di scope, decisa il 19 agosto sera:** Open-Meteo applica un rate limit **orario** (non solo al minuto) sulla Single Runs API, scoperto solo in fase di download bulk. Al ritmo osservato (~10 run/ora), scaricare il meteo per tutte le ~3.000 contee attive avrebbe richiesto ~23 ore — troppo rispetto al buffer verso il 24 agosto. **Il campione di training è stato ridotto da un'idea iniziale di ~250 contee a 102 contee** (stratificate per regione census + tier di attività storica, le 5 contee di reporting sempre incluse), portando il tempo stimato a ~9 ore, eseguibile durante la notte. Vedi `code/preprocessing/select_training_sample.py` per la logica di campionamento e `code/data_acquisition/bulk_download_training_weather.py` per il retry orario-aware. Compromesso da dichiarare nel report: meno contee di training rispetto al piano originale, ma comunque un campione diversificato per regione e livello di attività, non un sottoinsieme casuale.

*Fallback / ablation, solo se avanza tempo:* il campione 2014 (nov–dic) resta usabile per un pretraining su **ERA5** (Historical Weather API, che copre 2014 senza il vincolo IFS) seguito da fine-tuning su IFS 2025. Buon paragrafo di ablation, non un requisito — e comunque marginale con solo 2 mesi di dati aggiuntivi.

**Seconda riduzione di scope, decisa la notte del 19 agosto:** oltre al limite orario, Open-Meteo applica anche un limite **giornaliero** sulla Single Runs API (~110–130 chiamate/giorno osservate, non documentato). Cruciale: questa quota è **condivisa con la generazione della submission vera**, non solo con il training — `code/predict.py` richiede **367 chiamate distinte** (un run IFS per ogni combinazione univoca issue_time→run tra Task A e Task B, calcolato costruendo davvero lo schedule, non stimato), ed è vicino a un pavimento fisso perché le guidelines impongono una frequenza minima di emissione non comprimibile.

Con 204 run di training ancora mancanti + 367 di generazione = 571 chiamate contro un budget realistico di ~450–500 chiamate nei giorni rimasti prima della deadline, il rischio di non farcela con il solo livello gratuito era concreto. Scartata l'opzione di un piano a pagamento Open-Meteo ($29/mese, 1M chiamate — probabilmente avrebbe risolto il problema, ma la scelta di restare nel livello gratuito è stata presa esplicitamente). **Il training aggiuntivo è stato limitato a un budget fisso di 80 chiamate** (invece delle 204 rimaste), portando il totale a 120 run invece di 244, per lasciare margine di quota alla generazione. Il campionamento delle 80 chiamate aggiuntive è **uniformemente distribuito sulle date ancora mancanti** (non le prime cronologicamente), per non perdere la copertura di piena estate/inizio stagione uragani che era il motivo originale della finestra gennaio-agosto — vedi `ADDITIONAL_RUN_BUDGET` in `code/data_acquisition/bulk_download_training_weather.py`.

Compromesso onesto da scrivere nel report: il training finale userà ~120 run invece dei ~244 originariamente pianificati (già ridotti da un'idea iniziale multi-anno). È un dataset di training via via più piccolo a ogni vincolo scoperto — ma ogni riduzione è stata verificata sul campo, documentata, e il campionamento è sempre rimasto stratificato/uniforme, mai un troncamento cronologico che avrebbe distorto la copertura stagionale.

*Reintegrare altri anni EAGLE-I* non richiede più di trovare una fonte: `python -m data_acquisition.eagle_i <anni>` li scarica da figshare e `load_outage_events()` li raccoglie da `data/raw/eaglei_outages_<anno>.csv`. Il costo non è l'ingest ma il meteo abbinato — vale la pena solo per il 2024 post-14-marzo, o per la climatologia, che di meteo non ne consuma.

### 2.6 Selezione delle 5 contee — aggiornato

⚠️ **Trappola di leakage originale:** scegliere le contee guardando settembre–novembre 2025 (la finestra di test) sarebbe look-ahead bias. **Non disponendo di climatologia 2014–2024**, la selezione si basa su **gennaio–agosto 2025** — che è comunque *prima* della finestra valutata e rientra nel periodo di training raccomandato esplicitamente dalle guidelines (fino al 31 agosto 2025). Non è la climatologia decennale originariamente prevista, ma non è nemmeno un leakage: i mesi valutati restano non guardati.

Criteri (invariati nella sostanza, invariata la fonte dati disponibile):
1. Densità di eventi outage gen–ago 2025 (non più "densità storica decennale nella finestra stagionale set–nov", perché quella finestra specifica non è nei dati che abbiamo — usiamo il proxy disponibile e lo dichiariamo)
2. Copertura EAGLE-I continua nei 3.048 contee attive (poche lacune)
3. Diversità di regime meteo: costa atlantica/golfo, nord-est, interno
4. `total_customers` non troppo piccolo (contee minuscole → `x` rumorosissimo)

Da documentare nella §2 del report con i numeri a supporto, **incluso il limite esplicito**: la selezione non riflette una climatologia pluriennale per assenza di dati storici accessibili, solo 8 mesi di un singolo anno.

**Selezione finale** (eseguita il 19 agosto, `code/preprocessing/select_counties.py` → `data/processed/selected_counties.csv`):

| FIPS | Contea | Stato | Regime |
|---|---|---|---|
| 72013 | Arecibo | Puerto Rico | Tropicale/caraibico — segnale più alto in assoluto |
| 37119 | Mecklenburg | North Carolina | Piedmont/sud-est — resti di uragani + convezione |
| 54005 | Boone | West Virginia | Appalachi/interno — ghiaccio + temporali |
| 26097 | Mackinac | Michigan | Grandi Laghi — tempeste invernali |
| 22071 | Orleans | Louisiana | Golfo/uragani — inclusa deliberatamente per copertura di regime nonostante il segnale gen-ago sia debole (il picco stagione uragani cade nella finestra di test) |

Nota emersa durante l'esecuzione: ordinare per semplice conteggio di righe-evento è una trappola — riscopre solo le contee più popolose (Harris TX, Miami-Dade, ecc. avevano event_rate ≈99.9%, ma è un trickle permanente di guasti locali di routine, non rischio meteo estremo). Il ranking usato invece conta gli intervalli con `x > 1%` (evento reale) — vedi il modulo per il dettaglio.

### 2.7 Denominatore `total_customers`

**Risolto il 24 agosto.** Per mesi abbiamo dato per scontato che il valore di riferimento
fosse fissato dagli organizzatori e non pubblicato. Non è così: alla domanda diretta se lo
scoring divida per `total_customers` esattamente come sta in `MCC.csv` anche dove il
`customers_out` registrato lo supera, la risposta è stata *«even the official dataset could
have mistake, when grading we will ignore such timestamp»*. Solo un grader che divide per MCC
pubblicato può produrre un timestamp da ignorare, quindi l'unità è fissata: **il numero
consegnato è una frazione di MCC**.

Restano due denominatori, con due compiti diversi:

- **riconciliato** (`total_customers_reconciled.csv`) — è quello su cui il modello si allena.
  Un target che satura a `x = 1` per tutta la durata di una tempesta non dice al modello
  quanto grande fosse la tempesta.
- **MCC pubblicato** — è quello in cui la submission viene letta.

La conversione è una costante per contea, applicata in un solo punto (`to_grading_units()` in
`predict.py`): tre contee su cinque coincidono già con MCC (×1), Mecklenburg ×20,89 e Arecibo
×4,66.

---

## 3. Architettura del modello

### 3.1 Il problema statistico

`x` è un rapporto in `[0,1]`, **zero-inflated all'estremo** (la stragrande maggioranza dei quarti d'ora, in quasi tutte le contee, vale 0) e con coda pesante. Questo è *il* problema del task, ed è esattamente ciò che le guidelines chiedono di affrontare nella sezione "class-imbalance handling".

### 3.2 Modello primario: LightGBM con obiettivo Tweedie

`objective='tweedie'`, `tweedie_variance_power ≈ 1.3–1.7` (da tunare). Gestisce nativamente un target continuo, non negativo, con massa di probabilità in zero. Un solo modello, niente ricomposizione di pezzi.

### 3.3 Modello alternativo: hurdle a due stadi

- Classificatore: `P(x > τ)` con `τ ≈ 0.001`
- Regressore condizionale: `E[log x | x > τ]`
- Predizione: `P × exp(E[·])`

Serve a due scopi: è il confronto principale dell'**ablation analysis** ed è la rete di sicurezza se il Tweedie si comporta male.

### 3.4 Un modello per task

Modelli separati per A e B, ciascuno con **`lead_time` come feature esplicita** (un solo modello copre tutti i lead time del suo task). Le feature disponibili differiscono troppo tra i due task per giustificare un modello unico.

### 3.5 Feature

**Meteo (dal forecast IFS al lead time corretto)**
- `wind_gusts_10m`, `wind_speed_10m`, `wind_direction_10m`
- `precipitation`, `snowfall`, `rain`
- `temperature_2m`, `dew_point_2m`, `surface_pressure`
- `cape`, `cloud_cover`, `freezing_level_height`

**Derivate meteo — qui sta il valore aggiunto**
- `gust³` (l'energia del vento ∝ v³, il danno alle linee scala similmente)
- Massimo rolling delle raffiche su finestre ±3h, ±6h, ±12h attorno al target
- Precipitazione cumulata 6h / 24h fino al target
- **Superamento di quantili climatologici per-contea**: `gust > p95`, `> p99`, `> p99.9` della climatologia locale. È la feature che rende il modello sensibile all'*anomalia* invece che al valore assoluto — fondamentale per generalizzare tra contee costiere e interne.
- Combinazione vento×pioggia (terreno saturo + raffiche = alberi che cadono)
- Gradienti temporali: variazione di pressione, salto di raffica

**Temporali**
- `lead_time` (in ore/step)
- ora del giorno e giorno dell'anno, encoding sin/cos

**Statiche di contea**
- FIPS come categorica
- `total_customers`, densità di clienti
- Quantili climatologici di raffica/pioggia della contea

**Autoregressive (solo dati ≤ `issue_time`)** — *decisive per il Task B*
- `x` all'issue time, e a −15m, −30m, −1h, −2h, −6h
- Trend e derivata prima recente
- Flag "outage in corso", durata dell'outage corrente
- Massimo di `x` nelle ultime 24h

### 3.6 Gestione dello sbilanciamento

1. **Sottocampionamento dei negativi** con pesi correttivi: tenere tutte le righe con attività di outage nella contea ±24h, campionare al ~5% i periodi tranquilli, riassegnare `sample_weight = 1/p` per non distorcere la calibrazione.
2. Obiettivo Tweedie (già intrinsecamente adatto).
3. Valutazione **stratificata per regime**: le metriche globali sono dominate dagli zeri e non dicono nulla. Riportare sempre anche il sottoinsieme `x_true > 0`.

### 3.7 Validazione

**Split temporale, mai casuale.**
- Train: 2024-03-14 → 2025-05-31
- Validation: 2025-06-01 → 2025-08-31 (pre-finestra di test, include l'inizio della stagione uragani)

La validazione deve **simulare esattamente la rolling issuance** del test: stessi issue_time, stessa regola di mapping run→issue, stessi lead time. Un backtest che non replica la procedura di consegna non misura nulla di utile.

### 3.8 Metriche

⚠️ **Le guidelines non specificano la metrica di scoring.** Non essendoci un target ufficiale da ottimizzare, riportiamo una suite e lo dichiariamo nel report:

- MAE e RMSE su tutte le righe valutate
- MAE ristretto a `x_true > 0.001` (il regime che conta davvero)
- POD / FAR / CSI a soglie multiple; Brier score per `P(x > τ)`
- **Skill in funzione del lead time** — le guidelines dicono esplicitamente che lo scoring esaminerà l'accuratezza per lead time, quindi la curva va nel report
- Diagramma di affidabilità (calibrazione)

**Baseline obbligatorie per il confronto:** predittore costante zero, climatologia per-contea, persistenza. Se il modello non batte la persistenza sul Task B a lead brevi, c'è un bug.

---

## 4. Piano giorno per giorno

### Giorno 1 — giovedì 20 agosto: *de-riskare tutto ciò che è esterno*

Priorità assoluta: nessuna riga di modello finché non è certo che i dati arrivino.

- [ ] **Smoke test Single Runs API**: una chiamata con `run=2025-09-15T00:00`, `models=ecmwf_ifs`, 5 coordinate comma-separated. Verificare che il multi-coordinate funzioni su *quell'* endpoint e che le variabili richieste esistano tutte.
- [ ] Verificare copertura archivio sull'**intera** finestra (spot check: marzo 2024, gennaio 2025, novembre 2025)
- [ ] Misurare i rate limit reali con una raffica controllata
- [x] Scaricare **EAGLE-I 2025** e `MCC.csv` (figshare, anonimo). Anni storici 2014–2024 disponibili on demand: `python -m data_acquisition.eagle_i <anni>`
- [ ] Ispezionare il formato reale: colonne, timezone dei timestamp, se le righe a zero sono assenti (quasi certamente sì → serve reindicizzazione su griglia 15-min completa con fill a zero)
- [ ] **Selezionare le 5 contee** su climatologia 2014–2024, con i numeri a supporto
- [ ] Scrivere `data_acquisition/` con **caching su disco** (parquet). Ogni chiamata API fatta una sola volta nella vita del progetto.

**Gate di fine giornata:** se l'archivio IFS ha buchi sulla finestra di test, il piano cambia radicalmente. Va saputo il 20, non il 23.

### Giorno 2 — venerdì 21 agosto: *dati e feature*

- [ ] Scaricare i run IFS per il training (~150 contee, 2024-03-14 → 2025-08-31, run 00Z e 12Z)
- [ ] Scaricare i 376 run per la finestra di test
- [ ] Costruire la griglia target EAGLE-I: reindicizzazione 15-min, fill zero, calcolo di `x`
- [ ] Allineamento temporale run→lead→target (il punto dove si annidano i bug: scrivere test unitari su questo)
- [ ] Pipeline feature completa, con l'assert centralizzato di causalità
- [ ] **Baseline funzionante entro sera**: persistenza + climatologia, valutate sullo split di validation

**Gate:** una baseline end-to-end che produce un CSV valido. Da qui in poi si ha sempre qualcosa da consegnare.

### Giorno 3 — sabato 22 agosto: *modelli*

- [ ] LightGBM Tweedie, Task A e Task B
- [ ] Modello hurdle come confronto
- [ ] Tuning contenuto (poche run, non una grid search: il tempo è la risorsa scarsa)
- [ ] **Ablation** — servono per il report, quindi loggare tutto in modo strutturato:
  - senza feature autoregressive
  - senza quantili climatologici
  - senza feature derivate dalle raffiche
  - Tweedie vs hurdle
  - meno contee di training
- [ ] Curve di skill vs lead time, calibrazione

**Gate:** modello che batte le baseline in modo documentato.

### Giorno 4 — domenica 23 agosto: *generazione, validazione, scrittura*

- [ ] Generare tutti i 457 batch → `predictions.csv` (~65.900 righe)
- [ ] **Validatore CSV** automatico (vedi §6) — deve passare al 100%
- [ ] Scrivere il report (3–8 pagine)
- [ ] Scrivere `README.md` con setup ambiente, ordine di esecuzione, tempi attesi per step
- [ ] `requirements.txt` con versioni pinnate
- [ ] **Prova di riproducibilità da zero**: clonare in una cartella pulita, seguire il README alla lettera, verificare che esca lo stesso CSV

**Gate:** pacchetto completo e consegnabile la sera del 23.

### Giorno 5 — lunedì 24 agosto: *buffer e consegna*

La scadenza è poi slittata al 31 agosto (annuncio admin del 21), quindi questo giorno è
diventato buffer vero e non l'ultimo giorno utile.

- [x] Rilettura del report, controllo che ogni sezione richiesta ci sia — rifatta il 24
  agosto contro `docs/submission_guidelines_phase1-AIDC.docx` voce per voce: report §3
  (acquisizione/preprocessing), §4 (feature), §5 (modello), §6 (sbilanciamento), §7
  (risultati e ablation), §9 (fonti e licenze), §2 (giustificazione delle 5 contee)
- [x] Checklist di conformità §6
- [x] Pacchetto pronto: `dist/submission.zip`, struttura identica a quella suggerita in §5
  delle guidelines
- [ ] **Consegnare** — non fatto in automatico, va caricato a mano sul portale

---

## 5. Rischi

| Rischio | Impatto | Mitigazione |
|---|---|---|
| Archivio IFS con buchi sulla finestra | **Bloccante** | Verificare il giorno 1. Fallback: run più vecchio disponibile + flag "stale forecast" come feature |
| Multi-coordinate non supportato su single-runs | Basso | 5× chiamate, comunque trascurabile |
| Rate limit più stretti del previsto | Medio | Caching aggressivo, download notturno in background |
| ~~`total_customers` diverso da quello degli organizzatori~~ | **Chiuso il 24 ago** | Gli organizzatori valutano su MCC.csv pubblicato; `predict.py` converte nell'unità loro |
| EAGLE-I con lacune di copertura sulle contee scelte | Medio | Criterio di selezione contee include la continuità della copertura |
| Metrica di scoring ignota | Medio | Suite di metriche, non over-fittare su una sola |
| Tempo insufficiente per il modello "bello" | Alto (5 giorni) | Baseline consegnabile già il giorno 2; ogni giorno successivo migliora qualcosa di già valido |

**Principio guida:** avere sempre un pacchetto consegnabile. Meglio un LightGBM onesto, documentato e riproducibile che un'architettura ambiziosa incompleta il 24 sera.

---

## 6. Checklist di conformità pre-consegna

**Verificata voce per voce il 24 agosto 2026** contro `submission/predictions.csv` e
`report/report.pdf` rigenerati quel giorno. Le voci CSV sono controllate da
`code/validate_submission.py`, quindi non vanno rifatte a mano: vanno ri-eseguite.

CSV:
- [x] `fips_code` stringa a 5 cifre con zeri iniziali preservati — verificato sul testo grezzo, non dopo il parsing
- [x] Tutti i timestamp ISO 8601 UTC con suffisso `Z` — 66.120/66.120
- [x] Task A: esattamente 48 righe per (issue_time, contea); Task B: esattamente 24
- [x] Tutti i batch contengono **tutte e 5** le contee
- [x] Frequenza minima rispettata — A: 93 batch, uno per **ogni** giorno di calendario della finestra; B: 365 batch, uno per ogni slot da 6h. Verificata come copertura degli slot, non come gap fra emissioni consecutive: era il gap a nascondere il 30 novembre scoperto (vedi la rettifica in §2.1)
- [x] `predicted_x` ∈ [0,1], nessun NaN, nessuna notazione scientifica — *questa era l'unica voce fallita*: pandas scriveva in notazione scientifica sotto 1e-4 (21.039 righe). `predict.py` ora scrive in virgola fissa e il validator lo segnala se ricapita
- [x] Nessuna deduplicazione dei target sovrapposti — 600 target fuori finestra conservati, nessun duplicato su (task, issue, target, contea)
- [x] Copertura completa della finestra 2025-09-01 → 2025-11-30 — 91/91 giorni

Report:
- [x] 3–8 pagine, PDF — **8**, cioè al limite: ogni aggiunta va compensata
- [x] Acquisizione e preprocessing (§3), feature e razionale (§4), modello e training (§5)
- [x] **Gestione dello sbilanciamento** — §6, sezione dedicata come richiesto
- [x] Risultati sperimentali e **ablation** — §7, Tabella 2
- [x] **Dichiarazione fonti e licenze** — §9, Tabella 5, verificate alla fonte
- [x] Giustificazione della scelta delle 5 contee — §2, Mecklenburg dichiarata come debolezza nota

Codice:
- [x] Pipeline completa: acquisizione → preprocessing → feature → training → predizione
- [x] `requirements.txt` con versioni pinnate (dirette + transitive risolte)
- [x] README con configurazione ambiente, ordine di esecuzione, tempi attesi
- [x] **Testato da zero in ambiente pulito** — rifatto il 24 agosto sera al commit
  `f79125d`, cioè dopo l'aggiunta del batch A del 30 novembre, contro l'archivio corrente:
  clone in directory vuota fuori dal working tree, venv nuovo da `requirements.txt`,
  `data/raw` dall'archivio, passi 3–12 (17 minuti). Tutti e sette gli artefatti tornano
  byte-identici (MD5 nel README), `predict.py` non ha speso una sola chiamata API sui 93
  run, e il pacchetto si costruisce dal clone — con dentro i due file pinnati sotto
  `data/processed/`, che prima mancavano dallo zip.
- [x] Le tre componenti obbligatorie arrivano insieme — `code/make_submission_package.py`
  costruisce `dist/submission/` nella struttura suggerita dalle guidelines. **Si consegna
  `dist/submission.zip`, non la cartella `submission/`**, che da sola manca della codebase
  e quindi non verrebbe valutata.
- [x] Il pacchetto è autosufficiente per chi parte dallo zip — include
  `data/processed/training_counties.csv`, senza il quale il passo 5 ri-deriva un campione
  diverso e il passo 7 si allena su 59 contee riportando successo (vedi README §6).

---

## 7. Fonti dati e licenze (bozza per il report)

| Fonte | Uso | URL | Licenza — **da verificare** |
|---|---|---|---|
| EAGLE-I Power Outage Data 2014–2025 (ORNL) | Ground truth | https://www.osti.gov/biblio/3012826 | DOI 10.13139/ORNLNCCS/3012826 |
| Open-Meteo Single Runs API (ECMWF IFS HRES) | Feature meteo previsionali | https://open-meteo.com/en/docs/single-runs-api | CC BY 4.0 (Open-Meteo); dati ECMWF sotto licenza propria |
| Open-Meteo Historical Weather API (ERA5) | *Opzionale*, pretraining | https://open-meteo.com/en/docs/historical-weather-api | CC BY 4.0 |

⚠️ Le licenze vanno **verificate alla fonte e citate testualmente**, non assunte. È un requisito esplicito.

---

## 8. Fuori scope in Phase 1 (non lavorarci)

- Topologie di alimentazione AIDC (UPS 2N, UPS DR, HVDC 2N, Direct Utility 2N)
- Trasformazione sigmoidale con parametri `k`, `x0` dipendenti dalla topologia di alimentazione
- Critical-load coverage ratio e backup duration
- Dataset NaFIRS (UK)

Tutto questo torna in **Phase 2**, quando gli organizzatori pubblicheranno il dataset delle topologie e il design di scoring relativo.

---

# Piano 23–31 agosto (scadenza prorogata al 31)

Scritto il 22 agosto, dopo che la traccia "fix sicuri" è stata chiusa e verificata
(riproduzione da clone pulito identica bit-a-bit, `3f81222`). La submission attuale è
valida e committata: **tutto ciò che segue è miglioramento su un fallback solido**, e
ogni passo può essere abbandonato senza perdere la consegna.

## Il vincolo che governa tutto

Open-Meteo ha **una sola quota giornaliera condivisa tra tutti gli endpoint** — archive
(ERA5) e Single Runs attingono allo stesso contatore. Verificato il 22 agosto: tre sonde
ERA5 hanno esaurito il budget e il canale Single Runs ha risposto 429 subito dopo.

Conseguenza pratica: **ERA5 è fuori**. Costa molto di più per unità di dato utile (il
costo scala con l'ampiezza dell'intervallo di date) e avrebbe in più lo skew train/serve.
Il canale Single Runs rende ~110–130 chiamate/giorno, ognuna delle quali copre 102 contee
× 62 ore × 14 variabili, e usa la stessa fonte che serve le predizioni: nessuno skew.

## Perché densificare: già misurato, non ipotizzato

`code/densification_probe.py` è una cross-validation temporale a 3 fold su settembre,
ottobre e novembre 2024, con il set di valutazione **congelato** a
`data/processed/baseline_runs_20260822.json` — così i run scaricati dopo non possono
entrare in ciò che viene misurato, e il confronto prima/dopo resta onesto.

Simulando la densificazione con i dati già presenti (49 run autunnali contro 98):

| fold | Δ RMSE sistema completo |
|---|---|
| 2024-09 (Helene) | +0,71 % |
| 2024-10 (Milton) | +1,05 % |
| 2024-11 (calmo)  | −0,15 % |
| **aggregato, 354.858 righe** | **+0,70 % RMSE, +2,17 % MAE eventi** |

I mesi di tempesta guadagnano, il mese calmo è neutro — e ha un RMSE 7× più piccolo,
quindi pesa poco in qualunque aggregato. La finestra di test è proprio settembre–novembre.

## Le lacune, contate

| Blocco | Presenti | Mancanti | Priorità |
|---|---|---|---|
| Autunno 2024, griglia 00/12Z | 98/182 | **84** | 1 |
| Dicembre 2024 | 0/62 | **62** | 2 — mese oggi del tutto assente |
| Autunno 2024 a 4 cicli (00/06/12/18Z) | 98/364 | 266 | 3 — solo se 06/18Z esistono |
| Gen–ago 2025 | 144/486 | 342 | 4 — stagione sbagliata |

## Esecuzione

**23 ago — scaricare (una riga, poi lasciar stare).** Assorbe da sé tutte e tre le quote
e attraversa il tetto giornaliero aspettando: si avvia una volta e si lascia andare anche
per più giorni. 146 chiamate nuove.

    python code/data_acquisition/bulk_download_training_weather.py \
        --start 2024-09-01 --end 2024-12-31 --day-stride 1 --budget 200

Aggiungere `--dry-run` per vedere prima cosa farebbe, senza spendere una chiamata.

**23 ago — fatto.** Il download è finito da solo alle 04:43 del 23: 244/244 run,
attraversando una volta il tetto orario e una volta quello giornaliero. Autunno 2024
completo sulla griglia 00/12Z (182/182) e dicembre 2024 da 0 a 62.

**24 ago — go/no-go: ha stampato NO-GO, e va letto per quello che misura.**

    python code/features/build_training_table.py --years 2024 2025
    python code/densification_probe.py

Esito su 709.716 righe congelate: **+0,35 % RMSE, +1,54 % MAE eventi** in aggregato —
sotto la soglia di 0,5 % registrata in anticipo. Per fold: settembre +0,13 %, ottobre
**+4,79 %**, novembre +0,05 %. L'aggregato è più basso della simulazione (+0,70 %) perché
settembre domina l'RMSE ed è il mese che non si è mosso: Helene era già ben coperta.

Il NO-GO risponde a *«vale la pena spendere altra quota per scaricare ancora?»*, e la
risposta è no. Non risponde a *«uso i run che ho già su disco?»*: B non è mai peggio di A
in nessun fold e i run sono già pagati. **Decisione presa: rigenerare, senza spendere
altra quota.**

**24 ago — rigenerato.** Sequenza eseguita, in quest'ordine:

    python code/train.py
    python code/blend.py --season autumn
    python code/predict.py                    # 0 chiamate API: tutti i run erano in cache
    python code/validate_submission.py submission/predictions.csv
    python code/ablation.py && python code/seasonal_holdout.py
    python code/report_figures.py --season autumn
    python code/report_figures.py --season reference
    python code/report_tables.py
    python code/report_numbers.py             # nuovo: ricalcola i numeri citati nel report

`report_numbers.py` esiste perché i numeri del report venivano da cinque script diversi
più una manciata che nessuno script possedeva (la Tabella 4 fra stagioni, le quote di
split gain, il peso di persistenza effettivo sulle righe consegnate): proprio quelli che
diventano stantii in silenzio quando la tabella cresce. Ora hanno un proprietario.

Il PDF è stato ricostruito e resta a **8 pagine**, dentro il limite 3–8.

Cosa è cambiato nel report, oltre alle cifre: le ablation si sono ridotte e alcune hanno
cambiato segno (solo stato autoregressivo +0,30 % e meteo derivato +0,12 % restano
positive); lo split gain si è rovesciato — l'identità di contea scende dal 43 % al 19 %,
`x_at_issue` sale dal 5,7 % al 22,5 %; il fit del blend ora sceglie un decay in quattro
bucket su cinque invece di uno, ma vale lo 0,06 % complessivo. Tre numeri erano già
stantii prima di oggi (la didascalia della Figura 1, la p95 di Orleans, il confronto dello
stump): corretti.

**24 ago — chiusura, anticipata.** Il test di riproducibilità da clone pulito è stato
rifatto lo stesso giorno contro l'archivio densificato: identico bit-a-bit su tutti e sei
gli artefatti derivati. Il pacchetto `dist/submission.zip` è costruito e validato.

**Sulla scadenza — risolta il 24 agosto, con la fonte.** Messaggio di Morgane (admin) sul
forum, 21 agosto: *«We've decided to extend the submission deadline from August 24th to
August 31st. For those who have already submitted, you can revise your submission or keep
it as it is.»* La proroga è dunque reale, ma **la data è il 31 agosto, non il 30**: la
sezione qui sopra è stata scritta il 22 agosto con la data sbagliata e senza fonte, e la
memoria di progetto la ripeteva. `docs/rules.md` e il Q&A della diretta dicono ancora 24
agosto perché precedono l'annuncio. **Scadenza operativa: 31 agosto 2026.**

## Cosa hanno detto gli admin sul forum (letto il 24 agosto)

Quattro cose che toccano la consegna, in ordine di impatto.

1. **Scadenza 31 agosto** (annuncio del 21). Chi ha già consegnato può revisionare.

2. **Task A: «we will suggest hourly mean aggregation».** Noi predicevamo l'istantaneo
   alle :00. Misurato sulle 5 contee, gen–ago 2025, 28.867 ore complete: sulle ore con
   evento le due grandezze differiscono di 0,00206 in media (**19,9 % in relativo**), RMSE
   dello scarto 0,004296 — il **18 % dell'RMSE del sistema in autunno**. Corretto in
   `predict.py`: una riga Task A è ora la media dei quattro quarti d'ora che l'etichetta
   apre, `[H, H+1)`, convenzione left-label come un `resample("1h")` di default. Non serve
   riallenare (la media di stime non distorte dei quattro istanti stima senza distorsione
   la loro media) e non costa quota (il meteo a 15 minuti è interpolato in locale dallo
   stesso run in cache). Task B invariato: le sue righe sono già istanti a 15 minuti.

3. **Task B a 15 minuti è accettato.** Il chiarimento del 24 agosto dice che la definizione
   resta 5 minuti ma per Phase 1 «submissions at either 5-minute or 15-minute resolution
   are acceptable». Restiamo a 15: EAGLE-I registra a 15 minuti e scendere a 5 vorrebbe
   dire fabbricare ground truth. In Phase 2 servirà 5 minuti davvero.

4. **Nessun vincolo sulle fonti dati**, purché ogni fonte sia dichiarata con la licenza —
   che è esattamente ciò che fa §9 del report. EAGLE-I resta la base di valutazione.

### Il denominatore, e la domanda da fare sul forum

L'admin scrive: *«regarding the total customer number, MCC.CSV in the first link is the
total customer number listed in the fips code»*. Delle nostre 5 contee **3 usano già MCC**
(Orleans, Mackinac, Boone). Le altre due no, e per ragioni diverse:

- **Arecibo 72013** — MCC dice 41.122 clienti totali, EAGLE-I registra fino a 139.095 fuori
  servizio. Con MCC il ground truth stesso varrebbe x = 3,4: impossibile, non opinabile.
  Usiamo 191.803 (`mcc->consistency`).
- **Mecklenburg 37119** — MCC dice 28.172, noi usiamo 588.615 dalla colonna del file
  annuale 2024 (`infile`), perché il totale di stato della North Carolina in MCC copre
  circa un terzo del reale. Fattore **20,9×**. È l'unica esposizione vera.

**Chiesto, e risposto lo stesso giorno.** Domanda: la valutazione usa MCC.csv come
denominatore anche dove il numeratore osservato lo supera, cioè dove x risulterebbe > 1?
Risposta di Haoyang Zhang: *«Note even the official dataset could have mistake, when grading
we will ignore such timestamp»*. Cioè: sì, dividono per MCC pubblicato, e i timestamp che ne
risultano impossibili escono dal punteggio invece di essere riparati.

Questo ribalta la scelta precedente. L'argomento «sottostimare costa meno che sovrastimare»
era sbagliato, e misurabilmente: sull'autunno 2024, il grader scarta solo l'1,7 % degli
intervalli di Mecklenburg e lo 0,2 % di Arecibo perché superano 1; sul ~99 % che resta, la
verità in unità MCC vale in media 0,0151 e 0,0300 contro lo 0,00072 e 0,00643 che avremmo
consegnato. Non era una copertura prudente, era una sottostima sistematica di 21× e 4,7× su
due contee su cinque, su quasi tutte le righe su cui vengono valutate.

**Fatto:** `predict.py` converte `predicted_x` nell'unità del grader all'uscita
(`to_grading_units()`), il modello continua ad allenarsi sul denominatore riconciliato, e la
§3.4 del report racconta entrambi. Nota residua che vale la pena tenere presente: poiché MCC
è troppo piccolo per quelle due contee, i loro eventi più grandi superano 1 e **escono dal
set valutato** — le due contee con gli outage più severi vengono giudicate quasi solo sulle
ore tranquille.

## Se avanza quota e tempo

1. **Cicli 06/18Z.** Non è noto se l'archivio IFS HRES li esponga su questo endpoint.
   Sondare *una* chiamata prima di impegnare un budget:
   `--run-hours 6 --start 2024-09-01 --end 2024-09-01 --budget 1`.
   Se ci sono, l'autunno a 4 cicli vale 266 chiamate (~2,5 giorni).
2. ~~**Sostituzione di Mecklenburg** (93 chiamate + ~92 per la copertura di training).~~
   **Scartata il 24 agosto, decisione presa.** Due motivi. Il primo è che la risposta
   sul denominatore la riabilita in parte: i numeri con cui Mecklenburg era stata
   selezionata sono rapporti MCC, ed è MCC l'unità in cui viene valutata, quindi il
   criterio di selezione è soddisfatto nell'unità che conta. Il secondo è che sostituirla
   vorrebbe dire riscrivere la §2 del report e rifare la copertura di training a pochi
   giorni dalla consegna, per chiudere una debolezza che il report già dichiara apertamente.
   Resta dichiarata, non difesa.
3. **Meteo più fresco per Task B** (~91 chiamate). ~~Oggi Task B consuma una previsione
   vecchia 12–30 h per prevedere 15 minuti avanti.~~ **Scartato il 24 agosto, con i dati
   alla mano:** in `report_skill_by_lead_autumn.csv` a lead (0, 6] il blended coincide
   esattamente con la persistenza (RMSE 0,010506 entrambi) mentre il modello da solo fa
   0,032824. Nel bucket dove vive tutto Task B il blend dà peso 1,00 alla persistenza:
   una previsione meteo più fresca non ha spazio per pagare. Le 91 chiamate rendono di
   più altrove, o non si spendono.

## Cosa NON fare

- **ERA5**: stessa quota, costo molto più alto, in più skew train/serve.
- **Refit finale su tutti i dati**: aggiungerebbe maggio–agosto, la stagione meno simile
  al test, e non è validabile — l'unico autunno disponibile è già in training.
- **Passare all'hurdle**: misurato, dopo il blend i tre modelli stanno in 3 parti su
  10.000. Non vale il secondo modello da servire.

---

# 27 agosto — il modello predice il residuo, non il livello

Scritto dopo aver visto la leaderboard: 4° posto con 67.19, primo a 73.7. La domanda
«si può migliorare» ha trovato una risposta strutturale, non incrementale.

## Cosa non andava, misurato

Il LightGBM Tweedie sul livello **non aveva praticamente skill**. Sull'holdout autunnale
fuori campione, contro la costante zero:

| bucket | sempre zero | modello Tweedie | guadagno |
|---|---|---|---|
| (0, 6] | 0,034134 | 0,032824 | **3,8 %** |
| (48, 72] | 0,033913 | 0,033754 | **0,5 %** |

Due firme diagnostiche oltre al numero: l'RMSE del modello è **piatto rispetto al lead**
(0,0328 a 6 h contro 0,0338 a 72 h), cioè aveva smesso di condizionare sul presente; e la
sua predizione massima ovunque era **0,18** contro una verità che arriva a 1,0. Tutto lo
skill del sistema veniva dal blend con la persistenza, non dal booster.

La causa è l'obiettivo, non le feature. Tweedie ha un link logaritmico e su un target che
è al 69,9 % esattamente zero la risposta stimata collassa verso una costante piccola. Gli
alberi approssimano un'identità come `output = x_at_issue` solo a gradini moltiplicativi
grossolani, quindi `x_at_issue` essendo la feature a più alto gain **non** significava che
il modello sapesse riprodurla.

## La correzione

Il modello predice `target_x − x_at_issue` con obiettivo L2, e il livello consegnato è quel
residuo risommato allo stato osservato all'issue time. Il target è centrato vicino a zero e
quasi simmetrico, L2 è la perdita giusta, e la persistenza si ottiene esattamente quando il
modello emette 0 — quindi lo skill della persistenza è un **pavimento** da cui il modello
parte, invece di qualcosa che il blend deve rimettere dentro dopo.

## Quanto vale, nell'unità in cui viene valutato

Misurato sullo stesso holdout autunnale, convertito nel denominatore di grading, sulle 5
contee consegnate, applicando la regola degli organizzatori che scarta i timestamp in cui
la verità supera 1:

| contea | Tweedie + blend | delta + blend | variazione |
|---|---|---|---|
| Arecibo | 0,113798 | 0,084006 | **−26,2 %** |
| Mecklenburg | 0,097710 | 0,067678 | **−30,7 %** |
| Boone | 0,008649 | 0,007120 | −17,7 % |
| Orleans | 0,025162 | 0,022541 | −10,4 % |
| Mackinac | 0,018274 | 0,017252 | −5,6 % |
| **pooled** | **0,068473** | **0,049901** | **−27,1 %** |

**Perché il guadagno è così sbilanciato:** l'amplificazione del denominatore. Arecibo
(×4,66) e Mecklenburg (×20,89) valgono insieme il **95,5 %** del budget di errore
quadratico, perché l'errore scala col quadrato del fattore di conversione. Le altre tre
contee sono numericamente quasi irrilevanti per il punteggio. Ed è esattamente lì che il
modello delta guadagna di più, essendo le contee con più attività di outage.

Sul pool delle 102 contee di training il guadagno è solo **−1,66 %**: diluito dalle contee
tranquille che non vengono valutate. È il numero da non citare — misura una popolazione che
non è quella su cui si viene scoriati.

## Cosa è stato provato e non serve

- **Più capacità**: 1500 alberi/127 foglie fa −1,59 %, 3000/255 fa −1,47 %, contro il
  −1,66 % di 500/63. Il vincolo era l'obiettivo, non la capacità. Parametri invariati.

## Effetto sul blend

I pesi della persistenza **crollano** da 1,00 / 1,00 / 0,95 / 0,90 / 0,55 a
0,40 / 0,55 / 0,50 / 0,40 / 0,50. Nel bucket (0, 6], cioè tutto Task B, si passa da 1,00 a
0,40: prima il modello non contribuiva **per niente** a due terzi delle righe consegnate,
ora ne fa il 60 %. Il blend continua a pagare (−2,18 % sul modello da solo) ma ora migliora
una componente che ha skill propria invece di riparare una che non ne aveva.

## Nota di ingegneria

`model_bundle` ora porta `target_kind`, e `predict.py` chiama `predict_level()` invece di
`booster.predict()`. Senza quello un bundle delta letto da codice che si aspetta il livello
carica benissimo e predice numeri piccoli e plausibili — esattamente la classe di skew
silenzioso per cui quel modulo esiste. I bundle scritti prima del 27 agosto si leggono come
`level`, mai il contrario: il default opposto sommerebbe la persistenza due volte.

---

# 28 agosto — il modello è finito, e il perché è misurato

Dopo la riformulazione, tentativo sistematico di spremere altro dal modello. **Esito: niente
esce dal rumore.** Registrato qui perché è un risultato negativo utile: evita di rifarlo.

## Cosa è stato provato, sull'holdout autunnale

| variante | blended | graded (5 contee) |
|---|---|---|
| **spedita** (500 alberi, 63 foglie, lr 0,05) | 0,023188 | 0,049901 |
| lr 0,03 con 1000 alberi | 0,023184 | 0,049333 |
| 31 foglie | 0,023083 | 0,045649 |
| min_child_samples 300 | 0,023441 | 0,057224 |
| L2 = 10 | 0,023205 | 0,050028 |
| colsample 0,5 | 0,023138 | 0,051214 |
| delta troncato a ±0,30 | 0,023229 | 0,051015 |
| ensemble di 5 semi | 0,023154 | 0,049600 |

Le 31 foglie sembravano valere **−8,5 %** sulla metrica di grading. **Era rumore di un
singolo seme**, e la trappola era ben nascosta: sull'autunno la curva per foglie fa
0,0495 → 0,0483 → **0,0456** → 0,0498 → 0,0499, cioè un picco isolato e non un andamento;
sulla finestra di riferimento l'ordine non si riproduce (31 è peggio di 15 e pari a 63); e
mediando 5 semi il vantaggio sparisce.

## Il pavimento di rumore, misurato

Stessa configurazione, 8 semi diversi, metrica di grading sulle 5 contee:

| foglie | media | sd | min | max | spread |
|---|---|---|---|---|---|
| 63 (spedita) | 0,051892 | 0,00221 | 0,048787 | 0,056238 | **14,4 %** |
| 31 | 0,052139 | 0,00298 | 0,046821 | 0,055714 | **17,1 %** |

**Su 5 contee, qualunque differenza sotto il ~15 % è indistinguibile dal seme.** Questo
è il numero da tenere presente per ogni confronto futuro, e spiega perché il −27 % della
riformulazione è credibile (è quattro volte fuori dalla banda) mentre un −8 % non lo è.

Nota scomoda da tenere presente: il modello spedito (seme 42, 0,049901) sta **all'estremo
fortunato** di quella distribuzione, la cui media è 0,051892. Il numero riportato non è
sbagliato — è riproducibile bit-a-bit, il seme è fissato — ma è una realizzazione
favorevole, non il valore atteso.

L'ensemble di semi, che avrebbe rimosso quella lotteria, non aiuta: 1/3/5/10 semi danno
0,049314 / 0,050320 / 0,049600 / 0,049784 sull'autunno e 0,033360 / 0,033079 / 0,033326 /
0,033180 sul riferimento. Piatto su entrambe. Non vale il cambio di formato del bundle.

## Un difetto trovato per caso: `subsample` è inerte

`LGBM_PARAMS` contiene `subsample=0.8`, ma LightGBM fa bagging solo se `subsample_freq` > 0,
che non è impostato. Il modello salvato porta `bagging_fraction: 0.8` **insieme a**
`bagging_freq: 0`: ogni albero cresce su tutte le righe. Verificato — togliere il parametro
riproduce le predizioni **esattamente**, aggiungere `subsample_freq=1` no (scarto massimo
2,3e-2).

Non è un bug del modello, è un **difetto di documentazione**: il report dichiarava «0.8 row
and column subsampling», falso sulla metà «row». Corretto nel report; il parametro è lasciato
dov'è con una nota in `train.py`, perché rimuoverlo cambierebbe i byte di `model.txt` (la
stringa di configurazione salvata) senza cambiare una sola predizione, e gli MD5 nel README
sono un'affermazione che vale la pena non invalidare per una pulizia cosmetica.

## Conclusione operativa

L'accuratezza è esaurita. Il margine che resta è sugli **altri quattro criteri** di
valutazione (9-19 novembre): robustezza e generalizzazione, innovazione metodologica,
documentazione e riproducibilità, presentazione. Vedi il commento su dove spendere il tempo.
