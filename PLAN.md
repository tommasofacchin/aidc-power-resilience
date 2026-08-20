# Piano operativo — Huawei Tech Arena 2026, Topic 2 (Phase 1)

**Aggiornato:** 19 agosto 2026
**Deadline:** 24 agosto 2026, 23:59 CET → **5 giorni di lavoro**
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
| A | ogni giorno 00:00Z, dal **2025-08-30** al **2025-11-29** | +1h … +48h (48 step) | 240 | 92 |
| B | ogni 6h (00/06/12/18Z), dal **2025-08-31T18:00Z** al **2025-11-30T18:00Z** | +15m … +6h (24 step) | 120 | 365 |

Le prime emissioni partono **prima** del 1° settembre perché servono a coprire i `target_time` del primo giorno della finestra (le righe fuori finestra semplicemente non vengono valutate).

**Totale righe ≈ 65.900** (21.840 per A + 43.800 per B). Volume trascurabile.

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

Le guidelines dicono che il valore di riferimento è **fissato e mantenuto dagli organizzatori** — noi non lo conosciamo. Usiamo `MCC.csv` (max customer count per contea) che accompagna il rilascio EAGLE-I, e **lo dichiariamo esplicitamente nel report**. È il rischio residuo più concreto sul punteggio ed è fuori dal nostro controllo.

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

- [ ] Rilettura del report, controllo che ogni sezione richiesta ci sia
- [ ] Checklist di conformità §6
- [ ] **Consegnare entro il pomeriggio**, non alle 23:58

---

## 5. Rischi

| Rischio | Impatto | Mitigazione |
|---|---|---|
| Archivio IFS con buchi sulla finestra | **Bloccante** | Verificare il giorno 1. Fallback: run più vecchio disponibile + flag "stale forecast" come feature |
| Multi-coordinate non supportato su single-runs | Basso | 5× chiamate, comunque trascurabile |
| Rate limit più stretti del previsto | Medio | Caching aggressivo, download notturno in background |
| `total_customers` diverso da quello degli organizzatori | Medio, **fuori controllo** | Usare MCC.csv, dichiararlo nel report |
| EAGLE-I con lacune di copertura sulle contee scelte | Medio | Criterio di selezione contee include la continuità della copertura |
| Metrica di scoring ignota | Medio | Suite di metriche, non over-fittare su una sola |
| Tempo insufficiente per il modello "bello" | Alto (5 giorni) | Baseline consegnabile già il giorno 2; ogni giorno successivo migliora qualcosa di già valido |

**Principio guida:** avere sempre un pacchetto consegnabile. Meglio un LightGBM onesto, documentato e riproducibile che un'architettura ambiziosa incompleta il 24 sera.

---

## 6. Checklist di conformità pre-consegna

CSV:
- [ ] `fips_code` stringa a 5 cifre con zeri iniziali preservati (verificare aprendo il file come testo, **mai** con Excel)
- [ ] Tutti i timestamp ISO 8601 UTC con suffisso `Z`
- [ ] Task A: esattamente 48 righe per (issue_time, contea); Task B: esattamente 24
- [ ] Tutti i batch contengono **tutte e 5** le contee
- [ ] Frequenza minima rispettata: ≥1 batch/giorno per A, ≥1 batch/6h per B
- [ ] `predicted_x` ∈ [0,1], nessun NaN, nessuna notazione scientifica illeggibile
- [ ] Nessuna deduplicazione dei target sovrapposti
- [ ] Copertura completa della finestra 2025-09-01 → 2025-11-30

Report:
- [ ] 3–8 pagine, PDF
- [ ] Acquisizione e preprocessing: scelta fonti, valori mancanti, allineamento timestamp, matching spaziale
- [ ] Feature engineering e razionale
- [ ] Architettura del modello e strategia di training
- [ ] **Gestione dello sbilanciamento** (sezione esplicitamente richiesta)
- [ ] Risultati sperimentali e **ablation**
- [ ] **Dichiarazione di ogni fonte dati e relativa licenza** — EAGLE-I, Open-Meteo, ed eventuali altre. Le guidelines dicono testualmente che le omissioni non sono accettabili.
- [ ] Giustificazione della scelta delle 5 contee

Codice:
- [ ] Pipeline completa: acquisizione → preprocessing → feature → training → predizione
- [ ] `requirements.txt` con versioni pinnate
- [ ] README con configurazione ambiente, ordine di esecuzione, tempi attesi
- [ ] Testato da zero in ambiente pulito

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
