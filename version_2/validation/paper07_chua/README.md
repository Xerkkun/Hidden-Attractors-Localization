# Trazabilidad canónica del artículo 07: sistemas de Chua

Este directorio identifica la evidencia que debe conservarse para reproducir
los dos casos numéricos del artículo 07 y mantiene separados:

1. los **resultados reportados** por el trabajo propuesto;
2. los **controles bibliográficos** de Kuznetsov, Danca y Wu; y
3. la **exploración de parámetros y procedencia de semillas**.

El archivo [`manifest.json`](manifest.json) contiene los mismos datos en forma
estructurada. Todas las rutas de este documento son relativas a la raíz del
repositorio, salvo que un bloque indique que el directorio de trabajo es
`version_2`.

Los resultados extensos se generan en directorios `outputs/`, que están
excluidos del control de versiones. La proyección compacta y rastreable se
conserva dentro de la biblioteca en:

```text
version_2/validation/paper07_chua/evidence/
version_2/validation/chua_fractional_arctan_c590/*_rows.csv
version_2/validation/chua_fractional_arctan_c590/*_run_config.json
```

[`evidence_manifest.json`](evidence_manifest.json) fija el tamaño y la huella
SHA-256 de cada archivo. El comando
`python tools/sync_paper07_evidence.py --verify` valida el paquete sin depender
de los directorios ignorados.

## 1. Resultados reportados

### 1.1 Chua fraccionario no suave: cadena corregida

La cadena corregida emplea el sistema `chua-nonsmooth` con

```text
q = 0.9998
alpha = 8.4562
beta = 12.0732
gamma = 0.0052
m0 = -0.1768
m1 = -1.1468
```

Estos parámetros coinciden con el control físico de Danca, pero la semilla del
trabajo propuesto no se atribuye a ese artículo. Se obtiene mediante la función
descriptiva sesgada y se transporta al sistema original con continuación
Caputo de memoria completa.

La exploración recorre nueve pares `(m1,m0)`. Para cada par se usan
`5 x 9 x 6 = 270` inicios del solucionador en los dominios de amplitud, sesgo y
frecuencia; por tanto, la búsqueda completa contiene 2,430 inicios. Las raíces
se filtran por convergencia, residuo menor que `1e-4`, deduplicación a `1e-3` y
un máximo de dos ramas por par. En el caso reportado se elige la raíz positiva
con `c > 0.05` y residuo mínimo:

```text
A = 4.5778827210231485
c = 2.776397003273905
omega = 2.04028605107949
residuo = 4.549007847495242e-16
semilla = [7.354279724297053, 0.29096877878449423, -9.316054818665608]
```

La continuación usa 21 niveles
`eta = 0, 0.05, ..., 1`, conserva la historia causal entre niveles y produce el
punto final

```text
[0.8223107098499502, 1.4652727039952436, 1.2923082502480394].
```

La trayectoria de referencia se integra con ABM--PECE, derivada de Caputo,
memoria completa, `h=0.01`, horizonte `300` y descarte transitorio `100`.
La clasificación dinámica almacenada es `regular_periodic_rejected`; por ello,
la evidencia reportada corresponde a un conjunto atrayente regular y a su
geometría de cuenca finitamente muestreada, no a una certificación independiente
de caos.

La corrida completa se genera en `outputs/paper07_nonsmooth_corrected/`. Su
proyección canónica se conserva en
`version_2/validation/paper07_chua/evidence/nonsmooth_corrected/`:

- `candidate_and_reference.json`;
- `continuation_stages.csv`;
- `numerical_contract.json`;
- `probe_runs.csv` y `probe_summary.csv`;
- `target_reproduction_runs.csv`;
- `result.json`.

Las trayectorias completas y las figuras se regeneran con el flujo declarado
y no se duplican en el paquete compacto.

El sondeo esférico base de 675 condiciones iniciales queda como control corto.
Las 36 perturbaciones alrededor del extremo de continuación reproducen la nube
objetivo y verifican la atracción local de ese conjunto.

### 1.2 Sondeo volumétrico limpio con corte causal

La corrida extendida se genera en

```text
outputs/paper07_nonsmooth_corrected/extended_first_contact_clean/
```

Los archivos de evidencia se sincronizan bajo
`version_2/validation/paper07_chua/evidence/nonsmooth_corrected/extended_first_contact_clean/`.
El checkpoint y las figuras son salidas de ejecución regenerables y no forman
parte de la proyección rastreable.

El muestreo usa bolas uniformes en volumen, `PCG64`, semilla base `42` y la
regla

```text
seed = 42 + 100*indice_equilibrio + 10*indice_radio.
```

Cada radio se procesa por completo alrededor de `E0`, `E+` y `E-` antes de
avanzar al siguiente. Cuando aparece el primer radio con uno o más contactos,
se terminan todas las pruebas previstas en ese mismo radio y se omiten los
radios mayores.

El resultado limpio es:

| magnitud | valor |
| --- | ---: |
| pruebas realizadas | 17,400 |
| pruebas por equilibrio | 5,800 |
| fallos numéricos | 0 |
| contactos hasta `r=0.1` | 0 de 13,800 |
| primer radio con contactos | `r=0.3` |
| contactos en `r=0.3` | 37 de 3,600 |
| equilibrio estable | 5,827 |
| otro conjunto atrayente | 5,826 |
| divergencia | 5,710 |

Los siete radios locales hasta `r=0.01` contienen 7,200 pruebas y cero
contactos. Los radios exteriores `0.03`, `0.1` y `0.3` contienen 10,200 pruebas
y 37 contactos. El radio máximo presente físicamente en los resultados
declarados es `0.3`; no existen filas posteriores al corte.

Los archivos esenciales son:

- `extended_numerical_contract.json`;
- `extended_probe_plan.csv`;
- `extended_probe_runs.csv`;
- `extended_probe_summary.csv`;
- `extended_result.json`;
- `target_cloud_nn_sample.csv`.

Las huellas que enlazan plan, contrato y nube objetivo son:

```text
plan_sha256 =
6c07793e653fe116b3878fb0674ac778e313763781638d4eeed47d92e147be93

target_cloud_sample_sha256 =
be399e79cd6bcbc3f24edd6d6bc2f6955c413fec801d39a8acd2446ca1138fa3

contract_sha256 =
ed11d92fe56e0bbfeedfda84dcb3a27738871b6e171cfaa0c7974ce87e24e44c
```

### 1.3 Chua fraccionario suave: resultado c590

El resultado suave reportado es el candidato
`chua_arctan_c590_q09999_seed9`, con `q=0.9999`, Caputo ABM y memoria completa.
Su reconstrucción compacta y las filas de validación están en:

```text
version_2/validation/paper07_chua/evidence/c590_reconstruction/
version_2/validation/chua_fractional_arctan_c590/
```

En 8,400 sondeos sobre superficies esféricas no aparecen contactos hasta
`r=0.3`: 8,396 trayectorias completan el horizonte y cuatro llegan
anticipadamente a un equilibrio. La auditoría macroscópica contiene 5,100
sondeos y 610 contactos: en `r=1` hay 2,400 trayectorias completas y 22
contactos; en `r=2` hay 2,569 trayectorias completas, 131 que alcanzan el umbral
de divergencia declarado y 588 contactos. El resultado se reporta como
evidencia de ocultedad limitada a las vecindades muestreadas; los contactos
macroscópicos describen la geometría extendida de la cuenca.

## 2. Controles bibliográficos

Los controles bibliográficos verifican qué parte de cada referencia puede
reproducirse con la información publicada. No se mezclan con las semillas ni
con los resultados del método propuesto.

### 2.1 Kuznetsov et al. 2017

- Caso: Chua entero no suave, `q=1`.
- Estado: reproducción ejecutable del caso 18, incluida la semilla de función
  descriptiva y la trayectoria de referencia.
- Función: control entero del método de localización; no valida por sí mismo la
  extensión fraccionaria.
- Evidencia:
  `version_2/validation/published_cases/kuznetsov2017_chua_integer.yaml`,
  `version_2/validation/reference_cases/chua_integer_q1/` y
  `version_2/validation/references/kuznetsov2017_expected.json`.

### 2.2 Danca 2017

- Caso: Chua fraccionario no suave, `q=0.9998`.
- Estado: implementación bibliográfica parcial.
- Se verifican las ecuaciones, parámetros, equilibrios, estabilidad de
  Matignon y el contrato ABM/Caputo.
- La referencia no proporciona la condición inicial exacta del atractor,
  `omega0`, `k`, `a0`, `seed_plus`, `seed_minus` ni exponentes de Lyapunov
  suficientes para reconstruir independientemente su atractor reportado.
- Evidencia:
  `version_2/validation/published_cases/danca2017_chua_fractional_saturation.yaml`,
  `version_2/configs/examples/chua_nonsmooth_exact_danca_non_reproducible.yaml`
  y `version_2/validation/published_reference_coverage.json`.

La cadena no suave corregida conserva los parámetros físicos de este control,
pero su búsqueda sesgada, semilla, continuación con memoria y sondeos pertenecen
al método propuesto.

### 2.3 Wu et al. 2023

- Caso bibliográfico: Chua fraccionario con arctangente, `q=0.99`.
- Estado: reproducción bibliográfica parcial de álgebra, equilibrios,
  representación de Lur'e, condiciones iniciales reportadas y recurrencia ADM
  local.
- Las trayectorias desde las condiciones iniciales reportadas se clasifican
  como periódicas bajo ese contrato.
- La recurrencia ADM local no equivale a ABM Caputo de memoria completa.
- Evidencia:
  `version_2/validation/published_cases/wu2023_chua_fractional_arctan.yaml`,
  `version_2/validation/reference_cases/fractional_chua_arctan_wu2023/` y
  `version_2/examples/chua_arctan_wu2023/reproducibility.yaml`.

El candidato c590 usa `q=0.9999`, otra selección de parámetros y una semilla
propia; no es una reproducción del atractor de Wu.

## 3. Exploración y procedencia exacta de c590

`version_2/tools/reconstruct_c590_search_provenance.py` reconstruye exactamente
los bancos pseudoaleatorios de parámetros y semillas y verifica los valores
seleccionados con tolerancia absoluta `5e-15`.
`version_2/tools/rerun_c590_discovery.py` ejecuta de nuevo las 3,400 trayectorias
de cribado, la auditoría variacional y el refinamiento Caputo completo.

1. **Exploración global:** 2,400 casos, `PCG64`, semilla `2026062304`. Se
   selecciona el índice de base cero `1731`:

   ```text
   parametros =
   [18.485729510399246, 21.96695211004372, 0.005274610555721088,
    0.02629749304876286, -3.208777924503481, 1.949261203458447]

   semilla =
   [7.079605327144233, 0.5079327670652387, -14.49039352666749]
   ```

2. **Exploración local:** 1,000 perturbaciones alrededor del caso `1731`,
   `PCG64`, semilla `2026062305`. Se selecciona el índice de base cero `590`:

   ```text
   parametros c590 =
   [21.849356906616716, 19.081840840860202, 0.007378011979156531,
    0.04228979343578827, -3.3367815123026694, 1.7984259332820332]

   semilla entera c590 =
   [7.6733768928786095, 0.5079327670652387, -14.49039352666749]
   ```

3. **Refinamiento fraccionario:** se prueban
   `q = [0.9995, 0.9998, 0.9999, 0.99995]` como problemas de valor inicial
   independientes. Cada corrida usa ABM con memoria Caputo completa desde su
   propio tiempo inicial; la historia no se transfiere entre órdenes ni entre
   reinicios. Para `q=0.9999` se extraen 16 estados mediante
   `linspace(searchsorted(t,150),len(t)-1,16,dtype=int)`, con índices
   `[30001,32000,...,60000]`. El primer tiempo real es
   `150.00499999993608`. El índice de base cero `9`, en
   `t=239.99999999985423`, produce `seed9`:

   ```text
   [5.864244979081692, 1.5847111486491057, 3.2155806477633915].
   ```

   Seis reinicios superan la criba corta en los tres pasos. En la auditoría
   larga, `seed9` y `seed13` permanecen acotadas; `seed9` se selecciona porque
   además satisface `K_median > 0.8` en los dos pasos refinados
   `h=0.0025` y `h=0.005`.

La procedencia estructurada se conserva en
`version_2/validation/paper07_chua/evidence/c590_reconstruction/search_provenance.json`.
La selección es evidencia de exploración y procedencia; la clasificación
dinámica y los sondeos de cuenca se validan en etapas independientes.

## 4. Mandatos exactos de reproducción

### 4.1 Procedencia c590

Desde `version_2`:

```bash
python tools/reconstruct_c590_search_provenance.py --check-only
python tools/rerun_c590_discovery.py --stage all --dry-run
python -m pytest -q \
  tests/test_c590_search_provenance.py \
  tests/test_rerun_c590_discovery.py
```

Para regenerar el archivo canónico:

```bash
python tools/reconstruct_c590_search_provenance.py
```

Para repetir la ruta completa de descubrimiento en un directorio nuevo:

```bash
python tools/rerun_c590_discovery.py \
  --stage all \
  --execute \
  --output-dir outputs/reproductions/c590_discovery
```

Para repetir los sondeos c590 configurados:

```bash
python examples/chua_arctan_wu2023/run_example.py --steps verification
python tools/summarize_c590_hiddenness.py
```

Después de completar las corridas de c590 y del caso no suave, la proyección
canónica se actualiza y verifica desde `version_2` con:

```bash
python tools/sync_paper07_evidence.py --sync
python tools/sync_paper07_evidence.py --verify
python tools/sync_paper07_evidence.py --verify --verify-sources
python -m pytest -q tests/test_sync_paper07_evidence.py
```

La sincronización comprueba antes de copiar que la criba local reproduzca la
asociación numérica archivada completa: 148 filas no triviales, divididas en
87 inconclusas y 61 periódicas, con `c590` en el rango 10.

### 4.2 Cadena no suave y sondeo limpio

Para una repetición fresca sin tocar la evidencia canónica, desde la raíz del
repositorio:

```bash
python version_2/tools/rerun_paper07_nonsmooth_hiddenness.py \
  --output version_2/outputs/paper07_nonsmooth_reproduction \
  --run-extended \
  --extended-dirname extended_first_contact_clean \
  --workers 12
```

Si una ejecución se interrumpe después de haber terminado la referencia y el
sondeo base:

```bash
python version_2/tools/rerun_paper07_nonsmooth_hiddenness.py \
  --output version_2/outputs/paper07_nonsmooth_reproduction \
  --reuse-reference \
  --reuse-probes \
  --run-extended \
  --resume-extended \
  --extended-dirname extended_first_contact_clean \
  --workers 12
```

La estructura del plan y las semillas se comprueba con:

```bash
cd version_2
python -m pytest -q tests/test_biased_chua_extended_stop_policy.py
```

### 4.3 Controles bibliográficos

Desde `version_2`:

```bash
python examples/chua_integer_lure_reference/run_example.py
python -m hidden_attractors.cli.main published \
  danca-abm-sphere-controls \
  --output validation/outputs/published_cases/danca2017_chua_fractional_saturation
python examples/chua_arctan_wu2023/run_example.py \
  --steps published \
  --run-published-trajectories
python -m pytest -q \
  tests/test_published_case_reproduction.py \
  tests/test_published_continuation_comparison.py \
  tests/test_published_validation_coverage.py \
  tests/test_published_reference_claims_are_conservative.py
```

## 5. Política de conservación

Se conservan como evidencia canónica:

- la proyección compacta de la cadena base y `extended_first_contact_clean`;
- los bancos, cribados, selecciones y resúmenes Caputo que reconstruyen c590;
- las seis tablas completas de sondeos c590 y sus contratos de ejecución;
- los controles publicados y las matrices de trazabilidad bibliográfica; y
- las pruebas unitarias de procedencia, reconstrucción y corte por primer
  contacto.

Se retiraron de la librería las siguientes salidas sustituidas o
exploratorias:

- `outputs/paper07_nonsmooth_corrected/extended/`, ejecución retrospectiva que
  contiene filas posteriores al radio de corte;
- `outputs/paper07_nonsmooth_corrected/extended_contract_smoke/`;
- `outputs/example_chua_nonsmooth_biased_hidden_attractor/step3_hiddenness/`
  y `step5_summary/`, cuyas ejecuciones antiguas tenían contratos parciales;
- `outputs/chua_fractional_zero_one/`, ejemplo diagnóstico breve;
- `version_2/outputs/arctan_hidden_candidate_search/wu_q0999_long_20260623/`;
- la continuación DF centrada que llegó a una rama regular distinta de c590;
- las matrices preliminares de 120, 216 y 288 sondeos, sustituidas por la
  secuencia promovida `scaled/r003/r010/r030/r100/r200/latest`;
- el experimento no suave sin generador autocontenido y los scripts ad hoc que
  mezclaban matrices preliminares;
- las salidas Machado incompletas, binarios compilados de referencia y
  artefactos temporales de pruebas.

Los pasos 1 y 2 del ejemplo no suave permanecen porque contienen la búsqueda y
continuación que originan el sistema reportado. Los paquetes exploratorios
materiales se movieron a
`../.quarantine_hidden_attractors_noncanonical_20260728/`, fuera de la
biblioteca y de forma recuperable; los cachés y scripts pequeños regenerables
se retiraron directamente.
