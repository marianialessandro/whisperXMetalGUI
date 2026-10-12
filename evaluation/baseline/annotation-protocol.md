# Protocollo delle annotazioni indipendenti

Stato: implementato il validatore; nessuna annotazione naturale consegnata o certificata.

Utilizzare solo audio pubblico con licenza verificata e identificato da SHA-256. Il corpus non legge registrazioni o cataloghi dell'app. Assegnare split calibration/evaluation prima dell'oracle: recording_id, source_group e identità anonime speaker devono restare disgiunti. Un ritaglio mantiene identità del recording originale e offset; concatenazioni e mix restano derived_audio.

Per ciascun file consegnare JSON UTF-8 con audio_sha256, provenance `independent_human`, transcript originale, words (`word`, `start`, `end`), turns (`speaker`, `start`, `end`) e uem (intervalli valutabili). I confini sono secondi finiti, non negativi, con start < end, interamente nella UEM. Le sovrapposizioni fra turns restano presenti: non convertirle in etichette esclusive. Gli speaker hanno identificativi anonimi stabili nel recording, distinti fra recording quando la persona non è nota.

Il revisore ascolta e annota i confini; una seconda passata rivede parole, cambi voce e overlap, anche con lo stesso annotatore. `reviewers` contiene un'identità anonima per passata, almeno due voci; `reviewed` diventa true solo dopo la revisione. `uncertainties` contiene gli intervalli incerti e diventa una lista vuota soltanto dopo adjudication. L'oracle può suggerire una trascrizione, ma nessun suo output viene accettato come ground truth senza revisione indipendente. Il validatore verifica struttura e attestazioni, non può dimostrare da solo che l'ascolto sia realmente avvenuto.

L'importazione richiede checksum del JSON e dell'audio; non modifica l'originale e rifiuta destinazioni già esistenti. Transcript, tempi parola, RTTM derivabile dai turns e UEM devono riferirsi allo stesso audio. Annotazioni mancanti, hash errati o incertezza irrisolta impediscono il full gate. FLEURS di parlato letto può coprire ASR e lingue; non sostituisce conversazioni naturali lunghe o overlap italiani/inglesi.
